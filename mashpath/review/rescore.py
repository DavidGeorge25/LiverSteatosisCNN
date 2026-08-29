"""Per-tile detector scores re-derived from stored cells, under ONE config.

`tileset.tile_scores` reads the `score` column a run wrote, which is the right
thing when building a package and the wrong thing when comparing slides: that
column carries whatever ranking vetoes were in force the day the slide was
scored. The stored runs do not agree. 48 slides were scored under the vetoes a
pathologist gave us on 2026-08-16 (`min_area_ratio` 2.0, `max_area_ratio` 4.0)
and 23 under the ones they replaced (1.3 / 6.0), and the two groups are not
distributed evenly across staining batches -- so a batch difference measured on
the stored column is partly a difference in when the slide happened to be run.

`raw_score` is the weighted z-sum before any veto and does not depend on the
ranking config at all, and every input `apply_vetoes` needs is a column of
cells.csv. So the vetoes can simply be applied again, to every slide, under one
config. This calls the same `apply_vetoes` the pipeline calls rather than
restating the rules: a second copy of the veto logic would drift from the first
and the drift would look like biology.

Reproduces the stored column exactly on slides whose run used the config being
applied -- `tests/test_rescore.py` asserts that against real cells.csv files,
which is what makes the re-derivation trustworthy on the slides where it
changes something.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..features.ballooning.config import RankingConfig
from ..features.ballooning.rank import apply_vetoes

# Everything apply_vetoes needs, plus the tile a cell sits in.
NEEDED = ["tile_x", "tile_y", "raw_score", "territory_area_um2",
          "local_median_territory_um2", "cyto_gray_mean", "nucleus_area_um2",
          "is_hepatocyte", "solidity", "eccentricity"]


def cell_scores(cells_csv: str | Path, cfg: RankingConfig) -> pd.DataFrame:
    """cells.csv with a `score` column re-derived under `cfg`.

    NaN where a cell was vetoed, exactly as the pipeline writes it.
    """
    df = pd.read_csv(cells_csv, usecols=NEEDED)
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise KeyError(f"{cells_csv}: cells.csv is missing {missing}; it "
                       "predates the columns the vetoes are computed from")
    score, _ = apply_vetoes(
        df["raw_score"].to_numpy(dtype=float),
        df["territory_area_um2"].to_numpy(dtype=float),
        df["local_median_territory_um2"].to_numpy(dtype=float),
        df["cyto_gray_mean"].to_numpy(dtype=float),
        df["nucleus_area_um2"].to_numpy(dtype=float) > 0,
        df["is_hepatocyte"].to_numpy(dtype=bool),
        df["solidity"].to_numpy(dtype=float),
        df["eccentricity"].to_numpy(dtype=float),
        cfg,
    )
    df["score"] = score
    return df


def tile_scores(cells_csv: str | Path, cfg: RankingConfig) -> pd.DataFrame:
    """Per tile: max score over veto-passing cells, under `cfg`.

    Same shape and same semantics as `tileset.tile_scores` -- a tile the
    detector examined and found nothing in keeps a row with a NaN score and
    `n_scored_cells` 0, because "looked and proposed nothing" is a different
    state from "was never looked at" and the two must not collapse.
    """
    df = cell_scores(cells_csv, cfg)
    scored = df.dropna(subset=["score"])
    g = scored.groupby(["tile_x", "tile_y"])["score"]
    out = pd.DataFrame({"score": g.max(), "n_scored_cells": g.size()}).reset_index()
    covered = df[["tile_x", "tile_y"]].drop_duplicates()
    out = covered.merge(out, on=["tile_x", "tile_y"], how="left")
    out["n_scored_cells"] = out["n_scored_cells"].fillna(0).astype(int)
    return out.rename(columns={"tile_x": "x", "tile_y": "y"})


def restore_frame(frame: pd.DataFrame, sources, cfg: RankingConfig,
                  verbose: bool = True) -> pd.DataFrame:
    """Replace `score` / `n_scored_cells` in a tile frame, one config for all.

    `sources` is anything mapping slide name -> cells.csv path, including the
    `SlideSource` list `discover_sources` returns. Slides with no stored cells
    keep their NaN score and are reported, not dropped: a frame that quietly
    lost its unscored slides would understate how much of the collection the
    detector has never been run on.
    """
    paths = {s.name: s.cells_csv for s in sources
             if getattr(s, "cells_csv", None)} if not isinstance(sources, dict) \
        else dict(sources)

    out, seen, missing = [], set(), []
    for name, grp in frame.groupby("slide", sort=False):
        cells = paths.get(name)
        if cells is None or not Path(cells).exists():
            if bool(grp["scored"].iloc[0]):
                missing.append(name)
            out.append(grp)
            continue
        sc = tile_scores(cells, cfg)
        merged = grp.drop(columns=["score", "n_scored_cells"]).merge(
            sc, on=["x", "y"], how="left")
        merged.index = grp.index
        merged["scored"] = merged["n_scored_cells"].notna()
        merged["n_scored_cells"] = merged["n_scored_cells"].fillna(0).astype(int)
        out.append(merged)
        seen.add(name)

    res = pd.concat(out).loc[frame.index]
    # score_pct is a within-slide rank of the old column; recompute or it lies.
    filled = res["score"].fillna(-np.inf)
    pct = pd.Series(np.nan, index=res.index)
    for name, grp in res.groupby("slide", sort=False):
        if name not in seen:
            continue
        v = filled.loc[grp.index]
        p = pd.Series(0.0, index=grp.index)
        finite = np.isfinite(v)
        if finite.any():
            p[finite] = v[finite].rank(pct=True) * 100.0
        pct.loc[grp.index] = p.round(2)
    res["score_pct"] = pct
    if verbose:
        print(f"rescored {len(seen)} slide(s) under min_area_ratio="
              f"{cfg.min_area_ratio}, max_area_ratio={cfg.max_area_ratio}")
        for m in missing:
            print(f"  no cells.csv for {m}: score left as it was")
    return res
