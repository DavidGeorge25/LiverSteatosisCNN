"""Re-deriving per-tile scores from stored cells under one ranking config.

Runs under plain python (no pytest needed):

    python tests/test_rescore.py

The synthetic half proves the semantics. The half that matters runs against the
real cells.csv files under `bal_chunks/` and `bal_local/` when they are present
-- both are gitignored, so the tests skip cleanly on a fresh clone -- and
asserts that re-applying a run's OWN config reproduces the score column that run
wrote, cell for cell. That is the property that makes the re-derivation
trustworthy on the slides where it changes something: if it reproduces the
stored column exactly wherever the config is unchanged, then the differences
where the config differs are the config and nothing else.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mashpath.features.ballooning.config import RankingConfig  # noqa: E402
from mashpath.review.rescore import (NEEDED, cell_scores, restore_frame,  # noqa: E402
                                     tile_scores)

CFG = RankingConfig(min_area_ratio=2.0, max_area_ratio=4.0, min_solidity=0.8,
                    max_eccentricity=0.85, max_cyto_gray_mean=205.0,
                    require_nucleus=True)


def _cells(rows: list[dict]) -> Path:
    """A cells.csv with the columns the vetoes read. Defaults pass everything."""
    base = {"tile_x": 0, "tile_y": 0, "raw_score": 1.0,
            "territory_area_um2": 900.0, "local_median_territory_um2": 300.0,
            "cyto_gray_mean": 150.0, "nucleus_area_um2": 30.0,
            "is_hepatocyte": True, "solidity": 0.9, "eccentricity": 0.5}
    df = pd.DataFrame([{**base, **r} for r in rows])[NEEDED]
    path = Path(tempfile.mkdtemp()) / "cells.csv"
    df.to_csv(path, index=False)
    return path


def test_every_veto_removes_the_cell_it_names():
    path = _cells([
        {"raw_score": 1.0},                                   # passes
        {"raw_score": 2.0, "is_hepatocyte": False},           # not a hepatocyte
        {"raw_score": 3.0, "territory_area_um2": 400.0},      # not enlarged (1.3x)
        {"raw_score": 4.0, "solidity": 0.5},                  # not convex
        {"raw_score": 5.0, "eccentricity": 0.95},             # too elongated
        {"raw_score": 6.0, "territory_area_um2": 1800.0},     # merged (6x)
        {"raw_score": 7.0, "cyto_gray_mean": 230.0},          # a fat void
        {"raw_score": 8.0, "nucleus_area_um2": 0.0},          # no nucleus
    ])
    got = cell_scores(path, CFG)["score"].to_numpy()
    assert got[0] == 1.0, f"the passing cell lost its score: {got[0]}"
    assert np.isnan(got[1:]).all(), \
        f"a vetoed cell kept a score: {got[1:]}"


def test_the_tile_score_is_the_max_over_surviving_cells_only():
    path = _cells([
        {"tile_x": 512, "raw_score": 1.0},
        {"tile_x": 512, "raw_score": 9.0, "solidity": 0.1},   # highest, vetoed
        {"tile_x": 512, "raw_score": 3.0},
    ])
    t = tile_scores(path, CFG)
    assert len(t) == 1, f"expected one tile, got {len(t)}"
    assert t["score"].iloc[0] == 3.0, \
        f"a vetoed cell set the tile score: {t['score'].iloc[0]}"
    assert t["n_scored_cells"].iloc[0] == 2


def test_a_tile_the_detector_emptied_stays_in_the_frame():
    """"Looked and proposed nothing" is a different state from "never looked".

    If the emptied tile vanished, it would drop out of the frame entirely and
    the slide would silently look like it had fewer tiles rather than more
    negative ones -- and every within-slide percentile computed after that
    would be a percentile of a different population.
    """
    path = _cells([
        {"tile_x": 0, "raw_score": 5.0},
        {"tile_x": 512, "raw_score": 9.0, "is_hepatocyte": False},
    ])
    t = tile_scores(path, CFG).sort_values("x").reset_index(drop=True)
    assert len(t) == 2, f"the emptied tile was dropped: {t}"
    assert np.isnan(t["score"].iloc[1]), f"an emptied tile kept a score: {t}"
    assert t["n_scored_cells"].iloc[1] == 0


def test_a_stricter_config_can_only_remove():
    """Vetoes are necessary conditions, so tightening one cannot add a cell."""
    path = _cells([{"raw_score": float(i), "territory_area_um2": 300.0 * r}
                   for i, r in enumerate([1.1, 1.5, 2.5, 3.5, 5.0])])
    loose = cell_scores(path, RankingConfig(min_area_ratio=1.3, max_area_ratio=6.0))
    tight = cell_scores(path, CFG)
    kept_loose = set(np.flatnonzero(loose["score"].notna()))
    kept_tight = set(np.flatnonzero(tight["score"].notna()))
    assert kept_tight <= kept_loose, \
        f"the stricter config kept cells the looser one vetoed: {kept_tight - kept_loose}"
    assert kept_tight != kept_loose, "the two configs kept the same cells"


def test_restore_frame_leaves_a_slide_with_no_stored_cells_alone():
    frame = pd.DataFrame({
        "slide": ["A", "A", "B", "B"], "x": [0, 512, 0, 512], "y": [0, 0, 0, 0],
        "score": [7.0, 8.0, np.nan, np.nan], "n_scored_cells": [1, 1, 0, 0],
        "scored": [True, True, False, False], "score_pct": [50.0, 100.0, np.nan, np.nan],
    })
    path = _cells([{"tile_x": 0, "raw_score": 2.0}, {"tile_x": 512, "raw_score": 4.0}])
    out = restore_frame(frame, {"A": path}, CFG, verbose=False)
    assert list(out.loc[out.slide == "A", "score"]) == [2.0, 4.0], \
        f"slide A was not rescored: {out}"
    assert out.loc[out.slide == "B", "score"].isna().all(), \
        "slide B has no stored cells and must not acquire a score"
    assert not out.loc[out.slide == "B", "scored"].any()
    # score_pct is a within-slide rank of the OLD column and has to move with it.
    assert list(out.loc[out.slide == "A", "score_pct"]) == [50.0, 100.0]


def test_restore_frame_keeps_the_row_order_it_was_given():
    """The frame is merged back by position elsewhere; a reordered return would
    silently attach every score to the wrong tile."""
    frame = pd.DataFrame({
        "slide": ["B", "A", "B", "A"], "x": [0, 0, 512, 512], "y": 0,
        "score": np.nan, "n_scored_cells": 0, "scored": True, "score_pct": np.nan,
    })
    path = _cells([{"tile_x": 0, "raw_score": 2.0}, {"tile_x": 512, "raw_score": 4.0}])
    out = restore_frame(frame, {"A": path, "B": path}, CFG, verbose=False)
    assert list(out.index) == list(frame.index)
    assert list(out["slide"]) == ["B", "A", "B", "A"]
    assert list(out["x"]) == [0, 0, 512, 512]
    assert list(out["score"]) == [2.0, 2.0, 4.0, 4.0]


# ---- against the real runs ------------------------------------------------

def _stored_runs():
    out = []
    for d in ("bal_chunks", "bal_local"):
        for cfg_path in sorted((ROOT / d).glob("*/config_used.yaml")):
            cells = cfg_path.parent / "ballooning" / "cells.csv"
            if cells.exists():
                out.append((cfg_path, cells))
    return out


def test_reapplying_a_runs_own_config_reproduces_the_score_it_wrote():
    runs = _stored_runs()
    if not runs:
        print("    (skipped: no bal_chunks/ or bal_local/ output on this machine)")
        return
    checked = boundary = 0
    for cfg_path, cells in runs:
        r = yaml.safe_load(cfg_path.read_text())["ballooning"]["ranking"]
        cfg = RankingConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                               for k, v in r.items()})
        got = cell_scores(cells, cfg)
        stored = pd.read_csv(cells, usecols=["score"])["score"].to_numpy(float)
        assert len(got) == len(stored), f"{cells}: {len(got)} vs {len(stored)} cells"
        differs = np.flatnonzero(
            np.nan_to_num(stored, nan=-9e99) != np.nan_to_num(
                got["score"].to_numpy(float), nan=-9e99))
        for i in differs:
            # The only tolerated difference: a cell whose area ratio lands
            # EXACTLY on a threshold once it has been through the CSV. The
            # vetoes compare with a strict <, so the last binary digit decides,
            # and cells.csv stores the two areas rather than their quotient.
            ratio = (got["territory_area_um2"].iloc[i]
                     / got["local_median_territory_um2"].iloc[i])
            assert ratio in (cfg.min_area_ratio, cfg.max_area_ratio), (
                f"{cells} row {i}: stored {stored[i]}, re-derived "
                f"{got['score'].iloc[i]}, area ratio {ratio!r}")
            boundary += 1
        checked += 1
    print(f"    ({checked} runs reproduced exactly, {boundary} cell(s) on a "
          f"threshold boundary)")


def test_the_stored_cells_carry_everything_the_vetoes_need():
    runs = _stored_runs()
    if not runs:
        print("    (skipped: no bal_chunks/ or bal_local/ output on this machine)")
        return
    for _, cells in runs[:3]:
        cols = set(pd.read_csv(cells, nrows=1).columns)
        missing = [c for c in NEEDED if c not in cols]
        assert not missing, f"{cells} is missing {missing}"


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
