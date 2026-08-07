"""Random-tile cohort survey and parameter sweep.

A full run reads every tissue tile on a slide (~2,500-4,000 tiles, minutes per
slide). That is the right thing for producing pseudo-labels and the wrong thing
for answering "does this cohort have fat in it" or "which threshold separates
positives from negatives". Both of those are estimation problems, and a random
sample of a few dozen tiles per slide answers them in seconds.

Two entry points:

`survey`  one parameter set, N random tiles per slide -> per-slide fat fractions.
`sweep`   a grid of parameters over the same sampled tiles, scored on how well
          it separates a known-positive cohort from a known-negative one.

Sampling is seeded and the seed is derived from the slide name, so a slide's
sample is stable across runs and independent of which other slides are in the
folder.
"""

from __future__ import annotations

import random
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import Config, FatConfig
from .fat import Tally, extract_components, tally
from .slide import Slide
from .tiles import Tile
from .tissue import TissueMask, detect_tissue


def _slide_seed(name: str, base: int) -> int:
    """Per-slide seed: stable for a given (slide, base), unrelated across slides."""
    return zlib.crc32(name.encode()) ^ (base & 0xFFFFFFFF)


def candidate_tiles(slide: Slide, tissue: TissueMask, cfg) -> list[Tile]:
    """Every tile position passing the tissue filter. Reads no level-0 pixels."""
    w, h = slide.dimensions
    size, stride = cfg.tile_size, cfg.stride
    out: list[Tile] = []
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            frac = tissue.tissue_fraction(x, y, size, size)
            if frac >= cfg.min_tissue_fraction:
                out.append(Tile(x=x, y=y, size=size, tissue_fraction=frac))
    return out


def sample_tiles(
    slide: Slide, tissue: TissueMask, cfg, n: int, seed: int
) -> tuple[list[Tile], int]:
    """`n` tiles drawn uniformly from the candidates (all of them if n >= total).

    Uniform over tile positions, so the estimate is unbiased for the tissue the
    tiling stage would actually process -- the same population a full run sees.
    """
    cands = candidate_tiles(slide, tissue, cfg)
    if not cands:
        return [], 0
    if n <= 0 or n >= len(cands):
        return cands, len(cands)
    rng = random.Random(seed)
    return rng.sample(cands, n), len(cands)


# --------------------------------------------------------------------------
# survey: one parameter set
# --------------------------------------------------------------------------


def survey_slide(
    path: str | Path, cfg: Config, n_tiles: int, seed: int
) -> dict[str, Any]:
    """Sample `n_tiles` random tiles and report slide-level fat estimates."""
    path = Path(path)
    row: dict[str, Any] = {"slide": path.stem}
    t0 = time.time()
    slide = None
    try:
        # Opening is inside the guard: a truncated file raises here, and one
        # bad slide must not take down the whole cohort run.
        slide = Slide(str(path))
        tissue = detect_tissue(slide, cfg.tissue)
        tiles, n_cand = sample_tiles(
            slide, tissue, cfg.tiling, n_tiles, _slide_seed(path.stem, seed)
        )
        if not tiles:
            row["error"] = "no tiles passed the tissue fraction threshold"
            return row

        um2_px = slide.um2_per_pixel(cfg.tiling.level)
        agg = Tally()
        for tile in tiles:
            rgb = tile.read(slide, cfg.tiling.level)
            tmask = tissue.tile_mask(tile.x, tile.y, tile.size, tile.size)
            agg += tally(extract_components(rgb, tmask, cfg.fat, um2_px), cfg.fat)

        row.update(
            tissue_mm2=tissue.area_mm2,
            components=tissue.component_count,
            otsu=tissue.threshold,
            cand_tiles=n_cand,
            sampled=len(tiles),
            sampled_tissue_mm2=agg.tissue_area_um2 / 1e6,
            macro_pct=agg.macro_fraction * 100,
            micro_pct=agg.micro_fraction * 100,
            total_pct=agg.total_fraction * 100,
            macro_n=agg.macro_count,
            micro_n=agg.micro_count,
            macro_n_per_tile=agg.macro_count / len(tiles),
            micro_n_per_tile=agg.micro_count / len(tiles),
            macro_mean_um2=(
                agg.macro_area_um2 / agg.macro_count if agg.macro_count else 0.0
            ),
            border_pct_of_fat=(
                agg.border_area_um2 / (agg.macro_area_um2 + agg.micro_area_um2) * 100
                if (agg.macro_area_um2 + agg.micro_area_um2) > 0 else 0.0
            ),
            seconds=time.time() - t0,
        )
        return row
    except Exception as exc:  # one bad slide must not kill the cohort
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    finally:
        if slide is not None:
            slide.close()


# --------------------------------------------------------------------------
# sweep: a grid of parameters over the same sampled tiles
# --------------------------------------------------------------------------


def sweep_slide(
    path: str | Path,
    cfg: Config,
    n_tiles: int,
    seed: int,
    whites: Sequence[int],
    axes: dict[str, Sequence[Any]],
) -> list[dict[str, Any]]:
    """Evaluate a parameter grid on one slide's sampled tiles.

    `whites` is separated from the other axes because `white_threshold` is the
    only parameter that changes what gets *segmented*; everything in `axes` is
    a filter applied to already-measured components. So each tile is read once
    and thresholded once per white value, and the rest of the grid costs
    almost nothing. Total work scales with len(whites), not with grid size.

    `axes` maps FatConfig field names to the values to try, e.g.
    {"circularity_min": [0.6, 0.65], "min_area_um2": [20, 60]}.
    """
    path = Path(path)
    slide = None
    try:
        slide = Slide(str(path))
        tissue = detect_tissue(slide, cfg.tissue)
        tiles, _ = sample_tiles(
            slide, tissue, cfg.tiling, n_tiles, _slide_seed(path.stem, seed)
        )
        if not tiles:
            return []

        um2_px = slide.um2_per_pixel(cfg.tiling.level)
        names = list(axes)
        combos = [()]
        for n in names:
            combos = [c + (v,) for c in combos for v in axes[n]]
        cells = [(w, combo) for w in whites for combo in combos]

        acc = {k: Tally() for k in cells}
        fats = {}
        for w, combo in cells:
            over = dict(zip(names, combo))
            # min_area doubles as the macro/micro split point, so the micro
            # band's ceiling moves with it -- the invariant the CLI enforces.
            if "min_area_um2" in over:
                over["micro_max_area_um2"] = over["min_area_um2"]
            fats[(w, combo)] = replace(cfg.fat, white_threshold=w, **over)

        for tile in tiles:
            rgb = tile.read(slide, cfg.tiling.level)
            tmask = tissue.tile_mask(tile.x, tile.y, tile.size, tile.size)
            for w in whites:
                cset = extract_components(
                    rgb, tmask, replace(cfg.fat, white_threshold=w), um2_px
                )
                for key in cells:
                    if key[0] == w:
                        acc[key] += tally(cset, fats[key])

        return [
            {
                "slide": path.stem,
                "white": w,
                **dict(zip(names, combo)),
                "tiles": t.tiles,
                "macro_pct": t.macro_fraction * 100,
                "micro_pct": t.micro_fraction * 100,
                "macro_n": t.macro_count,
            }
            for (w, combo), t in acc.items()
        ]
    except Exception as exc:  # a bad slide contributes nothing, kills nothing
        print(f"  SKIP {path.stem}: {type(exc).__name__}: {exc}", flush=True)
        return []
    finally:
        if slide is not None:
            slide.close()


# --------------------------------------------------------------------------
# parallel drivers
# --------------------------------------------------------------------------


def _survey_worker(args) -> dict[str, Any]:
    path, cfg_dict, n_tiles, seed, group = args
    row = survey_slide(path, Config.from_dict(cfg_dict), n_tiles, seed)
    row["group"] = group
    return row


def _sweep_worker(args) -> list[dict[str, Any]]:
    path, cfg_dict, n_tiles, seed, group, whites, axes = args
    rows = sweep_slide(path, Config.from_dict(cfg_dict), n_tiles, seed, whites, axes)
    for r in rows:
        r["group"] = group
    return rows


def _run_pool(worker, jobs: list, workers: int, label: str) -> list:
    """Run `worker` over `jobs`, printing completions as they land."""
    out = []
    t0 = time.time()
    if workers <= 1:
        for i, job in enumerate(jobs, 1):
            out.append(worker(job))
            print(f"  [{i}/{len(jobs)}] {Path(job[0]).stem}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(worker, j): j for j in jobs}
            for i, fut in enumerate(as_completed(futures), 1):
                out.append(fut.result())
                name = Path(futures[fut][0]).stem
                print(f"  [{i}/{len(jobs)}] {name}", flush=True)
    print(f"  {label} done in {time.time() - t0:.0f}s ({workers} workers)")
    return out


def run_survey(
    groups: dict[str, Iterable[Path]],
    cfg: Config,
    n_tiles: int,
    seed: int = 0,
    workers: int = 1,
) -> pd.DataFrame:
    """Survey several labelled cohorts with one parameter set."""
    jobs = [
        (str(p), cfg.to_dict(), n_tiles, seed, g)
        for g, paths in groups.items()
        for p in paths
    ]
    rows = _run_pool(_survey_worker, jobs, workers, "survey")
    df = pd.DataFrame(rows).sort_values(["group", "slide"]).reset_index(drop=True)
    cols = ["group"] + [c for c in df.columns if c != "group"]
    return df[cols]


def run_sweep(
    groups: dict[str, Iterable[Path]],
    cfg: Config,
    n_tiles: int,
    whites: Sequence[int],
    axes: dict[str, Sequence[Any]],
    seed: int = 0,
    workers: int = 1,
) -> pd.DataFrame:
    """Per-slide grid results (long form: one row per slide x parameter cell)."""
    jobs = [
        (str(p), cfg.to_dict(), n_tiles, seed, g, list(whites),
         {k: list(v) for k, v in axes.items()})
        for g, paths in groups.items()
        for p in paths
    ]
    nested = _run_pool(_sweep_worker, jobs, workers, "sweep")
    return pd.DataFrame([r for rows in nested for r in rows])


def score_sweep(
    df: pd.DataFrame, positive: str = "MASH", negative: str = "CCl4"
) -> pd.DataFrame:
    """Collapse per-slide sweep rows into one scored row per parameter cell.

    Scored on cohort separation: mean positive signal retained versus mean and
    worst-case false positive on tissue known to be fat-free. `margin` is the
    gap between the weakest positive slide and the worst negative slide -- the
    quantity that decides whether a single threshold can classify slides.
    """
    meta = {"slide", "group", "tiles", "macro_pct", "micro_pct", "macro_n"}
    keys = [c for c in df.columns if c not in meta]
    pos = df[df["group"] == positive].groupby(keys)["macro_pct"]
    neg = df[df["group"] == negative].groupby(keys)["macro_pct"]

    out = pd.DataFrame({
        "pos_mean": pos.mean(),
        "pos_min": pos.min(),
        "pos_median": pos.median(),
        "neg_mean": neg.mean(),
        "neg_max": neg.max(),
        "neg_median": neg.median(),
    }).reset_index()

    out["margin"] = out["pos_min"] - out["neg_max"]
    out["ratio"] = out["pos_mean"] / out["neg_mean"].replace(0, np.nan)
    out["worst_ratio"] = out["pos_min"] / out["neg_max"].replace(0, np.nan)
    return out.sort_values("margin", ascending=False).reset_index(drop=True)
