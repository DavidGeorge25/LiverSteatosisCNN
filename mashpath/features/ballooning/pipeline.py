"""Ballooning candidate generation, end to end for one slide.

Two passes, because the judgment is relative and a tile is too small to hold
the comparison:

  1. MEASURE -- stream tiles at level 0 (never the whole level in memory),
     segment hepatocytes, and record every cell's geometry, cytoplasm intensity
     and cytoplasm texture, in level-0 coordinates.

  2. COMPARE -- z-score each cell against the cells within `radius_um` of it,
     across tile boundaries; score, veto, rank, and sample a review set that
     deliberately includes the ordinary and the unremarkable as well as the
     extreme.

Only after pass 1 has the whole slide's cells in one table can a cell be
compared against its true spatial neighbours. Doing it per tile would give
every tile its own reference population, make a cell's score depend on where
the tile grid happened to fall, and -- worst -- normalize away the signal
completely on tissue that is uniformly ballooned.

Nothing this module emits is a detection. The output is a ranked set of
proposals plus the evidence behind each one, for a pathologist to confirm or
reject; the confirmed set is the training data.
"""

from __future__ import annotations

import json
import signal

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...core.io import SlideOutputs
from ...core.slide import Slide
from ...core.tiling import count_candidate_tiles, iter_tiles
from ...core.tissue import detect_tissue
from ...core.viz import save_rgb
from ...review import manifest as manifest_mod
from ...review.candidates import Candidate, CandidateSet
from . import crops as crop_mod
from . import features as feat
from . import normalize as norm_mod
from . import rank as rank_mod
from .segment import TileCells, scale_note, segment_tile

FEATURE = "ballooning"

# Geometry measured per cell, normalized alongside the cytoplasm features.
_GEOMETRY = (
    "territory_area_um2", "nucleus_area_um2", "cyto_area_um2",
    "circularity", "solidity", "eccentricity",
    "equivalent_diameter_um", "nc_ratio", "nn_distance_um",
)



class TileTimeout(Exception):
    """One tile exceeded its budget."""


def _tile_budget_seconds(cfg: Any) -> int:
    """Per-tile wall-clock ceiling, 0 to disable.

    A cluster run that loses 6 hours because ONE tile stopped returning is a
    design fault, not bad luck. Observed on Fir twice: pass 1 ran 675 of 1187
    tiles at a steady 4s each and then stopped inside a tile, burning the job's
    whole time limit with the GPU at 0% and one core spinning. The same tiles
    complete in 2.5s on a laptop, so whatever it is, it is environmental and
    not something the caller can predict or fix.

    Skipping a tile loses ~0.08% of one slide's cells. Losing the job loses
    everything, and on a 53-slide array that is the difference between a
    result with a footnote and no result at all.
    """
    return int(getattr(cfg.ballooning.segmentation, "tile_timeout_s", 0) or 0)


def measure_slide(
    slide: Slide, cfg: Any, verbose: bool = True
) -> tuple[pd.DataFrame, list[np.ndarray], list[np.ndarray], dict[str, int]]:
    """Pass 1: every cell on the slide, with its raw features.

    Contours come back alongside the table because a review crop has to outline
    the actual cell, and re-segmenting the tile later just to recover its shape
    would cost a second pass over StarDist for no new information.
    """
    tissue = detect_tissue(slide, cfg.tissue)
    if verbose:
        print(
            f"tissue: {tissue.area_mm2:.2f} mm^2, "
            f"{tissue.coverage*100:.1f}% coverage, "
            f"{tissue.component_count} component(s)"
        )
        n_total = count_candidate_tiles(slide, tissue, cfg.tiling)
        n_do = min(cfg.limit, n_total) if cfg.limit else n_total
        print(
            f"tiles:  {n_total} pass tissue_fraction >= "
            f"{cfg.tiling.min_tissue_fraction}"
            + (f"; --limit {cfg.limit} -> processing {n_do}" if cfg.limit else "")
        )

    bcfg = cfg.ballooning
    feature_names = feat.feature_names(bcfg.features)

    frames: list[pd.DataFrame] = []
    contours: list[np.ndarray] = []
    nuc_contours: list[np.ndarray] = []
    rejected: dict[str, int] = {}
    n_tiles = 0
    t0 = time.time()

    budget = _tile_budget_seconds(cfg)
    timed_out: list[tuple[int, int]] = []
    if budget:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TileTimeout()))

    for tile in iter_tiles(slide, tissue, cfg.tiling, limit=cfg.limit):
        try:
            if budget:
                signal.alarm(budget)
            cells, _rgb = segment_tile(
                slide, tile.x, tile.y, tile.size, bcfg, tissue, measure=True
            )
        except TileTimeout:
            timed_out.append((tile.x, tile.y))
            print(f"    TIMEOUT after {budget}s on tile x={tile.x} y={tile.y} "
                  f"-- skipped ({len(timed_out)} so far)", flush=True)
            continue
        finally:
            if budget:
                signal.alarm(0)
        n_tiles += 1
        for k, v in cells.rejected.items():
            rejected[k] = rejected.get(k, 0) + v
        if len(cells) == 0:
            continue

        frames.append(_tile_frame(cells, tile.x, tile.y, feature_names))
        contours.extend(cells.contours)
        nuc_contours.extend(cells.nucleus_contours)

        if verbose and n_tiles % 10 == 0:
            done = sum(len(f) for f in frames)
            print(f"    {n_tiles} tiles, {done} cells "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    if not frames:
        return pd.DataFrame(), [], [], rejected

    df = pd.concat(frames, ignore_index=True)
    if timed_out:
        # Loud, and counted: a skipped tile is missing tissue, and if the
        # skips cluster in one region the slide's numbers are not comparable
        # to a clean one.
        rejected["tile_timeout"] = len(timed_out)
        print(f"  WARNING: {len(timed_out)} tile(s) exceeded {budget}s and were "
              f"skipped: {timed_out[:5]}{' ...' if len(timed_out) > 5 else ''}",
              flush=True)
    if verbose:
        print(f"  pass 1: {len(df)} cells from {n_tiles} tiles "
              f"[{time.time() - t0:.0f}s]")
    return df, contours, nuc_contours, rejected


def _tile_frame(
    cells: TileCells, tile_x: int, tile_y: int, feature_names: list[str]
) -> pd.DataFrame:
    data: dict[str, Any] = {
        "x": cells.x.astype(int),
        "y": cells.y.astype(int),
        "nucleus_x": cells.nucleus_x.astype(int),
        "nucleus_y": cells.nucleus_y.astype(int),
        "tile_x": tile_x,
        "tile_y": tile_y,
        "is_hepatocyte": cells.is_hepatocyte,
        "stardist_prob": cells.prob,
    }
    for name in _GEOMETRY:
        if name == "nn_distance_um":
            continue  # slide-wide; filled in after the concat
        data[name] = getattr(cells, name)
    for name in feature_names:
        data[name] = cells.measurements.get(
            name, np.full(len(cells), np.nan)
        )
    return pd.DataFrame(data)


def compare_and_rank(
    df: pd.DataFrame, cfg: Any, mpp: float, verbose: bool = True
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Pass 2: local z-scores, score, vetoes, and the review-set bands."""
    bcfg = cfg.ballooning
    ncfg, rcfg = bcfg.normalization, bcfg.ranking

    x_um = df["x"].to_numpy(dtype=float) * mpp
    y_um = df["y"].to_numpy(dtype=float) * mpp
    df["nn_distance_um"] = norm_mod.neighbour_distance_um(x_um, y_um)

    feature_cols = [c for c in (*_GEOMETRY, *feat.feature_names(bcfg.features))
                    if c in df.columns]
    values = df[feature_cols].to_numpy(dtype=float)

    is_ref = df["is_hepatocyte"].to_numpy(dtype=bool)
    if not ncfg.reference_hepatocytes_only:
        is_ref = np.ones(len(df), dtype=bool)

    z, counts, centre = norm_mod.local_zscores(x_um, y_um, values, is_ref, ncfg)
    if verbose:
        print("  " + norm_mod.summarize(counts, ncfg))

    z_names = [f"z_{c}" for c in feature_cols]
    for i, name in enumerate(z_names):
        df[name] = z[:, i]
    df["n_neighbours"] = counts
    # What "normal" was here, for the one axis a reviewer will want to check.
    area_i = feature_cols.index("territory_area_um2")
    df["local_median_territory_um2"] = centre[:, area_i]

    raw_score, used = rank_mod.score_cells(z, z_names, rcfg)
    score, vetoes = rank_mod.apply_vetoes(
        raw_score,
        df["territory_area_um2"].to_numpy(dtype=float),
        centre[:, area_i],
        df["cyto_gray_mean"].to_numpy(dtype=float),
        df["nucleus_area_um2"].to_numpy(dtype=float) > 0,
        df["is_hepatocyte"].to_numpy(dtype=bool),
        df["solidity"].to_numpy(dtype=float),
        df["eccentricity"].to_numpy(dtype=float),
        rcfg,
    )
    df["score"] = score
    # The pre-veto score, kept so a vetoed cell sent for review still carries
    # the number it would have had. Without it those rows export score=NaN.
    df["raw_score"] = raw_score
    # A cell that scored but was then vetoed -- the population the `vetoed`
    # review band is drawn from.
    vetoed_mask = np.isfinite(raw_score) & ~np.isfinite(score)
    # Rank 1 is the most ballooned-looking cell; unscorable cells get no rank.
    df["rank"] = df["score"].rank(ascending=False, method="min").astype("Int64")

    idx, bands = rank_mod.stratified_select(score, rcfg, vetoed=vetoed_mask)
    df["review_band"] = pd.Series([""] * len(df), dtype=object)
    df.loc[idx, "review_band"] = bands
    if verbose:
        print("  " + rank_mod.summarize(score, vetoes, bands))
    return df, used, vetoes


def export_review(
    slide: Slide, df: pd.DataFrame, contours: list[np.ndarray],
    nuc_contours: list[np.ndarray], out: SlideOutputs, cfg: Any, mpp: float,
    verbose: bool = True,
) -> Path:
    """Write the crops, the manifest and the operator's contact sheet."""
    bcfg = cfg.ballooning
    ccfg = bcfg.crops
    review_dir = out.review_dir(FEATURE)
    (review_dir / "overlays").mkdir(parents=True, exist_ok=True)
    if ccfg.save_unmarked:
        (review_dir / "images").mkdir(parents=True, exist_ok=True)

    # Clear crops from any previous run of this slide. Candidate ids encode
    # coordinates, so a re-run with different parameters writes a different set
    # of files and leaves the old ones behind -- a reviewer browsing the folder
    # would then see candidates that are not in the manifest and no longer
    # proposed. Only PNGs in these two generated directories are touched.
    for sub in ("overlays", "images"):
        for stale in (review_dir / sub).glob("*.png"):
            stale.unlink()

    picked = df[df["review_band"] != ""]
    if picked.empty:
        return manifest_mod.write(review_dir / "manifest.csv", [])

    # Measurements travelling with each candidate: enough for a reviewer to see
    # why it was proposed without opening the full per-cell table.
    evidence = [c for c in ("z_territory_area_um2", "z_cyto_eosin_od_mean",
                            "z_cyto_gray_mean", "z_cyto_local_std_mean",
                            "z_cyto_entropy_mean", "z_nn_distance_um",
                            "territory_area_um2", "local_median_territory_um2",
                            "nc_ratio") if c in picked.columns]

    found = CandidateSet(slide_id=slide.name, feature=FEATURE)
    rows: list[dict[str, Any]] = []
    sheet_panels: list[np.ndarray] = []
    sheet_caps: list[str] = []
    top_sorted = picked.sort_values("score", ascending=False)

    for pos, (i, row) in enumerate(picked.iterrows(), 1):
        contour = contours[i]
        cand = Candidate(
            slide_id=slide.name, feature=FEATURE,
            x=int(row["x"]), y=int(row["y"]),
            width=int(contour[:, 0].max() - contour[:, 0].min()) if len(contour) else 0,
            height=int(contour[:, 1].max() - contour[:, 1].min()) if len(contour) else 0,
            # Vetoed candidates have score NaN by construction; fall back to
            # the pre-veto value so the manifest never carries a NaN score.
            score=float(row["score"]) if pd.notna(row["score"])
            else float(row.get("raw_score", 0.0) or 0.0),
            tile_x=int(row["tile_x"]), tile_y=int(row["tile_y"]),
            measurements={k: float(row[k]) for k in evidence
                          if np.isfinite(row[k])},
        )
        found.candidates.append(cand)

        image_rel, overlay_rel, marked = crop_mod.write_candidate_images(
            slide, cand.candidate_id, contour, nuc_contours[i],
            review_dir, ccfg, mpp,
        )

        rows.append(manifest_mod.row(
            cand, image=image_rel, overlay=overlay_rel,
            # The crop is the cell's own bbox plus padding, so its physical
            # size varies per candidate -- record what was actually written
            # rather than the padding it was derived from.
            crop_um=marked.shape[1] * mpp,
            crop_mode=manifest_mod.CROP_PADDED,
            extra={
                "review_band": row["review_band"],
                "rank": int(row["rank"]) if pd.notna(row["rank"]) else "",
                "padding_um": ccfg.padding_um,
            },
        ))

        if row.name in top_sorted.index[: ccfg.contact_sheet_n]:
            # `rank` is NA for a cell that was measured but never ranked (too
            # few scoreable neighbours). The manifest above already guards it;
            # this caption did not, and crashed finalize outright on any slide
            # whose measured region was small enough for an unranked cell to
            # reach the sheet. Caption what exists rather than dropping the
            # panel -- the image is still the most useful part of the sheet.
            rank_txt = f"#{int(row['rank'])} " if pd.notna(row["rank"]) else ""
            score_txt = (f"s={row['score']:.2f}"
                         if pd.notna(row["score"]) else "s=n/a")
            sheet_panels.append(marked)
            sheet_caps.append(f"{rank_txt}{score_txt}")

        if verbose and pos % 50 == 0:
            print(f"    {pos}/{len(picked)} crops", flush=True)

    manifest = manifest_mod.write(review_dir / "manifest.csv", rows)
    _write_instructions(review_dir, len(rows), len(df), cfg)

    if sheet_panels:
        sheet = crop_mod.contact_sheet(sheet_panels, sheet_caps, ccfg)
        out.write_qc(sheet, FEATURE, "contact_sheet_top")
    return manifest


def run_slide(cfg: Any, verbose: bool = True) -> dict[str, Any]:
    """Everything, for one slide."""
    t0 = time.time()
    slide = Slide(cfg.slide_path, cfg.slide)
    try:
        if verbose:
            print(slide.describe())
            print(f"\nsegmentation:     "
                  f"{scale_note(slide, cfg.ballooning.segmentation)}\n")

        mx, my = slide.mpp_at_level(0)
        mpp = (mx + my) / 2.0

        df, contours, nuc_contours, rejected = measure_slide(slide, cfg, verbose)
        summary: dict[str, Any] = {
            "slide": slide.name, "feature": FEATURE, "cells": len(df),
        }
        if df.empty:
            if verbose:
                print("  no cells segmented")
            return summary

        bcfg = cfg.ballooning
        df, used, vetoes = compare_and_rank(df, cfg, mpp, verbose)

        out = SlideOutputs.for_slide(cfg.output_dir, slide.name).mkdirs(FEATURE)
        out.write_table(df, FEATURE, "cells")
        manifest = export_review(
            slide, df, contours, nuc_contours, out, cfg, mpp, verbose
        )
        out.write_config(cfg)

        hep_mask = df["is_hepatocyte"].to_numpy(dtype=bool)
        summary.update(
            candidates=int((df["review_band"] != "").sum()),
            hepatocytes=int(hep_mask.sum()),
            # Counted over HEPATOCYTES, not all cells: the normalization
            # reference population is hepatocytes, so a fraction taken over
            # every segmented object would be diluted by the immune and
            # endothelial cells that were never eligible to be scored.
            normalizable=int(np.isfinite(
                df["raw_score"].to_numpy(dtype=float))[hep_mask].sum()),
            scored=int(np.isfinite(df["score"].to_numpy(dtype=float))[hep_mask].sum()),
            manifest=str(manifest),
            seconds=round(time.time() - t0, 1),
        )
        if verbose:
            print(
                f"\n  per-cell table -> {out.feature_dir(FEATURE) / 'cells.csv'}"
                f"\n  review package -> {manifest.parent}"
                f"\n  contact sheet  -> "
                f"{out.feature_qc_dir(FEATURE) / 'contact_sheet_top.png'}"
                f"\n  {summary['candidates']} candidates from {len(df)} cells "
                f"in {summary['seconds']}s"
                "\n\n  These are proposals, not detections. Expect to reject a "
                "large fraction --\n  that is the system working."
            )
        return summary
    finally:
        slide.close()


def _write_instructions(out_dir: Path, n: int, n_cells: int, cfg: Any) -> None:
    r = cfg.ballooning.ranking
    (out_dir / "REVIEW.md").write_text(
        f"""# Hepatocyte ballooning -- candidate review

{n} candidates, drawn from {n_cells} segmented cells on this slide.

These are **proposals, not detections**. Nothing here was detected by a
threshold; each cell was scored on how far it deviates from the hepatocytes
physically around it, and the scoring weights are an untested hypothesis. You
are the ground truth, not the checker.

**The set is deliberately mixed.** {r.top_n} are top-ranked, {r.mid_n} are
mid-ranked and {r.low_n} are low-ranked, and the rows are shuffled so the order
tells you nothing. If it were only the top of the ranking, the confirmed set
would contain no near-misses and the model trained on it would learn a boundary
no one ever looked at. Expect many of these to be obviously negative; that is
the design, not a failure.

## How to review

1. Open `manifest.csv`.
2. For each row look at `overlays/<candidate_id>.png` -- the proposed cell
   outlined in yellow, its nucleus in blue -- and at
   `images/<candidate_id>.png`, the identical crop with nothing drawn on it.
   Judge from the unmarked one where you can: the outline is a territory
   estimate, not a measured membrane, so a slightly wrong boundary does not
   make the cell a wrong candidate.
   Each crop carries {cfg.ballooning.crops.padding_um:.0f} um of context around
   the cell so its neighbours are in frame -- ballooning is a comparison, and
   the comparison has to be visible.
3. Fill in `verdict`:
     y  - yes, this is a ballooned hepatocyte
     n  - no, it is not
     ?  - cannot tell from this crop
4. `reviewer` for your initials, `notes` for anything worth recording --
   especially *why* for a rejection. "this is a fat droplet", "this is two
   cells merged", "pale but not enlarged" each point at a different fix.

Leave a row blank if you did not look at it. Blank means unreviewed, not "no":
only rows marked `y` become positive training data, and only rows marked `n`
become negatives. An unreviewed row trained on as a negative would teach the
model exactly the wrong thing.

## What the columns mean

`score` is the weighted sum of local z-scores; only its ordering is meaningful.
`m_z_*` columns are that cell's deviation from its local neighbours in standard
deviations -- `m_z_territory_area_um2` = 2.0 means "twice-the-local-spread
larger than the hepatocytes around it".
`m_territory_area_um2` is the segmented territory, which is a nuclear-spacing
estimate rather than a measured cell boundary, and runs about 2x a true mouse
hepatocyte cross-section. Compare it against `m_local_median_territory_um2`
from the same slide region rather than against an absolute expectation.
"""
    )


# ---- chunked pass 1 --------------------------------------------------------
#
# Why this exists. A single process segmenting a whole slide stalls on Fir at
# a REPRODUCIBLE point -- three runs, each stopping at tile 675 / 97,043 cells
# / ~2,620 s, on different BLAS backends and thread settings, with the GPU at
# 0% and one core spinning. The same tiles complete in 2.5 s on a laptop, and a
# per-tile SIGALRM watchdog does not fire because the process is stuck inside a
# C call and Python cannot run a signal handler until it returns.
#
# Whatever accumulates, it accumulates IN THE PROCESS. Chunking pass 1 across
# array tasks caps how long any one process lives, which sidesteps it without
# needing to identify it -- and parallelises properly as a side effect: four
# chunks of ~300 tiles finish in the wall-clock of the slowest, not the sum.
#
# Pass 2 cannot be chunked. Z-scoring compares each cell against neighbours
# within a 150 um radius that regularly cross tile boundaries, so it needs the
# whole slide's cells at once. It is cheap (seconds) and runs as its own step.


def _save_contours(path: Path, contours: list[np.ndarray]) -> None:
    """Store variable-length contours as one flat array plus offsets.

    An object array of 100k small arrays pickles slowly and is fragile across
    numpy versions; a flat int32 array with an offset index is neither.
    """
    if contours:
        lens = np.array([len(c) for c in contours], dtype=np.int64)
        flat = (np.concatenate([np.asarray(c, dtype=np.int32).reshape(-1, 2)
                                for c in contours if len(c)])
                if lens.sum() else np.zeros((0, 2), dtype=np.int32))
    else:
        lens = np.zeros(0, dtype=np.int64)
        flat = np.zeros((0, 2), dtype=np.int32)
    np.savez_compressed(path, flat=flat, lens=lens)


def _load_contours(path: Path) -> list[np.ndarray]:
    with np.load(path) as z:
        flat, lens = z["flat"], z["lens"]
    out, pos = [], 0
    for n in lens.tolist():
        out.append(flat[pos:pos + n])
        pos += n
    return out


def chunk_dir_for(cfg: Any, slide_name: str) -> Path:
    return Path(cfg.output_dir) / slide_name / FEATURE / "chunks"


def measure_chunk(
    cfg: Any, start: int, stop: int, verbose: bool = True
) -> dict[str, Any]:
    """Pass 1 over tiles [start, stop). Writes one chunk to disk."""
    slide = Slide(cfg.slide_path, cfg.slide)
    try:
        tissue = detect_tissue(slide, cfg.tissue)
        tiles = list(iter_tiles(slide, tissue, cfg.tiling, limit=cfg.limit))
        sel = tiles[start:stop]
        if verbose:
            print(f"chunk [{start}:{stop}] -> {len(sel)} of {len(tiles)} tiles")
        if not sel:
            return {"tiles": 0, "cells": 0}

        bcfg = cfg.ballooning
        names = feat.feature_names(bcfg.features)
        frames, contours, nuc_contours = [], [], []
        rejected: dict[str, int] = {}
        t0 = time.time()

        for i, tile in enumerate(sel, 1):
            cells, _rgb = segment_tile(
                slide, tile.x, tile.y, tile.size, bcfg, tissue, measure=True
            )
            for k, v in cells.rejected.items():
                rejected[k] = rejected.get(k, 0) + v
            if len(cells):
                frames.append(_tile_frame(cells, tile.x, tile.y, names))
                contours.extend(cells.contours)
                nuc_contours.extend(cells.nucleus_contours)
            if verbose and i % 10 == 0:
                print(f"    {i}/{len(sel)} tiles, "
                      f"{sum(len(f) for f in frames)} cells "
                      f"[{time.time() - t0:.0f}s]", flush=True)

        d = chunk_dir_for(cfg, slide.name)
        d.mkdir(parents=True, exist_ok=True)
        tag = f"{start:06d}_{stop:06d}"
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        # Pickle, not CSV. A chunk is an intermediate read back by `finalize`
        # in the same run, and CSV cannot round-trip it exactly: column dtypes
        # promote (float32 -> float64) on re-read, which shifts every derived
        # z-score by ~1e-13. Measured: two single-process runs agree on all 76
        # columns bit-for-bit, while a CSV-chunked run differed on 63 of them.
        # The drift is far too small to move a candidate -- the review set was
        # identical either way -- but exact reproducibility is worth more than
        # a human-readable intermediate, and cells.csv is still written as CSV.
        df.to_pickle(d / f"cells_{tag}.pkl")
        _save_contours(d / f"contours_{tag}.npz", contours)
        _save_contours(d / f"nuccontours_{tag}.npz", nuc_contours)
        (d / f"rejected_{tag}.json").write_text(json.dumps(rejected))

        if verbose:
            print(f"  chunk done: {len(df)} cells from {len(sel)} tiles "
                  f"[{time.time() - t0:.0f}s] -> {d}")
        return {"tiles": len(sel), "cells": len(df), "chunk": tag}
    finally:
        slide.close()


def merge_chunks(
    cfg: Any, slide_name: str, verbose: bool = True, allow_disjoint: bool = False
) -> tuple[pd.DataFrame, list[np.ndarray], list[np.ndarray], dict[str, int]]:
    """Reassemble every chunk, in tile order. Fails loudly on a gap.

    A missing chunk is not a warning. Pass 2 z-scores each cell against its
    spatial neighbours, so a hole in the middle of the slide silently changes
    every score near it -- the run would complete and report numbers that are
    wrong in a way nothing downstream could detect.

    `allow_disjoint` is for the one case where the gaps are the POINT rather
    than a failure: sampling a slide as several spaced windows instead of one
    contiguous strip, so the measured tiles cover the whole section rather than
    one band of it. The z-score argument above still holds and is what makes
    this safe -- normalization.radius_um is 150 um against a 253.6 um tile, so
    cells in windows hundreds of tiles apart were never in each other's
    neighbourhoods and merging them is arithmetically identical to scoring each
    window alone. What it does NOT excuse is the extra boundary: four windows
    have four times the edge of one, and a cell on an edge is normalized
    against a half-neighbourhood. That is bounded by `min_neighbours` (such
    cells are kept but left unscored) and is reported below as the scored
    fraction, so the cost is visible rather than assumed away. Never pass this
    to cover for a chunk whose job died -- that hole is adjacent to measured
    tissue, which is exactly the case the check exists for.
    """
    d = chunk_dir_for(cfg, slide_name)
    if not d.exists():
        raise FileNotFoundError(f"no chunk directory at {d}")
    cell_files = sorted(d.glob("cells_*.pkl"))
    if not cell_files:
        raise FileNotFoundError(f"no chunks in {d}")

    spans = []
    for f in cell_files:
        a, b = f.stem.split("_")[1:3]
        spans.append((int(a), int(b), f))
    spans.sort()
    gaps = [(b0, a1) for (a0, b0, _), (a1, _, _) in zip(spans, spans[1:])
            if a1 != b0]
    if gaps and not allow_disjoint:
        b0, a1 = gaps[0]
        raise ValueError(
            f"chunk gap: one chunk ends at tile {b0}, the next starts at "
            f"{a1}. Every tile range must be covered exactly once -- a gap "
            f"would corrupt the local z-scores silently. Chunks present: "
            f"{[(a, b) for a, b, _ in spans]}. If the gaps are deliberate "
            f"(a slide sampled as spaced windows), pass allow_disjoint=True "
            f"or --allow-disjoint, which says so on the record."
        )
    if gaps and verbose:
        print(f"  {len(gaps) + 1} disjoint window(s), gaps at "
              + ", ".join(f"{b}->{a}" for b, a in gaps))

    frames, contours, nuc_contours = [], [], []
    rejected: dict[str, int] = {}
    for a, b, f in spans:
        tag = f"{a:06d}_{b:06d}"
        part = pd.read_pickle(f)
        if len(part):
            frames.append(part)
        contours.extend(_load_contours(d / f"contours_{tag}.npz"))
        nuc_contours.extend(_load_contours(d / f"nuccontours_{tag}.npz"))
        rj = d / f"rejected_{tag}.json"
        if rj.exists():
            for k, v in json.loads(rj.read_text()).items():
                rejected[k] = rejected.get(k, 0) + v

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if verbose:
        print(f"  merged {len(spans)} chunk(s): {len(df)} cells, "
              f"tiles [{spans[0][0]}:{spans[-1][1]}]")
    if len(df) != len(contours):
        raise ValueError(
            f"{len(df)} cells but {len(contours)} contours -- chunks are "
            "inconsistent; re-run the affected chunk."
        )
    return df, contours, nuc_contours, rejected


def finalize_slide(cfg: Any, verbose: bool = True,
                   allow_disjoint: bool = False) -> dict[str, Any]:
    """Pass 2 on merged chunks: z-score, rank, export. Cheap, one process."""
    t0 = time.time()
    slide = Slide(cfg.slide_path, cfg.slide)
    try:
        mx, my = slide.mpp_at_level(0)
        mpp = (mx + my) / 2.0
        df, contours, nuc_contours, rejected = merge_chunks(
            cfg, slide.name, verbose, allow_disjoint=allow_disjoint)
        summary: dict[str, Any] = {
            "slide": slide.name, "feature": FEATURE, "cells": len(df),
        }
        if df.empty:
            print("  no cells in any chunk")
            return summary

        bcfg = cfg.ballooning
        df, used, vetoes = compare_and_rank(df, cfg, mpp, verbose)
        out = SlideOutputs.for_slide(cfg.output_dir, slide.name).mkdirs(FEATURE)
        out.write_table(df, FEATURE, "cells")
        manifest = export_review(
            slide, df, contours, nuc_contours, out, cfg, mpp, verbose
        )
        out.write_config(cfg)
        hep_mask = df["is_hepatocyte"].to_numpy(dtype=bool)
        summary.update(
            candidates=int((df["review_band"] != "").sum()),
            hepatocytes=int(hep_mask.sum()),
            # Counted over HEPATOCYTES, not all cells: the normalization
            # reference population is hepatocytes, so a fraction taken over
            # every segmented object would be diluted by the immune and
            # endothelial cells that were never eligible to be scored.
            normalizable=int(np.isfinite(
                df["raw_score"].to_numpy(dtype=float))[hep_mask].sum()),
            scored=int(np.isfinite(df["score"].to_numpy(dtype=float))[hep_mask].sum()),
            manifest=str(manifest),
            seconds=round(time.time() - t0, 1),
        )
        if verbose:
            # Two different losses, reported apart because they mean opposite
            # things. `normalizable` is how many hepatocytes had enough
            # neighbours to z-score against at all -- that is the cost of
            # window boundaries, and the number to watch when a slide is
            # sampled as several small windows rather than one block. What
            # survives the VETOES is a property of the detector's thresholds,
            # not of the sampling, and is normally the far larger loss. Adding
            # them together (an earlier version printed only the total, called
            # it neighbours, and made the boundary cost look ~20x its real
            # size) hides both.
            hep = max(summary["hepatocytes"], 1)
            print(f"\n  {summary['candidates']} candidates from {len(df)} cells "
                  f"in {summary['seconds']}s"
                  f"\n  normalizable {summary['normalizable']}/"
                  f"{summary['hepatocytes']} hepatocytes "
                  f"({summary['normalizable'] / hep:.1%}) -- the rest had fewer "
                  f"than {bcfg.normalization.min_neighbours} neighbours"
                  f"\n  passed the vetoes {summary['scored']} "
                  f"({summary['scored'] / hep:.1%} of hepatocytes)"
                  f"\n  review package -> {manifest.parent}")
        return summary
    finally:
        slide.close()
