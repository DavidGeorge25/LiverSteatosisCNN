"""Stage 1-2: segment nuclei over a slide and split them into classes.

This is the viability check that has to pass before focus detection is worth
building. It produces, for one slide:

    inflammation/nuclei.csv              every nucleus measured
    inflammation/nuclei_per_tile.csv     per-tile counts and densities
    inflammation/nuclei_summary.json     slide-level numbers
    qc/inflammation/class_overlay_*.png  tiles with nuclei coloured by class
    qc/inflammation/nucleus_distribution.png   area vs intensity, cutoffs drawn
    qc/inflammation/immune_map.png       immune centroids over the thumbnail

and prints the observed distributions next to the configured cutoffs, because
a cutoff that sits outside the data does nothing while still looking like a
decision. `check_cutoffs` reports exactly that case.

Tiles are streamed one at a time; level 0 is never held whole.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...core.io import SlideOutputs
from ...core.slide import Slide
from ...core.tiling import Tile, iter_tiles
from ...core.tissue import TissueMask, detect_tissue
from . import viz
from .classify import (
    AMBIGUOUS, CLASS_NAMES, DEBRIS, HEPATOCYTE, IMMUNE,
    bimodality_coefficient, classify, intensity_for, is_dark,
    suggest_cutoffs, tally,
)
from .nuclei import TileNuclei, scale_note, segment_tile

FEATURE = "inflammation"


def select_tiles(
    slide: Slide, tissue: TissueMask, cfg: Any, sample: int, seed: int
) -> tuple[list[Tile], str]:
    """Pick the tiles to process.

    `--limit` walks in raster order, which is what makes a run reproducible and
    bounded. For the viability question that is the wrong sample -- the first N
    tiles are all one corner of one lobe -- so `--sample` draws uniformly from
    the whole tissue instead. Enumerating candidates costs no pixel reads.
    """
    if sample > 0:
        all_tiles = list(iter_tiles(slide, tissue, cfg.tiling, limit=0))
        if sample >= len(all_tiles):
            return all_tiles, f"all {len(all_tiles)} candidate tiles"
        rng = random.Random(seed)
        chosen = sorted(rng.sample(range(len(all_tiles)), sample))
        return [all_tiles[i] for i in chosen], (
            f"{sample} of {len(all_tiles)} candidate tiles, random (seed {seed})"
        )

    tiles = list(iter_tiles(slide, tissue, cfg.tiling, limit=cfg.limit))
    how = "raster order" + (f", --limit {cfg.limit}" if cfg.limit else "")
    return tiles, f"{len(tiles)} tiles, {how}"


def run_nuclei(
    cfg: Any,
    sample: int = 0,
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Segment and classify nuclei across a slide, and write the QC package."""
    icfg = cfg.inflammation
    ncfg, ccfg, qcfg = icfg.nuclei, icfg.classify, icfg.nucleus_qc

    slide = Slide(cfg.slide_path, cfg.slide)
    try:
        if verbose:
            print(slide.describe())
            print()

        t0 = time.time()
        tissue = detect_tissue(slide, cfg.tissue)
        if verbose:
            print(f"tissue: {tissue.area_mm2:.2f} mm^2, "
                  f"{tissue.coverage * 100:.1f}% coverage [{time.time() - t0:.1f}s]")

        tiles, how = select_tiles(slide, tissue, cfg, sample, seed)
        if not tiles:
            raise RuntimeError("no tiles passed the tissue fraction threshold")
        if verbose:
            print(f"tiles:  {how}")
            print(f"scale:  {scale_note(slide, ncfg)}")
            print(f"model:  {ncfg.model_name}"
                  + (f", prob>{ncfg.prob_threshold}" if ncfg.prob_threshold else "")
                  + f", margin {ncfg.margin_px}px")
            print()

        out = SlideOutputs.for_slide(cfg.output_dir, slide.name).mkdirs(FEATURE)

        rng = random.Random(seed)
        panel_idx = set(rng.sample(range(len(tiles)), min(qcfg.sample_tiles, len(tiles))))

        chunks: list[np.ndarray] = []
        tile_rows: list[dict[str, Any]] = []
        panels: list[np.ndarray] = []
        rejected_total: dict[str, int] = {}
        um2_px = slide.um2_per_pixel(ncfg.level)
        tile_area_um2 = tiles[0].size * tiles[0].size * um2_px

        t_seg = time.time()
        for i, tile in enumerate(tiles):
            nuc, rgb = segment_tile(slide, tile.x, tile.y, tile.size, ncfg, tissue)
            cls = classify(nuc.area_um2, nuc.gray, nuc.hematoxylin, ccfg)
            counts = tally(cls)

            for k, v in nuc.rejected.items():
                rejected_total[k] = rejected_total.get(k, 0) + v

            if len(nuc):
                chunks.append(_stack(nuc, cls, tile))

            # Density uses the tile's actual tissue area, not its full extent --
            # an edge tile that is half glass would otherwise read as half as
            # inflamed as it is.
            tissue_mm2 = tile_area_um2 * tile.tissue_fraction / 1e6
            tile_rows.append({
                "tile_x": tile.x, "tile_y": tile.y,
                "tissue_fraction": round(tile.tissue_fraction, 4),
                "tissue_mm2": round(tissue_mm2, 6),
                "n_nuclei": len(nuc),
                "n_immune": counts.get(IMMUNE),
                "n_hepatocyte": counts.get(HEPATOCYTE),
                "n_ambiguous": counts.get(AMBIGUOUS),
                "n_debris": counts.get(DEBRIS),
                "immune_fraction": round(counts.fraction(IMMUNE), 4),
                "immune_per_mm2": round(counts.get(IMMUNE) / tissue_mm2, 1)
                if tissue_mm2 > 0 else 0.0,
                "nuclei_per_mm2": round(len(nuc) / tissue_mm2, 1)
                if tissue_mm2 > 0 else 0.0,
            })

            if i in panel_idx:
                caption = (f"x{tile.x} y{tile.y}  immune {counts.get(IMMUNE)}"
                           f"/{counts.scored}")
                fig = viz.tile_class_figure(rgb, nuc.labels, cls, qcfg, caption, counts)
                panels.append(fig)
                if qcfg.save_full_size_panels:
                    out.write_qc(fig, FEATURE, f"class_overlay_x{tile.x:06d}_y{tile.y:06d}")

            if verbose and (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(tiles)} tiles  [{time.time() - t_seg:.0f}s]")

        seg_seconds = time.time() - t_seg
        data = (np.concatenate(chunks) if chunks
                else np.zeros((0, len(_NUC_COLS)), dtype=np.float64))
        df = pd.DataFrame(data, columns=list(_NUC_COLS))
        df["class_name"] = [CLASS_NAMES.get(int(c), "?") for c in df["class"]]

        tile_df = pd.DataFrame(tile_rows)
        summary = _summarize(
            slide, tissue, df, tile_df, icfg, len(tiles), rejected_total, seg_seconds
        )

        _write_outputs(out, slide, tissue, df, tile_df, summary, icfg, panels, how)

        if verbose:
            _print_report(summary, df, icfg)
            print(f"\n  outputs -> {out.feature_dir(FEATURE)}")
            print(f"  QC      -> {out.feature_qc_dir(FEATURE)}")
        return summary
    finally:
        slide.close()


_NUC_COLS = (
    "x", "y", "area_um2", "circularity", "eccentricity", "solidity",
    "gray", "hematoxylin", "prob", "class", "tile_x", "tile_y",
)


def _stack(nuc: TileNuclei, cls: np.ndarray, tile: Tile) -> np.ndarray:
    return np.column_stack([
        nuc.x, nuc.y, nuc.area_um2, nuc.circularity, nuc.eccentricity,
        nuc.solidity, nuc.gray, nuc.hematoxylin, nuc.prob,
        cls.astype(np.float64),
        np.full(len(nuc), tile.x, dtype=np.float64),
        np.full(len(nuc), tile.y, dtype=np.float64),
    ])


def check_cutoffs(
    df: pd.DataFrame, icfg: Any, tile_df: pd.DataFrame | None = None
) -> list[str]:
    """Flag cutoffs that cannot bite, given the observed data.

    A threshold outside the range of the measurements is not a conservative
    choice -- it is a no-op wearing the costume of a decision. Worth saying out
    loud, because the classification still returns a confident-looking split
    driven entirely by whichever cutoff *is* inside the data.
    """
    warnings: list[str] = []
    if df.empty:
        return ["no nuclei detected -- nothing to check"]

    ccfg = icfg.classify
    channel = ccfg.intensity_channel
    inten = df[channel].to_numpy()
    lo, hi = float(np.nanmin(inten)), float(np.nanmax(inten))

    if ccfg.require_dark:
        cut = (ccfg.immune_max_intensity if channel == "gray"
               else ccfg.immune_min_hematoxylin)
        frac = float(is_dark(ccfg, inten).mean())
        if not (lo <= cut <= hi):
            warnings.append(
                f"classify.{'immune_max_intensity' if channel == 'gray' else 'immune_min_hematoxylin'}"
                f" = {cut:g} is outside the observed {channel} range "
                f"[{lo:.3g}, {hi:.3g}] -- the intensity test passes "
                f"{frac * 100:.0f}% of nuclei, so the split is on area alone"
            )
        elif frac > 0.95 or frac < 0.05:
            warnings.append(
                f"the {channel} cutoff {cut:g} admits {frac * 100:.0f}% of "
                f"nuclei -- close to inert; the split is effectively on area alone"
            )

    area = df["area_um2"].to_numpy()
    if ccfg.immune_max_area_um2 > np.percentile(area, 99):
        warnings.append(
            f"classify.immune_max_area_um2 = {ccfg.immune_max_area_um2:g} sits above "
            f"the 99th percentile of area ({np.percentile(area, 99):.1f}) -- "
            "nearly everything qualifies as immune"
        )
    if ccfg.hepatocyte_min_area_um2 < np.percentile(area, 1):
        warnings.append(
            f"classify.hepatocyte_min_area_um2 = {ccfg.hepatocyte_min_area_um2:g} sits "
            f"below the 1st percentile of area -- nearly everything qualifies "
            "as hepatocyte"
        )

    bc_area = bimodality_coefficient(area)
    if np.isfinite(bc_area) and bc_area < 0.555:
        warnings.append(
            f"nuclear area looks unimodal (bimodality coefficient {bc_area:.3f} "
            "< 0.555) -- a size cutoff is cutting one population in half rather "
            "than separating two"
        )

    # Shape: round lymphocytes vs flattened endothelial nuclei. Both are small
    # and dark, so the size/intensity rule cannot tell them apart, but they
    # differ sharply in eccentricity.
    imm = df[df["class"] == IMMUNE]
    if len(imm) > 50:
        oval = float((imm["eccentricity"] > 0.70).mean())
        if oval > 0.30:
            warnings.append(
                f"{oval * 100:.0f}% of immune-classified nuclei have "
                "eccentricity > 0.70 -- flattened sinusoidal lining cells "
                "(endothelial/Kupffer), not lymphocytes. Shape is the "
                "discriminator the size/intensity rule is missing"
            )

    # Pattern: lobular inflammation is focal. A class present at similar
    # density in every tile is a resident population, whatever it is called.
    if tile_df is not None and len(tile_df) > 5 and "immune_per_mm2" in tile_df:
        v = tile_df["immune_per_mm2"].to_numpy()
        zero = int((tile_df["n_immune"] == 0).sum())
        cv = float(v.std() / v.mean()) if v.mean() else float("inf")
        if zero == 0 and cv < 0.60:
            warnings.append(
                f"the immune class appears in {len(tile_df)}/{len(tile_df)} tiles "
                f"at CV {cv:.2f} (max/median {v.max() / max(np.median(v), 1e-9):.1f}x) "
                "-- that is a uniform resident population, not focal "
                "inflammation. Focus detection must key on local density "
                "EXCESS over this baseline, not on absolute counts"
            )
    return warnings


def _summarize(
    slide: Slide, tissue: TissueMask, df: pd.DataFrame, tile_df: pd.DataFrame,
    icfg: Any, n_tiles: int, rejected: dict[str, int], seconds: float,
) -> dict[str, Any]:
    ccfg = icfg.classify
    counts = tally(df["class"].to_numpy().astype(np.int8)) if not df.empty else tally(
        np.zeros(0, np.int8))
    sampled_mm2 = float(tile_df["tissue_mm2"].sum()) if not tile_df.empty else 0.0

    s: dict[str, Any] = {
        "slide": slide.name,
        "feature": FEATURE,
        "stage": "nuclei+classify",
        "tissue_mm2": round(tissue.area_mm2, 3),
        "tiles_processed": n_tiles,
        "sampled_tissue_mm2": round(sampled_mm2, 4),
        "sampled_fraction_of_tissue": round(sampled_mm2 / tissue.area_mm2, 4)
        if tissue.area_mm2 else 0.0,
        "n_nuclei": int(len(df)),
        "nuclei_per_mm2": round(len(df) / sampled_mm2, 1) if sampled_mm2 else 0.0,
        "seconds": round(seconds, 1),
        "rejected": rejected,
        "intensity_channel": ccfg.intensity_channel,
    }
    for code in (IMMUNE, HEPATOCYTE, AMBIGUOUS, DEBRIS):
        s[f"n_{CLASS_NAMES[code]}"] = counts.get(code)
        s[f"frac_{CLASS_NAMES[code]}"] = round(counts.fraction(code), 4)
    s["immune_per_mm2"] = (round(counts.get(IMMUNE) / sampled_mm2, 1)
                           if sampled_mm2 else 0.0)

    if not df.empty:
        area = df["area_um2"].to_numpy()
        inten = df[ccfg.intensity_channel].to_numpy()
        s["suggested"] = {k: round(v, 4)
                          for k, v in suggest_cutoffs(area, inten,
                                                      ccfg.intensity_channel).items()}
        s["bimodality"] = {
            "area_um2": round(bimodality_coefficient(area), 4),
            ccfg.intensity_channel: round(bimodality_coefficient(inten), 4),
        }
        # Per-tile spread matters: real inflammation is patchy, so a flat
        # immune fraction across tiles is evidence the class is picking up
        # resident sinusoidal cells rather than foci.
        if not tile_df.empty:
            f = tile_df["immune_per_mm2"].to_numpy()
            s["immune_per_mm2_tile_spread"] = {
                "p10": round(float(np.percentile(f, 10)), 1),
                "median": round(float(np.median(f)), 1),
                "p90": round(float(np.percentile(f, 90)), 1),
                "max": round(float(f.max()), 1),
            }
    if not df.empty:
        imm = df[df["class"] == IMMUNE]
        s["immune_shape"] = {
            "ecc_p25": round(float(imm["eccentricity"].quantile(0.25)), 3),
            "ecc_p50": round(float(imm["eccentricity"].quantile(0.50)), 3),
            "ecc_p75": round(float(imm["eccentricity"].quantile(0.75)), 3),
            "frac_round_ecc_lt_0.70": round(float((imm["eccentricity"] <= 0.70).mean()), 4),
        } if len(imm) else {}
        if not tile_df.empty:
            v = tile_df["immune_per_mm2"].to_numpy()
            s["immune_uniformity"] = {
                "cv": round(float(v.std() / v.mean()), 3) if v.mean() else None,
                "tiles_with_zero_immune": int((tile_df["n_immune"] == 0).sum()),
                "tiles": int(len(tile_df)),
            }
    s["warnings"] = check_cutoffs(df, icfg, tile_df)
    return s


def _write_outputs(
    out: SlideOutputs, slide: Slide, tissue: TissueMask, df: pd.DataFrame,
    tile_df: pd.DataFrame, summary: dict[str, Any], icfg: Any,
    panels: list[np.ndarray], how: str,
) -> None:
    from ...core.viz import grid, save_rgb

    qcfg, ccfg = icfg.nucleus_qc, icfg.classify

    if qcfg.save_nucleus_csv and not df.empty:
        out.write_table(df, FEATURE, "nuclei")
    out.write_table(tile_df, FEATURE, "nuclei_per_tile")

    path = out.feature_dir(FEATURE) / "nuclei_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str))

    if panels:
        save_rgb(out.feature_qc_dir(FEATURE) / "class_overlay_sheet.png",
                 grid(panels, cols=max(1, qcfg.cols), gap=10))

    if not df.empty:
        viz.distribution_figure(
            df["area_um2"].to_numpy(),
            df[ccfg.intensity_channel].to_numpy(),
            df["class"].to_numpy().astype(np.int8),
            ccfg, qcfg,
            out.feature_qc_dir(FEATURE) / "nucleus_distribution.png",
            title=f"{slide.name} -- nuclear area vs {ccfg.intensity_channel}",
            subtitle=(f"{len(df):,} nuclei from {how}; dashed = area cutoffs, "
                      f"dotted = intensity cutoff"),
        )
        viz.distribution_figure(
            df["area_um2"].to_numpy(),
            df["eccentricity"].to_numpy(),
            df["class"].to_numpy().astype(np.int8),
            ccfg, qcfg,
            out.feature_qc_dir(FEATURE) / "nucleus_shape.png",
            title=f"{slide.name} -- nuclear area vs eccentricity",
            subtitle=("round lymphocytes sit low; flattened endothelial and "
                      "Kupffer nuclei sit high. Neither is separated by the "
                      "size/intensity rule."),
            y_label="eccentricity  (0 = circle, 1 = line)",
        )
        imm = df[df["class"] == IMMUNE]
        viz.thumbnail_density_figure(
            slide.thumbnail(2000), imm["x"].to_numpy(), imm["y"].to_numpy(),
            downsample=slide.dimensions[0] / slide.thumbnail(2000).shape[1],
            path=out.feature_qc_dir(FEATURE) / "immune_map.png",
            title=f"{slide.name} -- immune-classified nuclei ({len(imm):,})",
            color=tuple(qcfg.immune_color),
        )


def _print_report(summary: dict[str, Any], df: pd.DataFrame, icfg: Any) -> None:
    ch = icfg.classify.intensity_channel
    print()
    print("=" * 74)
    print("NUCLEUS SEGMENTATION AND CLASS SPLIT")
    print("=" * 74)
    print(f"  nuclei:          {summary['n_nuclei']:,} over "
          f"{summary['sampled_tissue_mm2']:.3f} mm^2 sampled tissue "
          f"({summary['sampled_fraction_of_tissue'] * 100:.1f}% of slide)")
    print(f"  density:         {summary['nuclei_per_mm2']:,.0f} nuclei/mm^2")
    for code in (IMMUNE, HEPATOCYTE, AMBIGUOUS, DEBRIS):
        name = CLASS_NAMES[code]
        print(f"  {name + ':':<16} {summary[f'n_{name}']:>8,}  "
              f"{summary[f'frac_{name}'] * 100:>5.1f}%")
    print(f"  immune density:  {summary['immune_per_mm2']:,.0f} /mm^2")
    if "immune_per_mm2_tile_spread" in summary:
        sp = summary["immune_per_mm2_tile_spread"]
        print(f"  per-tile spread: p10 {sp['p10']:,.0f} | median {sp['median']:,.0f} "
              f"| p90 {sp['p90']:,.0f} | max {sp['max']:,.0f} /mm^2")
    print(f"  rejected:        {summary['rejected']}")
    print(f"  elapsed:         {summary['seconds']}s")

    if not df.empty:
        print()
        print("  observed distributions")
        for col, fmt in (("area_um2", "8.1f"), (ch, "8.3f"), ("circularity", "8.2f")):
            v = df[col].to_numpy()
            ps = np.percentile(v, [5, 25, 50, 75, 95])
            print(f"    {col:<12} p5/25/50/75/95 = "
                  + " ".join(f"{p:{fmt}}" for p in ps))
        bm = summary.get("bimodality", {})
        print(f"    bimodality     area {bm.get('area_um2', float('nan')):.3f}  "
              f"{ch} {bm.get(ch, float('nan')):.3f}   (>0.555 suggests two modes)")
        sug = summary.get("suggested", {})
        if sug:
            print(f"    otsu           area {sug.get('otsu_area_um2', float('nan')):.1f} um^2"
                  f"   {ch} {sug.get(f'otsu_{ch}', float('nan')):.3f}")
        sh = summary.get("immune_shape") or {}
        if sh:
            print(f"    immune shape   eccentricity p25/50/75 = "
                  f"{sh['ecc_p25']:.2f} {sh['ecc_p50']:.2f} {sh['ecc_p75']:.2f}"
                  f"   round (<=0.70) {sh['frac_round_ecc_lt_0.70'] * 100:.0f}%")
        un = summary.get("immune_uniformity") or {}
        if un and un.get("cv") is not None:
            print(f"    immune pattern CV {un['cv']:.2f} across tiles; "
                  f"{un['tiles_with_zero_immune']}/{un['tiles']} tiles have none")

    if summary["warnings"]:
        print()
        print("  WARNINGS")
        for w in summary["warnings"]:
            print(f"    - {w}")
