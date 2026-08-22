"""Ballooning pipeline entry point.

    python -m mashpath.features.ballooning.cli segment \\
        --slide data/R25-264_MASH_HE/R25-264-1.svs \\
        --config configs/core.yaml --config configs/ballooning.yaml \\
        --tiles 8

`segment` is the QC gate: it runs the segmentation on a handful of tiles and
writes native-resolution overlays, so the cell boundaries can be confirmed by
eye before any feature is computed on top of them. Nothing downstream is worth
looking at until those overlays are right.

This lives beside the feature rather than in `mashpath.cli` because it needs
StarDist, and importing TensorFlow into the top-level CLI would make
`--help` and every steatosis run pay a multi-second import they never use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ...config import MashConfig
from ...core.io import SlideOutputs
from ...core.slide import Slide
from ...core.tiling import Tile, count_candidate_tiles, iter_tiles
from ...core.tissue import detect_tissue
from ...core.viz import save_rgb
from . import pipeline as pipe
from . import qc as qc_mod
from .segment import scale_note, segment_tile


def _sample_tiles(tiles: list[Tile], n: int) -> list[Tile]:
    """Evenly spaced across the slide's accepted tiles, not the first N.

    Tiles come out of `iter_tiles` in raster order, so the first N are all in
    one strip at the top of the section -- one lobule, one staining gradient,
    one focus plane. Spreading the sample is the difference between checking
    the segmentation and checking one corner of it.
    """
    if n <= 0 or len(tiles) <= n:
        return tiles
    idx = np.linspace(0, len(tiles) - 1, n).round().astype(int)
    return [tiles[i] for i in dict.fromkeys(idx.tolist())]


def cmd_segment(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    bcfg = cfg.ballooning
    scfg = bcfg.segmentation

    slide = Slide(cfg.slide_path, cfg.slide)
    try:
        print(slide.describe())
        print(f"\nsegmentation:     {scale_note(slide, scfg)}")
        print(f"model:            {scfg.model_name}")
        print(
            f"expansion:        <= {scfg.max_expansion_um} um, "
            f"edge_weight={scfg.edge_weight}, void>{scfg.void_intensity}"
        )
        print(f"margin:           {scfg.margin_px} px "
              f"({scfg.margin_px * slide.mpp_at_level(scfg.level)[0]:.1f} um)\n")

        tissue = detect_tissue(slide, cfg.tissue)
        print(
            f"tissue: {tissue.area_mm2:.2f} mm^2, "
            f"{tissue.coverage*100:.1f}% coverage, "
            f"{tissue.component_count} component(s)"
        )

        pool = list(iter_tiles(slide, tissue, cfg.tiling, limit=cfg.limit))
        chosen = _sample_tiles(pool, args.tiles)
        print(
            f"tiles:  {len(pool)} pass tissue_fraction >= "
            f"{cfg.tiling.min_tissue_fraction}"
            + (f"; --limit {cfg.limit}" if cfg.limit else "")
            + f"; segmenting {len(chosen)} spread across the section\n"
        )

        out = SlideOutputs.for_slide(cfg.output_dir, slide.name).mkdirs("ballooning")
        qc_dir = out.feature_qc_dir("ballooning") / "segment"
        mx, my = slide.mpp_at_level(scfg.level)
        mpp = (mx + my) / 2.0

        rows = []
        rejected_total: dict[str, int] = {}
        for i, tile in enumerate(chosen, 1):
            cells, rgb = segment_tile(
                slide, tile.x, tile.y, tile.size, bcfg, tissue, measure=False
            )
            stats = qc_mod.summarize(cells, mpp)
            for k, v in cells.rejected.items():
                rejected_total[k] = rejected_total.get(k, 0) + v

            fig = qc_mod.tile_qc_figure(
                rgb, cells, bcfg.qc,
                caption=f"tissue={tile.tissue_fraction:.2f}",
            )
            path = save_rgb(qc_dir / f"{tile.name}.png", fig)

            print(
                f"  [{i}/{len(chosen)}] {tile.name}  "
                f"cells={int(stats['cells']):4d}  "
                f"hep={int(stats['hepatocytes']):4d}  "
                + (
                    f"median_terr={stats['median_territory_area_um2']:7.1f} um^2  "
                    f"median_diam={stats['median_diameter_um']:5.1f} um  "
                    f"N:C={stats['median_nc_ratio']:.3f}"
                    if "median_territory_area_um2" in stats else "(no hepatocytes)"
                )
            )
            rows.append({"tile": tile.name, "x": tile.x, "y": tile.y,
                         "tissue_fraction": round(tile.tissue_fraction, 4),
                         **{k: round(v, 4) for k, v in stats.items()},
                         "overlay": str(path)})

        if rows:
            out.write_table(rows, "ballooning", "segment_qc")
            _print_pooled(rows, rejected_total)
            print(f"\n  overlays -> {qc_dir}")
            print("  Open them at 100%: the boundary errors that matter are "
                  "invisible when scaled down.")
        else:
            print("  no tiles segmented")
        return 0
    finally:
        slide.close()


def cmd_run(args: argparse.Namespace) -> int:
    """The full two-pass pipeline for one slide."""
    cfg = _load_config(args)
    pipe.run_slide(cfg)
    return 0



def cmd_count(args: argparse.Namespace) -> int:
    """Print the tile count, so a submit script can size the array."""
    cfg = _load_config(args)
    slide = Slide(cfg.slide_path, cfg.slide)
    try:
        tissue = detect_tissue(slide, cfg.tissue)
        n = count_candidate_tiles(slide, tissue, cfg.tiling)
        if cfg.limit:
            n = min(n, cfg.limit)
        print(n)
    finally:
        slide.close()
    return 0


def cmd_chunk(args: argparse.Namespace) -> int:
    """Pass 1 over one tile range. Safe to run many of these in parallel."""
    cfg = _load_config(args)
    pipe.measure_chunk(cfg, args.start, args.stop)
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
    """Pass 2 on the merged chunks: z-score, rank, export the review package."""
    cfg = _load_config(args)
    pipe.finalize_slide(cfg, allow_disjoint=getattr(args, "allow_disjoint", False))
    return 0


def _print_pooled(rows: list[dict], rejected: dict[str, int]) -> None:
    """Pooled numbers, which is where an obviously wrong scale shows up.

    A normal mouse hepatocyte is ~20-25 um across (~350-500 um^2 in section).
    If the pooled median lands far off that, the segmentation is wrong in a way
    no amount of downstream normalization will fix.
    """
    areas = [r["median_territory_area_um2"] for r in rows
             if "median_territory_area_um2" in r]
    if not areas:
        return
    total_cells = sum(int(r["cells"]) for r in rows)
    total_hep = sum(int(r["hepatocytes"]) for r in rows)
    print(
        f"\npooled: {total_cells} cells, {total_hep} hepatocytes "
        f"({total_hep / max(total_cells, 1) * 100:.0f}%), "
        f"median territory {np.median(areas):.0f} um^2"
    )
    if rejected:
        dropped = ", ".join(f"{k}={v}" for k, v in sorted(rejected.items()) if v)
        if dropped:
            print(f"dropped: {dropped}")


def _load_config(args: argparse.Namespace) -> MashConfig:
    cfg = MashConfig.from_yaml(args.config) if args.config else MashConfig()
    cfg.apply_overrides({
        "slide_path": args.slide,
        "output_dir": args.output_dir,
        "limit": args.limit,
        "tiling.tile_size": args.tile_size,
        "tiling.min_tissue_fraction": args.min_tissue_fraction,
        "ballooning.segmentation.prob_threshold": args.prob_threshold,
        "ballooning.segmentation.nms_threshold": args.nms_threshold,
        "ballooning.segmentation.max_expansion_um": args.max_expansion_um,
        "ballooning.segmentation.edge_weight": args.edge_weight,
        "ballooning.segmentation.void_intensity": args.void_intensity,
        "ballooning.segmentation.margin_px": args.margin_px,
        "ballooning.normalization.radius_um": getattr(args, "radius_um", None),
        "ballooning.ranking.top_n": getattr(args, "top_n", None),
        "ballooning.ranking.mid_n": getattr(args, "mid_n", None),
        "ballooning.ranking.low_n": getattr(args, "low_n", None),
        "ballooning.crops.padding_um": getattr(args, "padding_um", None),
    })
    if not cfg.slide_path:
        raise SystemExit("--slide is required")
    if not Path(cfg.slide_path).exists():
        raise SystemExit(f"no such slide: {cfg.slide_path}")
    return cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mashpath.features.ballooning.cli",
        description="Hepatocyte ballooning: candidate generation for review",
    )
    sub = p.add_subparsers(dest="command", required=True)

    ps = sub.add_parser(
        "segment",
        help="segment a handful of tiles and write native-resolution overlays",
    )
    _add_common(ps)
    ps.add_argument("--tiles", type=int, default=8,
                    help="tiles to segment, spread across the section")
    ps.set_defaults(func=cmd_segment)

    pr = sub.add_parser(
        "run",
        help="full pipeline: measure every cell, z-score against local "
             "neighbours, rank, and export a review package",
    )
    _add_common(pr)
    pr.add_argument("--radius-um", type=float, default=None,
                    help="neighbourhood radius the z-scores are taken over")
    pr.add_argument("--top-n", type=int, default=None,
                    help="top-ranked candidates sent for review")
    pr.add_argument("--mid-n", type=int, default=None,
                    help="mid-ranked candidates; do not set to 0 without "
                         "meaning to -- the review set needs near-misses")
    pr.add_argument("--low-n", type=int, default=None,
                    help="low-ranked candidates, so the reviewer sees negatives")
    pr.add_argument("--padding-um", type=float, default=None,
                    help="context around each cell in the review crop")
    pr.set_defaults(func=cmd_run)

    # --- chunked pass 1, for clusters -------------------------------------
    # `run` is one process over the whole slide, which is right on a laptop and
    # stalls on Fir at a reproducible point (see pipeline's chunk section).
    # These three split the same work so no process lives long enough to hit
    # it, and parallelise across array tasks as a side effect.
    pc = sub.add_parser(
        "count", help="print the tile count, to size a job array")
    _add_common(pc)
    pc.set_defaults(func=cmd_count)

    pk = sub.add_parser(
        "chunk",
        help="pass 1 over one tile range; run many in parallel, then finalize",
    )
    _add_common(pk)
    pk.add_argument("--start", type=int, required=True,
                    help="first tile index, inclusive")
    pk.add_argument("--stop", type=int, required=True,
                    help="last tile index, exclusive")
    pk.set_defaults(func=cmd_chunk)

    pf = sub.add_parser(
        "finalize",
        help="merge chunks, z-score against local neighbours, rank, export",
    )
    _add_common(pf)
    pf.add_argument("--radius-um", type=float, default=None)
    pf.add_argument("--top-n", type=int, default=None)
    pf.add_argument("--mid-n", type=int, default=None)
    pf.add_argument("--low-n", type=int, default=None)
    pf.add_argument("--padding-um", type=float, default=None)
    pf.add_argument("--allow-disjoint", action="store_true",
                    help="accept gaps between chunks. Only for a slide sampled "
                         "as deliberately spaced windows -- NOT for a chunk "
                         "whose job died, which is the case the gap check is "
                         "there to catch.")
    pf.set_defaults(func=cmd_finalize)
    return p


def _add_common(p: argparse.ArgumentParser) -> None:
    """Flags shared by every subcommand."""
    p.add_argument("--slide", default=None, help="path to a whole-slide image")
    p.add_argument("--config", action="append", default=None,
                   help="YAML config; repeat to layer left to right")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N tiles (0 = all)")
    p.add_argument("--tile-size", type=int, default=None)
    p.add_argument("--min-tissue-fraction", type=float, default=None)
    p.add_argument("--prob-threshold", type=float, default=None,
                   help="StarDist objectness cutoff (model default if unset)")
    p.add_argument("--nms-threshold", type=float, default=None)
    p.add_argument("--max-expansion-um", type=float, default=None,
                   help="hard cap on how far a cell grows from its nucleus")
    p.add_argument("--edge-weight", type=float, default=None,
                   help="0 = pure distance tessellation; higher follows eosin edges")
    p.add_argument("--void-intensity", type=int, default=None,
                   help="brighter than this is lumen/droplet, not cytoplasm")
    p.add_argument("--margin-px", type=int, default=None)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
