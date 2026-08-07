"""Command line entry point.

    python -m steatosis.cli info   --slide <path.svs>
    python -m steatosis.cli tissue --slide <path.svs> [--config configs/default.yaml]

Any config value can be overridden from the command line; flags beat the YAML
file, which beats the built-in defaults.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import Config
from .pipeline import find_slides, run_batch, run_slide
from .slide import Slide
from .survey import run_survey, run_sweep, score_sweep
from .tissue import detect_tissue
from .viz import save_rgb, tissue_qc_figure


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--slide", required=True, help="path to an .svs whole slide image")
    p.add_argument("--config", default=None, help="YAML config file")
    p.add_argument("--output-dir", default=None, help="where to write results")


def _add_tissue_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("tissue detection")
    g.add_argument("--tissue-level", type=int, default=None,
                   help="pyramid level for detection (default 2)")
    g.add_argument("--tissue-method", choices=["otsu", "fixed"], default=None,
                   help="saturation threshold strategy")
    g.add_argument("--saturation-threshold", type=int, default=None,
                   help="saturation cutoff 0-255 when --tissue-method fixed")
    g.add_argument("--max-value", type=int, default=None,
                   help="HSV V above this is forced to background")
    g.add_argument("--close-radius-um", type=float, default=None)
    g.add_argument("--open-radius-um", type=float, default=None)
    g.add_argument("--min-object-area-um2", type=float, default=None,
                   help="drop tissue components smaller than this (speck removal)")
    g.add_argument("--max-hole-area-um2", type=float, default=None,
                   help="fill interior holes smaller than this")
    g.add_argument("--keep-largest-n", type=int, default=None,
                   help="keep only the N largest components (0 = all)")
    g.add_argument("--thumbnail-max-dim", type=int, default=None,
                   help="long edge of the QC overlay image")


def _add_tiling_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("tiling")
    g.add_argument("--tile-size", type=int, default=None, help="tile edge in px (default 512)")
    g.add_argument("--stride", type=int, default=None, help="tile stride in px")
    g.add_argument("--min-tissue-fraction", type=float, default=None,
                   help="skip tiles below this tissue fraction (default 0.5)")
    g.add_argument("--no-save-tiles", dest="save_tiles", action="store_false",
                   default=None, help="do not write tile/mask PNGs")


def _add_fat_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("fat detection")
    g.add_argument("--fat-method", choices=["fixed", "otsu"], default=None,
                   help="white-region threshold strategy")
    g.add_argument("--white-threshold", type=int, default=None,
                   help="grayscale cutoff 0-255 when --fat-method fixed")
    g.add_argument("--open-radius-px", type=int, default=None)
    g.add_argument("--close-radius-px", type=int, default=None)
    g.add_argument("--no-fill-holes", dest="fill_holes", action="store_false",
                   default=None, help="do not fill holes inside droplets")
    g.add_argument("--min-area-um2", type=float, default=None,
                   help="macro lower bound AND the macro/micro split point")
    g.add_argument("--max-area-um2", type=float, default=None)
    g.add_argument("--circularity-min", type=float, default=None,
                   help="4*pi*A/P^2 cutoff for macro droplets")
    g.add_argument("--solidity-min", type=float, default=None,
                   help="area / convex hull area cutoff for macro droplets")
    g.add_argument("--max-eccentricity", type=float, default=None,
                   help="ellipse eccentricity ceiling for macro droplets; "
                        "rejects elongated vessel lumens (1.0 disables)")
    g.add_argument("--include-microvesicular", dest="include_microvesicular",
                   action="store_true", default=None,
                   help="write micro droplets into the exported pseudo-labels")
    g.add_argument("--micro-min-area-um2", type=float, default=None)
    g.add_argument("--micro-circularity-min", type=float, default=None)
    g.add_argument("--micro-solidity-min", type=float, default=None)
    g.add_argument("--exclude-border", dest="exclude_border",
                   action="store_true", default=None,
                   help="drop droplets touching the tile edge")


def _add_qc_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("QC")
    g.add_argument("--contact-sheet-tiles", type=int, default=None,
                   help="number of random tiles in the contact sheet")
    g.add_argument("--contact-sheet-cols", type=int, default=None)
    g.add_argument("--random-seed", type=int, default=None)
    g.add_argument("--print-tiles", type=int, default=20,
                   help="how many per-tile stat lines to print")
    g.add_argument("--limit", type=int, default=None,
                   help="process only the first N tiles (0 = all)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="steatosis",
        description="Weakly supervised steatosis quantification -- pseudo-labeling pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="print slide metadata and exit")
    _add_common(p_info)

    p_tissue = sub.add_parser("tissue", help="detect tissue and write a QC overlay")
    _add_common(p_tissue)
    _add_tissue_args(p_tissue)

    p_batch = sub.add_parser(
        "batch", help="run the full pipeline over every slide in a folder"
    )
    p_batch.add_argument("--slide-dir", required=True,
                         help="folder containing whole slide images")
    p_batch.add_argument("--pattern", default="*",
                         help="glob to select slides within the folder")
    p_batch.add_argument("--config", default=None, help="YAML config file")
    p_batch.add_argument("--output-dir", default=None, help="where to write results")
    p_batch.add_argument("--workers", type=int, default=1,
                         help="slides to process concurrently (per-slide console "
                              "output goes to outputs/<slide>/run.log)")
    _add_tissue_args(p_batch)
    _add_tiling_args(p_batch)
    _add_fat_args(p_batch)
    _add_qc_args(p_batch)

    p_run = sub.add_parser(
        "run", help="full pipeline: tissue -> tiles -> fat -> QC + CSV"
    )
    _add_common(p_run)
    _add_tissue_args(p_run)
    _add_tiling_args(p_run)
    _add_fat_args(p_run)
    _add_qc_args(p_run)

    p_survey = sub.add_parser(
        "survey",
        help="estimate fat per slide from N random tiles (fast cohort screen)",
    )
    _add_group_args(p_survey)
    _add_tissue_args(p_survey)
    _add_tiling_args(p_survey)
    _add_fat_args(p_survey)

    p_sweep = sub.add_parser(
        "sweep",
        help="score a parameter grid on how well it separates two cohorts",
    )
    _add_group_args(p_sweep)
    _add_tissue_args(p_sweep)
    _add_tiling_args(p_sweep)
    _add_fat_args(p_sweep)
    g = p_sweep.add_argument_group("grid")
    g.add_argument("--white", default="200,210,220,230",
                   help="comma-separated white_threshold values")
    g.add_argument("--circ", default="0.60,0.65,0.70",
                   help="comma-separated circularity_min values")
    g.add_argument("--sol", default="0.85,0.90,0.95",
                   help="comma-separated solidity_min values")
    g.add_argument("--min-area", default=None,
                   help="comma-separated min_area_um2 values "
                        "(also moves the macro/micro split point)")
    g.add_argument("--ecc", default=None,
                   help="comma-separated max_eccentricity values")
    g.add_argument("--positive", default=None,
                   help="group name treated as known-positive (default: first)")
    g.add_argument("--negative", default=None,
                   help="group name treated as known-negative (default: second)")
    g.add_argument("--top", type=int, default=15,
                   help="how many scored parameter cells to print")

    return parser


def _add_group_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--group", action="append", required=True, metavar="NAME=DIR",
                   help="labelled cohort folder; repeat for each cohort")
    p.add_argument("--pattern", default="*", help="glob to select slides in each folder")
    p.add_argument("--tiles", type=int, default=60,
                   help="random tiles sampled per slide (0 = all candidates)")
    p.add_argument("--seed", type=int, default=0, help="sampling seed")
    p.add_argument("--workers", type=int, default=1, help="parallel slide workers")
    p.add_argument("--out", default=None, help="CSV output path")
    p.add_argument("--config", default=None, help="YAML config file")
    p.add_argument("--output-dir", default=None, help="where to write results")


def _parse_groups(specs: list[str], pattern: str) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--group must be NAME=DIR, got {spec!r}")
        name, _, folder = spec.partition("=")
        slides = find_slides(folder, pattern)
        if not slides:
            raise SystemExit(f"no whole-slide images in {folder}")
        groups[name] = slides
    return groups


def load_config(args: argparse.Namespace) -> Config:
    cfg = Config.from_yaml(args.config) if args.config else Config()

    overrides = {
        "slide_path": getattr(args, "slide", None),
        "output_dir": getattr(args, "output_dir", None),
        "tissue.level": getattr(args, "tissue_level", None),
        "tissue.method": getattr(args, "tissue_method", None),
        "tissue.saturation_threshold": getattr(args, "saturation_threshold", None),
        "tissue.max_value": getattr(args, "max_value", None),
        "tissue.close_radius_um": getattr(args, "close_radius_um", None),
        "tissue.open_radius_um": getattr(args, "open_radius_um", None),
        "tissue.min_object_area_um2": getattr(args, "min_object_area_um2", None),
        "tissue.max_hole_area_um2": getattr(args, "max_hole_area_um2", None),
        "tissue.keep_largest_n": getattr(args, "keep_largest_n", None),
        "qc.thumbnail_max_dim": getattr(args, "thumbnail_max_dim", None),
        "limit": getattr(args, "limit", None),
        "tiling.tile_size": getattr(args, "tile_size", None),
        "tiling.stride": getattr(args, "stride", None),
        "tiling.min_tissue_fraction": getattr(args, "min_tissue_fraction", None),
        "tiling.save_tiles": getattr(args, "save_tiles", None),
        "fat.method": getattr(args, "fat_method", None),
        "fat.white_threshold": getattr(args, "white_threshold", None),
        "fat.open_radius_px": getattr(args, "open_radius_px", None),
        "fat.close_radius_px": getattr(args, "close_radius_px", None),
        "fat.fill_holes": getattr(args, "fill_holes", None),
        "fat.min_area_um2": getattr(args, "min_area_um2", None),
        "fat.max_area_um2": getattr(args, "max_area_um2", None),
        "fat.circularity_min": getattr(args, "circularity_min", None),
        "fat.solidity_min": getattr(args, "solidity_min", None),
        "fat.max_eccentricity": getattr(args, "max_eccentricity", None),
        "fat.include_microvesicular": getattr(args, "include_microvesicular", None),
        "fat.micro_min_area_um2": getattr(args, "micro_min_area_um2", None),
        "fat.micro_circularity_min": getattr(args, "micro_circularity_min", None),
        "fat.micro_solidity_min": getattr(args, "micro_solidity_min", None),
        "fat.exclude_border": getattr(args, "exclude_border", None),
        "qc.contact_sheet_tiles": getattr(args, "contact_sheet_tiles", None),
        "qc.contact_sheet_cols": getattr(args, "contact_sheet_cols", None),
        "qc.random_seed": getattr(args, "random_seed", None),
    }
    cfg.apply_overrides(overrides)

    # The macro lower bound doubles as the macro/micro split point, so the
    # micro band must stay below it.
    if cfg.fat.micro_max_area_um2 != cfg.fat.min_area_um2:
        cfg.fat.micro_max_area_um2 = cfg.fat.min_area_um2
    return cfg


def cmd_info(cfg: Config) -> int:
    with Slide(cfg.slide_path) as slide:
        print(slide.describe())
        for lvl in range(slide.level_count):
            mx, my = slide.mpp_at_level(lvl)
            w, h = slide.level_dimensions[lvl]
            print(f"  level {lvl}: {w:>6} x {h:<6}  {mx:.4f} um/px")
    return 0


def cmd_tissue(cfg: Config) -> int:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Slide(cfg.slide_path) as slide:
        print(slide.describe())
        print()

        t0 = time.time()
        tm = detect_tissue(slide, cfg.tissue)
        elapsed = time.time() - t0

        mask_h, mask_w = tm.shape
        slide_w, slide_h = slide.dimensions
        slide_area_mm2 = (slide_w * slide.mpp_x) * (slide_h * slide.mpp_y) / 1e6

        print("tissue detection")
        print(f"  level:          {tm.level} (downsample {tm.downsample:.1f}x)")
        print(f"  mask size:      {mask_w} x {mask_h}")
        print(f"  method:         {cfg.tissue.method} (saturation > {tm.threshold:.1f})")
        print(f"  components:     {tm.component_count}")
        print(f"  coverage:       {tm.coverage * 100:.1f}% of slide area")
        print(f"  tissue area:    {tm.area_mm2:.2f} mm^2  (slide {slide_area_mm2:.2f} mm^2)")
        print(f"  elapsed:        {elapsed:.1f}s")

        thumb = slide.thumbnail(cfg.qc.thumbnail_max_dim)
        caption = (
            f"{slide.name} | level {tm.level} | {cfg.tissue.method} "
            f"sat>{tm.threshold:.0f} | {tm.area_mm2:.1f} mm^2 tissue "
            f"({tm.coverage * 100:.1f}%) | {tm.component_count} components"
        )
        fig = tissue_qc_figure(
            thumb,
            tm.mask,
            outline_color=tuple(cfg.qc.outline_color),
            thickness=cfg.qc.outline_thickness,
            caption=caption,
        )

        qc_dir = out_dir / slide.name / "qc"
        fig_path = save_rgb(qc_dir / "tissue_mask.png", fig)
        mask_path = save_rgb(
            qc_dir / "tissue_mask_raw.png", (tm.mask.astype("uint8") * 255)[..., None].repeat(3, 2)
        )
        cfg_path = out_dir / slide.name / "config_used.yaml"
        cfg.save(cfg_path)

        print()
        print(f"  QC overlay ->   {fig_path}")
        print(f"  raw mask   ->   {mask_path}")
        print(f"  config     ->   {cfg_path}")
    return 0


def _print_survey(df) -> None:
    print()
    print("=" * 104)
    print("COHORT SURVEY -- random-tile estimate, not a full run")
    print("=" * 104)
    ok = df[df.get("error").isna()] if "error" in df else df
    print(f"{'group':<8}{'slide':<20}{'tissue mm2':>11}{'cand':>7}{'smpl':>6}"
          f"{'otsu':>6}{'macro %':>9}{'micro %':>9}{'macro n/tile':>14}")
    print("-" * 104)
    for _, r in ok.iterrows():
        print(f"{r['group']:<8}{r['slide']:<20}{r['tissue_mm2']:>11.2f}"
              f"{int(r['cand_tiles']):>7}{int(r['sampled']):>6}{r['otsu']:>6.0f}"
              f"{r['macro_pct']:>9.2f}{r['micro_pct']:>9.2f}{r['macro_n_per_tile']:>14.1f}")
    print("-" * 104)
    for g, sub in ok.groupby("group"):
        print(f"{g:<8}{'n=' + str(len(sub)):<20}{sub['tissue_mm2'].sum():>11.2f}"
              f"{'':>7}{'':>6}{'':>6}{sub['macro_pct'].mean():>9.2f}"
              f"{sub['micro_pct'].mean():>9.2f}"
              f"   (mean; range {sub['macro_pct'].min():.2f}-{sub['macro_pct'].max():.2f})")
    if "error" in df:
        bad = df[df["error"].notna()]
        for _, r in bad.iterrows():
            print(f"  FAILED {r['slide']}: {r['error']}")


def cmd_survey(cfg: Config, args: argparse.Namespace) -> int:
    groups = _parse_groups(args.group, args.pattern)
    total = sum(len(v) for v in groups.values())
    print(f"survey: {total} slide(s) in {len(groups)} group(s), "
          f"{args.tiles} random tiles each, seed {args.seed}")
    for g, paths in groups.items():
        print(f"  {g}: {len(paths)} slides")
    df = run_survey(groups, cfg, args.tiles, args.seed, args.workers)
    _print_survey(df)

    out = Path(args.out) if args.out else Path(cfg.output_dir) / "survey.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    cfg.save(out.with_suffix(".config.yaml"))
    print(f"\n  survey CSV -> {out}")
    print(f"  config     -> {out.with_suffix('.config.yaml')}")
    return 0


def cmd_sweep(cfg: Config, args: argparse.Namespace) -> int:
    groups = _parse_groups(args.group, args.pattern)
    names = list(groups)
    positive = args.positive or names[0]
    negative = args.negative or (names[1] if len(names) > 1 else names[0])
    if positive not in groups or negative not in groups:
        raise SystemExit(f"--positive/--negative must name a group: {names}")

    def _vals(spec, default):
        return [float(v) for v in spec.split(",") if v.strip()] if spec else [default]

    whites = [int(v) for v in args.white.split(",") if v.strip()]
    axes = {
        "circularity_min": _vals(args.circ, cfg.fat.circularity_min),
        "solidity_min": _vals(args.sol, cfg.fat.solidity_min),
        "min_area_um2": _vals(args.min_area, cfg.fat.min_area_um2),
        "max_eccentricity": _vals(args.ecc, cfg.fat.max_eccentricity),
    }

    total = sum(len(v) for v in groups.values())
    n_cells = len(whites)
    for v in axes.values():
        n_cells *= len(v)
    print(f"sweep: {n_cells} parameter cells over {total} slides, "
          f"{args.tiles} tiles each")
    print(f"  positive (fat expected): {positive}  n={len(groups[positive])}")
    print(f"  negative (fat-free):     {negative}  n={len(groups[negative])}")
    print(f"  white_threshold={whites}")
    for k, v in axes.items():
        print(f"  {k}={v}")

    raw = run_sweep(groups, cfg, args.tiles, whites, axes, args.seed, args.workers)
    scored = score_sweep(raw, positive, negative)

    param_cols = [c for c in scored.columns
                  if c not in {"pos_mean", "pos_min", "pos_median", "neg_mean",
                               "neg_max", "neg_median", "margin", "ratio",
                               "worst_ratio"}]
    short = {"white": "white", "circularity_min": "circ", "solidity_min": "sol",
             "min_area_um2": "minA", "max_eccentricity": "ecc"}

    print()
    print("=" * 108)
    print(f"PARAMETER GRID -- sorted by margin (weakest {positive} slide "
          f"minus worst {negative} slide)")
    print("=" * 108)
    head = "".join(f"{short.get(c, c):>7}" for c in param_cols)
    print(head + f"{'pos mean':>10}{'pos min':>9}{'neg mean':>10}{'neg max':>9}"
                 f"{'margin':>9}{'ratio':>8}{'worst':>8}")
    print("-" * 108)
    for _, r in scored.head(args.top).iterrows():
        cells = "".join(
            f"{r[c]:>7.0f}" if c in ("white", "min_area_um2") else f"{r[c]:>7.2f}"
            for c in param_cols
        )
        print(cells + f"{r['pos_mean']:>10.2f}{r['pos_min']:>9.2f}"
                      f"{r['neg_mean']:>10.2f}{r['neg_max']:>9.2f}"
                      f"{r['margin']:>9.2f}{r['ratio']:>8.1f}{r['worst_ratio']:>8.1f}")

    out = Path(args.out) if args.out else Path(cfg.output_dir) / "sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out, index=False)
    raw.to_csv(out.with_name(out.stem + "_per_slide.csv"), index=False)
    print(f"\n  scored grid   -> {out}")
    print(f"  per-slide raw -> {out.with_name(out.stem + '_per_slide.csv')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args)
    if args.command == "info":
        return cmd_info(cfg)
    if args.command == "tissue":
        return cmd_tissue(cfg)
    if args.command == "run":
        run_slide(cfg, verbose_tiles=args.print_tiles)
        return 0
    if args.command == "batch":
        run_batch(cfg, args.slide_dir, args.pattern,
                  verbose_tiles=args.print_tiles, workers=args.workers)
        return 0
    if args.command == "survey":
        return cmd_survey(cfg, args)
    if args.command == "sweep":
        return cmd_sweep(cfg, args)
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    sys.exit(main())
