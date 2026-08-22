"""Command line entry point.

    python -m mashpath.cli info         --slide <path.svs>
    python -m mashpath.cli tissue       --slide <path.svs>
    python -m mashpath.cli tile         --slide <path.svs> --limit 20
    python -m mashpath.cli steatosis    --slide <path.svs> --config configs/core.yaml \
                                        --config configs/steatosis.yaml
    python -m mashpath.cli ballooning   --slide <path.svs> --plan
    python -m mashpath.cli inflammation --slide <path.svs> --plan
    python -m mashpath.cli review       --slide <path.svs> --feature ballooning

Any config value can be overridden from the command line; flags beat the YAML
files, which beat the built-in defaults. `--config` may be repeated and the
files are layered left to right, so shared settings and per-feature thresholds
can live in separate files.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import MashConfig as Config
from .core.io import SlideOutputs, write_index
from .core.slide import Slide, find_slides
from .core.tiling import count_candidate_tiles, iter_tiles
from .core.tissue import detect_tissue
from .core.viz import save_rgb, tissue_qc_figure
from .features.ballooning import detect as ballooning_detect
from .features.inflammation import detect as inflammation_detect
from .features.steatosis.pipeline import run_batch, run_slide
from .features.steatosis.survey import run_survey, run_sweep, score_sweep
from .review.export import load_verdicts, verdict_summary
from .review.scan import scan_slide


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--slide", required=True,
                   help="path to a whole slide image (.svs, .tif, .ndpi, ...)")
    _add_config_arg(p)
    p.add_argument("--output-dir", default=None, help="where to write results")
    _add_slide_args(p)


def _add_config_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", action="append", default=None, metavar="YAML",
                   help="YAML config file; repeat to layer files left to right")


def _add_slide_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("slide reading")
    g.add_argument("--mpp", type=float, default=None,
                   help="microns per pixel at level 0, for formats with no "
                        "embedded scale (plain TIFF). Sets both axes.")
    g.add_argument("--mpp-x", type=float, default=None)
    g.add_argument("--mpp-y", type=float, default=None)
    g.add_argument("--override-mpp", action="store_true", default=None,
                   help="use --mpp even when the slide carries its own scale")
    g.add_argument("--max-in-memory-mp", type=float, default=None,
                   help="largest TIFF level the tifffile fallback may decode "
                        "whole, in megapixels")


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


def _add_review_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("pathologist review export")
    g.add_argument("--context-um", type=float, default=None,
                   help="width of each candidate crop in microns; must contain "
                        "the neighbouring cells the call is made against")
    g.add_argument("--max-per-slide", type=int, default=None,
                   help="cap on candidates exported for review per slide")
    g.add_argument("--strata", type=int, default=None,
                   help="score bands sampled evenly, so the extremes appear")
    g.add_argument("--review-seed", type=int, default=None)


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
        prog="mashpath",
        description="Computational pathology for the mouse MASH model: "
                    "steatosis, hepatocyte ballooning, lobular inflammation",
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
    _add_config_arg(p_batch)
    _add_slide_args(p_batch)
    p_batch.add_argument("--output-dir", default=None, help="where to write results")
    p_batch.add_argument("--workers", type=int, default=1,
                         help="slides to process concurrently (per-slide console "
                              "output goes to outputs/<slide>/run.log)")
    _add_tissue_args(p_batch)
    _add_tiling_args(p_batch)
    _add_fat_args(p_batch)
    _add_qc_args(p_batch)

    p_tile = sub.add_parser(
        "tile", help="tile the tissue and write the tile index (all features)"
    )
    _add_common(p_tile)
    _add_tissue_args(p_tile)
    _add_tiling_args(p_tile)
    p_tile.add_argument("--limit", type=int, default=None,
                        help="stop after N tiles (0 = all)")

    # "run" is what this command was called before the restructure.
    p_run = sub.add_parser(
        "steatosis", aliases=["run"],
        help="steatosis: tissue -> tiles -> fat -> QC + CSV",
    )
    _add_common(p_run)
    _add_tissue_args(p_run)
    _add_tiling_args(p_run)
    _add_fat_args(p_run)
    _add_qc_args(p_run)

    for feature, helptext in (
        ("ballooning", "hepatocyte ballooning: propose candidates for review"),
        ("inflammation", "lobular inflammation: propose candidates for review"),
    ):
        p_f = sub.add_parser(feature, help=helptext)
        _add_common(p_f)
        _add_tissue_args(p_f)
        _add_tiling_args(p_f)
        _add_review_args(p_f)
        p_f.add_argument("--limit", type=int, default=None,
                         help="process only the first N tiles (0 = all)")
        p_f.add_argument("--plan", action="store_true",
                         help="stop after the tile plan, without running the "
                              "detector")

    p_review = sub.add_parser(
        "review", help="report on a review manifest and collect the confirmed set"
    )
    _add_common(p_review)
    p_review.add_argument("--feature", required=True,
                          choices=["steatosis", "ballooning", "inflammation"])
    p_review.add_argument("--manifest", default=None,
                          help="path to manifest.csv (default: the one under "
                               "outputs/<slide>/<feature>/review/)")
    p_review.add_argument("--confirmed-out", default=None,
                          help="write the confirmed candidates to this CSV")

    p_study = sub.add_parser(
        "study", help="build the pathologist annotation package from slides")
    p_study.add_argument("--slide-dir", action="append", required=True,
                         help="directory of slides; repeatable")
    p_study.add_argument("--out", default=None,
                         help="folder to send; not needed with --stage render, "
                              "which only fills the render pool")
    p_study.add_argument("--score-dir", action="append",
                         default=["bal_chunks", "bal_local"],
                         help="detector output; optional. Read for the record only -- the draw is 100%% uniform and the score steers nothing")
    # No --fields and no --enriched, on purpose. Fields per slide is derived
    # from --hours so that adding slides makes the draw wider rather than
    # longer, and the enriched stratum was removed outright (see fullset.py).
    p_study.add_argument("--hours", type=float, default=4.0,
                         help="reviewer time budget; fields per section are "
                              "derived from it. 0 means NO cap -- keep every "
                              "slide at full depth and let her stop when she "
                              "stops (default 4)")
    p_study.add_argument("--fields-per-section", type=int, default=None,
                         help="fix the depth of every section instead of "
                              "deriving it from --hours")
    p_study.add_argument("--warmup-out", default=None,
                         help="also build the short warm-up package here")
    p_study.add_argument("--warmup-slides", type=int, default=3,
                         help="slides held out of the main study for the warm-up")
    p_study.add_argument("--web", action="store_true",
                         help="build for hosting: no launchers, noindex, and a "
                              "README that does not tell her to double-click "
                              "something that is not there")
    p_study.add_argument("--reuse-frame", action="store_true",
                         help="reuse review_sets/study_frame.csv instead of "
                              "re-deriving tissue for every slide")
    p_study.add_argument("--work-dir", default="review_sets/ballooning_full")
    p_study.add_argument("--round", default="ballooning_full_v1")
    p_study.add_argument("--stage", choices=("render", "assemble", "both"),
                         default="both",
                         help="wave fetch: `render` per wave while the slides "
                              "are on disk, `assemble` once at the end over "
                              "every slide. Default `both`, for when every "
                              "slide is already local")
    p_study.add_argument("--append", action="store_true",
                         help="add these slides to an existing work dir rather than "
                              "replacing it -- for fetching in waves")

    p_bundle = sub.add_parser(
        "bundle",
        help="package a labelling set to send to a reviewer's own machine")
    p_bundle.add_argument("package", help="a built review_sets/<name> directory")
    p_bundle.add_argument("--out", required=True, help="folder to create")
    p_bundle.add_argument("--port", type=int, default=8000)
    p_bundle.add_argument("--brief", default="docs/BALLOONING_TILE_BRIEF.md")

    p_app = sub.add_parser(
        "review-app",
        help="serve the review app for pathologists (localhost only)",
    )
    p_app.add_argument("roots", nargs="*", default=["outputs"],
                       help="output directories to find review packages under "
                            "(default: outputs)")
    p_app.add_argument("--port", type=int, default=8000)
    p_app.add_argument("--host", default="127.0.0.1",
                       help="localhost by default, and it should stay there: "
                            "the server has no authentication and the crops "
                            "are patient-adjacent material")

    p_tapp = sub.add_parser(
        "tile-app",
        help="serve the TILE-level binary labelling app (localhost only)",
    )
    p_tapp.add_argument("roots", nargs="+",
                        help="tile package directories built by "
                             "`python -m mashpath.review.tileset`")
    p_tapp.add_argument("--port", type=int, default=8000)
    p_tapp.add_argument("--host", default="127.0.0.1",
                        help="localhost by default, and it should stay there: "
                             "the server has no authentication and the crops "
                             "are patient-adjacent material")

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
    _add_config_arg(p)
    _add_slide_args(p)
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

    mpp = getattr(args, "mpp", None)
    overrides = {
        "slide_path": getattr(args, "slide", None),
        "slide.mpp_x": getattr(args, "mpp_x", None) or mpp,
        "slide.mpp_y": getattr(args, "mpp_y", None) or mpp,
        "slide.override_mpp": getattr(args, "override_mpp", None),
        "slide.max_in_memory_megapixels": getattr(args, "max_in_memory_mp", None),
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
        "review.context_um": getattr(args, "context_um", None),
        "review.max_per_slide": getattr(args, "max_per_slide", None),
        "review.strata": getattr(args, "strata", None),
        "review.seed": getattr(args, "review_seed", None),
    }
    cfg.apply_overrides(overrides)

    # The macro lower bound doubles as the macro/micro split point, so the
    # micro band must stay below it.
    if cfg.fat.micro_max_area_um2 != cfg.fat.min_area_um2:
        cfg.fat.micro_max_area_um2 = cfg.fat.min_area_um2
    return cfg


def cmd_info(cfg: Config) -> int:
    with Slide(cfg.slide_path, cfg.slide) as slide:
        print(slide.describe())
        for lvl in range(slide.level_count):
            mx, my = slide.mpp_at_level(lvl)
            w, h = slide.level_dimensions[lvl]
            print(f"  level {lvl}: {w:>6} x {h:<6}  {mx:.4f} um/px")
    return 0


def cmd_tissue(cfg: Config) -> int:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Slide(cfg.slide_path, cfg.slide) as slide:
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

        outputs = SlideOutputs.for_slide(out_dir, slide.name)
        fig_path = save_rgb(outputs.qc_dir / "tissue_mask.png", fig)
        mask_path = save_rgb(
            outputs.qc_dir / "tissue_mask_raw.png",
            (tm.mask.astype("uint8") * 255)[..., None].repeat(3, 2),
        )
        cfg_path = outputs.write_config(cfg)

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


def cmd_tile(cfg: Config) -> int:
    """Tile the tissue once, for whichever feature comes next."""
    with Slide(cfg.slide_path, cfg.slide) as slide:
        print(slide.describe())
        print()
        t0 = time.time()
        tissue = detect_tissue(slide, cfg.tissue)
        print(f"tissue: {tissue.area_mm2:.2f} mm^2, "
              f"{tissue.coverage*100:.1f}% coverage [{time.time()-t0:.1f}s]")

        n_cand = count_candidate_tiles(slide, tissue, cfg.tiling)
        n_process = min(cfg.limit, n_cand) if cfg.limit else n_cand
        print(f"tiles:  {n_cand} of {cfg.tiling.tile_size}px positions pass "
              f"tissue_fraction >= {cfg.tiling.min_tissue_fraction}")
        if cfg.limit:
            print(f"        --limit {cfg.limit} -> writing {n_process}")

        outputs = SlideOutputs.for_slide(cfg.output_dir, slide.name)
        mx, my = slide.mpp_at_level(cfg.tiling.level)
        rows = []
        t0 = time.time()
        for tile in iter_tiles(slide, tissue, cfg.tiling, limit=cfg.limit):
            rows.append({
                "tile": tile.name,
                "x": tile.x, "y": tile.y, "size": tile.size,
                "level": cfg.tiling.level,
                "tissue_fraction": round(tile.tissue_fraction, 4),
                "width_um": round(tile.size * mx, 2),
                "height_um": round(tile.size * my, 2),
            })
            if cfg.tiling.save_tiles:
                outputs.write_tile(tile.read(slide, cfg.tiling.level), tile.name)

        index = write_index(outputs, rows, "tiles")
        outputs.write_config(cfg)
        print(f"\n  {len(rows)} tiles in {time.time()-t0:.1f}s")
        print(f"  tile index -> {index}")
        if cfg.tiling.save_tiles:
            print(f"  tile images -> {outputs.tiles_dir}")
    return 0


CANDIDATE_FEATURES = {
    "ballooning": (ballooning_detect.detect_candidates, "ballooning"),
    "inflammation": (inflammation_detect.detect_candidates, "inflammation"),
}

# Features whose generation is NOT a per-tile call and so cannot go through
# `scan_slide`. Ballooning z-scores each cell against a ~150 um neighbourhood,
# which is wider than a 254 um tile, so it runs as two passes over the slide.
# Routing it through the per-tile path raised NotImplementedError -- meaning
# the command advertised at the top of this file crashed on every invocation.
SLIDE_LEVEL_FEATURES = {
    "ballooning": "mashpath.features.ballooning.pipeline:run_slide",
}


def cmd_candidates(cfg: Config, feature: str, plan_only: bool) -> int:
    """Propose candidates for a feature that needs pathologist confirmation."""
    if feature in SLIDE_LEVEL_FEATURES and not plan_only:
        module, func = SLIDE_LEVEL_FEATURES[feature].split(":")
        import importlib

        getattr(importlib.import_module(module), func)(cfg)
        return 0
    detector, section = CANDIDATE_FEATURES[feature]
    scan_slide(cfg, feature, detector, getattr(cfg, section), plan_only=plan_only)
    return 0


def cmd_review_app(args: argparse.Namespace) -> int:
    """Serve the review app over the given output directories."""
    from .review.app import serve

    serve(args.roots, port=args.port, host=args.host)
    return 0


def cmd_review(cfg: Config, args: argparse.Namespace) -> int:
    slide_id = Path(cfg.slide_path).stem
    outputs = SlideOutputs.for_slide(cfg.output_dir, slide_id)
    manifest = Path(args.manifest) if args.manifest else (
        outputs.review_dir(args.feature) / "manifest.csv"
    )
    if not manifest.exists():
        raise SystemExit(
            f"no review manifest at {manifest}\n"
            f"run `{args.feature} --slide {cfg.slide_path}` first to generate one"
        )

    counts = verdict_summary(manifest)
    total = sum(counts.values())
    print(f"review manifest: {manifest}")
    print(f"  candidates:   {total}")
    for k in ("confirmed", "rejected", "unreviewed", "other"):
        pct = counts[k] / total * 100 if total else 0.0
        print(f"  {k:<12} {counts[k]:>6}  ({pct:.1f}%)")
    vpath = manifest.parent / "verdicts.csv"
    if counts["unreviewed"] == total:
        if vpath.exists():
            # The manifest's own column is the offline Excel route. The web app
            # writes to the verdict store instead, so an all-unreviewed
            # manifest here means "reviewed in the app", not "not reviewed".
            print("\n  manifest verdict column empty -- verdicts were recorded "
                  "in the app; see the store below")
        else:
            print("\n  nothing reviewed yet -- see REVIEW.md next to the "
                  "manifest, or serve it with `mashpath review-app`")

    # Verdicts recorded by the web app live beside the manifest, not in it.
    from .review import verdicts as verdicts_mod

    if vpath.exists():
        print()
        print(verdicts_mod.agreement_report(vpath))

    confirmed = load_verdicts(manifest)
    if args.confirmed_out:
        import pandas as pd

        out = Path(args.confirmed_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([c.to_row() for c in confirmed]).to_csv(out, index=False)
        print(f"\n  {len(confirmed)} confirmed candidate(s) -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Serving the app is the one command with no slide and no config: it reads
    # review packages that a pipeline already wrote.
    if args.command == "review-app":
        return cmd_review_app(args)
    if args.command == "study":
        from .review.study import build_study
        if args.stage != "render" and not args.out:
            build_parser().error("--out is required unless --stage render")
        build_study(args.slide_dir, args.out, work_dir=args.work_dir,
                    score_dirs=args.score_dir, round_name=args.round,
                    append=args.append, hours=args.hours,
                    warmup_dir=args.warmup_out,
                    warmup_slides=args.warmup_slides,
                    reuse_frame=args.reuse_frame, stage=args.stage,
                    fields_per_section=args.fields_per_section,
                    for_web=args.web)
        return 0

    if args.command == "bundle":
        from .review.bundle import build as build_bundle
        build_bundle(args.package, args.out, args.port, args.brief)
        return 0

    if args.command == "tile-app":
        from .review.tiles_app import serve as serve_tiles

        serve_tiles(args.roots, port=args.port, host=args.host)
        return 0
    cfg = load_config(args)
    if args.command == "info":
        return cmd_info(cfg)
    if args.command == "tissue":
        return cmd_tissue(cfg)
    if args.command == "tile":
        return cmd_tile(cfg)
    if args.command in ("steatosis", "run"):
        run_slide(cfg, verbose_tiles=args.print_tiles)
        return 0
    if args.command in CANDIDATE_FEATURES:
        return cmd_candidates(cfg, args.command, args.plan)
    if args.command == "review":
        return cmd_review(cfg, args)
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
