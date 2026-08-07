"""End-to-end single-slide run: tissue -> tiles -> fat -> QC + CSV.

Tiles are processed one at a time and released; nothing accumulates except
scalar per-tile statistics, droplet areas, and the handful of QC figures
sampled for the contact sheet.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .fat import MACRO, MICRO, FatResult, detect_fat
from .slide import Slide
from .tiles import Tile, count_candidate_tiles, iter_tiles
from .tissue import TissueMask, detect_tissue
from .viz import grid, label_panel, save_rgb, tile_qc_figure


def _tile_row(tile: Tile, res: FatResult) -> dict[str, Any]:
    return {
        "tile_x": tile.x,
        "tile_y": tile.y,
        "tile_size": tile.size,
        "tissue_fraction": round(tile.tissue_fraction, 4),
        "tissue_area_um2": round(res.tissue_area_um2, 1),
        "threshold": round(res.threshold, 1),
        "macro_count": res.count(MACRO),
        "macro_area_um2": round(res.area_um2(MACRO), 1),
        "macro_fat_fraction": round(res.fat_fraction(MACRO), 6),
        "macro_mean_area_um2": round(res.mean_area_um2(MACRO), 2),
        "macro_median_area_um2": round(res.median_area_um2(MACRO), 2),
        "micro_count": res.count(MICRO),
        "micro_area_um2": round(res.area_um2(MICRO), 1),
        "micro_fat_fraction": round(res.fat_fraction(MICRO), 6),
        "micro_mean_area_um2": round(res.mean_area_um2(MICRO), 2),
        "micro_median_area_um2": round(res.median_area_um2(MICRO), 2),
        "total_count": res.count(),
        "total_area_um2": round(res.area_um2(), 1),
        "total_fat_fraction": round(res.fat_fraction(), 6),
        "border_count": res.border_count,
        "border_area_um2": round(res.border_area_um2, 1),
        "rej_too_small": res.rejected.get("too_small", 0),
        "rej_too_large": res.rejected.get("too_large", 0),
        "rej_low_circularity": res.rejected.get("low_circularity", 0),
        "rej_low_solidity": res.rejected.get("low_solidity", 0),
        "rej_elongated": res.rejected.get("elongated", 0),
        "rej_border": res.rejected.get("border", 0),
    }


def _histogram(areas: np.ndarray, bins: int, label_txt: str, indent: str = "  ") -> str:
    if areas.size == 0:
        return f"{indent}{label_txt}: (none)"
    lo, hi = float(areas.min()), float(areas.max())
    if hi <= lo:
        hi = lo + 1.0
    counts, edges = np.histogram(areas, bins=bins, range=(lo, hi))
    peak = max(1, counts.max())
    lines = [f"{indent}{label_txt}  (n={areas.size}, {lo:.1f}-{hi:.1f} um^2)"]
    for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(round(40 * c / peak))
        lines.append(f"{indent}  {e0:8.1f}-{e1:8.1f} um^2 | {c:6d} {bar}")
    return "\n".join(lines)


def _tile_caption(res: FatResult) -> str:
    return (
        f"macro {res.count(MACRO)} ({res.fat_fraction(MACRO)*100:.1f}%)  "
        f"micro {res.count(MICRO)} ({res.fat_fraction(MICRO)*100:.1f}%)  "
        f"total {res.fat_fraction()*100:.1f}%"
    )


def run_slide(cfg: Config, verbose_tiles: int = 20, detail: bool = True) -> dict[str, Any]:
    out_root = Path(cfg.output_dir)
    slide = Slide(cfg.slide_path)
    try:
        print(slide.describe())
        print()

        # ---- tissue ----
        t0 = time.time()
        tissue: TissueMask = detect_tissue(slide, cfg.tissue)
        print(
            f"tissue: {tissue.area_mm2:.2f} mm^2, {tissue.coverage*100:.1f}% coverage, "
            f"{tissue.component_count} component(s), sat>{tissue.threshold:.0f} "
            f"[{time.time()-t0:.1f}s]"
        )

        out_dir = out_root / slide.name
        qc_dir = out_dir / "qc"
        tiles_dir = out_dir / "tiles"
        masks_dir = out_dir / "masks"
        results_dir = out_dir / "results"
        for d in (qc_dir, results_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ---- tile plan (no pixels read) ----
        t0 = time.time()
        n_candidates = count_candidate_tiles(slide, tissue, cfg.tiling)
        n_process = min(cfg.limit, n_candidates) if cfg.limit else n_candidates
        print(
            f"tiles:  {n_candidates} of "
            f"{(slide.dimensions[0]//cfg.tiling.stride)*(slide.dimensions[1]//cfg.tiling.stride)} "
            f"grid positions pass tissue_fraction >= {cfg.tiling.min_tissue_fraction} "
            f"[{time.time()-t0:.1f}s]"
        )
        if cfg.limit:
            print(f"        --limit {cfg.limit} -> processing {n_process}")
        if n_process == 0:
            raise RuntimeError("no tiles passed the tissue fraction threshold")

        rng = random.Random(cfg.qc.random_seed)
        k = min(cfg.qc.contact_sheet_tiles, n_process)
        sample_idx = set(rng.sample(range(n_process), k))

        um2_px = slide.um2_per_pixel(cfg.tiling.level)

        rows: list[dict[str, Any]] = []
        macro_areas: list[np.ndarray] = []
        micro_areas: list[np.ndarray] = []
        sheet_panels: list[np.ndarray] = []
        sampled_reports: list[str] = []
        border_area_total = 0.0
        border_count_total = 0

        print()
        print(
            f"{'#':>5} {'x':>7} {'y':>7} {'tis%':>5} "
            f"{'macroN':>7} {'macro%':>7} {'microN':>7} {'micro%':>7} "
            f"{'tot%':>6} {'mean':>7} {'med':>7}"
        )
        print("-" * 84)

        t0 = time.time()
        for i, tile in enumerate(iter_tiles(slide, tissue, cfg.tiling, limit=cfg.limit)):
            rgb = tile.read(slide, cfg.tiling.level)
            tmask = tissue.tile_mask(tile.x, tile.y, tile.size, tile.size)
            res = detect_fat(rgb, tmask, cfg.fat, um2_px)

            rows.append(_tile_row(tile, res))
            macro_areas.append(res.areas(MACRO))
            micro_areas.append(res.areas(MICRO))
            border_area_total += res.border_area_um2
            border_count_total += res.border_count

            if cfg.tiling.save_tiles:
                tiles_dir.mkdir(parents=True, exist_ok=True)
                masks_dir.mkdir(parents=True, exist_ok=True)
                save_rgb(tiles_dir / f"{tile.name}.png", rgb)
                save_rgb(
                    masks_dir / f"{tile.name}_mask.png",
                    (res.label_mask.astype(np.uint8) * 255)[..., None].repeat(3, 2),
                )

            if i in sample_idx:
                fig = tile_qc_figure(
                    rgb, res.macro_mask, res.micro_mask,
                    macro_color=tuple(cfg.qc.macro_color),
                    micro_color=tuple(cfg.qc.micro_color),
                    thickness=cfg.qc.outline_thickness,
                    caption=f"{tile.name}  {_tile_caption(res)}",
                )
                sheet_panels.append(fig)
                if cfg.qc.save_per_tile_qc:
                    save_rgb(qc_dir / "tiles" / f"{tile.name}_qc.png", fig)
                sampled_reports.append(
                    f"\n[{tile.name}]  {_tile_caption(res)}\n"
                    + _histogram(res.areas(MACRO), 8, "macro droplet sizes", "    ")
                    + "\n"
                    + _histogram(res.areas(MICRO), 8, "micro droplet sizes", "    ")
                )

            if i < verbose_tiles:
                print(
                    f"{i:>5} {tile.x:>7} {tile.y:>7} {tile.tissue_fraction*100:>5.0f} "
                    f"{res.count(MACRO):>7} {res.fat_fraction(MACRO)*100:>7.2f} "
                    f"{res.count(MICRO):>7} {res.fat_fraction(MICRO)*100:>7.2f} "
                    f"{res.fat_fraction()*100:>6.2f} "
                    f"{res.mean_area_um2(MACRO):>7.1f} {res.median_area_um2(MACRO):>7.1f}"
                )
            elif i == verbose_tiles:
                print(f"      ... {n_process - verbose_tiles} more tiles (see CSV)")

        elapsed = time.time() - t0

        # ---- aggregate ----
        df = pd.DataFrame(rows)
        csv_path = results_dir / "tiles.csv"
        df.to_csv(csv_path, index=False)

        all_macro = np.concatenate(macro_areas) if macro_areas else np.array([])
        all_micro = np.concatenate(micro_areas) if micro_areas else np.array([])
        tissue_um2 = float(df["tissue_area_um2"].sum())
        macro_um2 = float(df["macro_area_um2"].sum())
        micro_um2 = float(df["micro_area_um2"].sum())

        summary = {
            "slide": slide.name,
            "tiles_processed": int(len(df)),
            "tiles_candidate": int(n_candidates),
            "slide_tissue_area_mm2": tissue.area_mm2,
            "processed_tissue_area_mm2": tissue_um2 / 1e6,
            "macro_count": int(df["macro_count"].sum()),
            "micro_count": int(df["micro_count"].sum()),
            "macro_area_mm2": macro_um2 / 1e6,
            "micro_area_mm2": micro_um2 / 1e6,
            "macro_fat_fraction": macro_um2 / tissue_um2 if tissue_um2 else 0.0,
            "micro_fat_fraction": micro_um2 / tissue_um2 if tissue_um2 else 0.0,
            "total_fat_fraction": (macro_um2 + micro_um2) / tissue_um2 if tissue_um2 else 0.0,
            "border_count": border_count_total,
            "border_area_um2": border_area_total,
            "border_fraction_of_fat": (
                border_area_total / (macro_um2 + micro_um2)
                if (macro_um2 + micro_um2) else 0.0
            ),
            "seconds": elapsed,
        }

        # ---- contact sheet ----
        if sheet_panels:
            sheet = grid(sheet_panels, cols=cfg.qc.contact_sheet_cols, gap=14)
            header = (
                f"{slide.name} | {len(df)} tiles | macro={summary['macro_fat_fraction']*100:.2f}% "
                f"micro={summary['micro_fat_fraction']*100:.2f}% "
                f"total={summary['total_fat_fraction']*100:.2f}% | "
                f"red=macro cyan=micro | thr={cfg.fat.method}:{cfg.fat.white_threshold} "
                f"area={cfg.fat.min_area_um2}-{cfg.fat.max_area_um2}um2 "
                f"circ>={cfg.fat.circularity_min} sol>={cfg.fat.solidity_min}"
            )
            sheet = label_panel(sheet, header, height=34)
            sheet_path = save_rgb(qc_dir / "contact_sheet.png", sheet)
            summary["contact_sheet"] = str(sheet_path)

        cfg.save(out_dir / "config_used.yaml")
        _print_summary(
            summary, all_macro, all_micro, cfg,
            sampled_reports if detail else [], csv_path,
        )
        return summary
    finally:
        slide.close()


# OpenSlide-readable whole-slide formats.
SLIDE_EXTENSIONS = (
    ".svs", ".tif", ".tiff", ".ndpi", ".vms", ".vmu",
    ".scn", ".mrxs", ".svslide", ".bif",
)


def find_slides(folder: str | Path, pattern: str = "*") -> list[Path]:
    """Whole-slide files in `folder`, sorted naturally by name."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    found = [
        p for p in sorted(folder.glob(pattern))
        if p.is_file() and p.suffix.lower() in SLIDE_EXTENSIONS
        and not p.name.startswith(".")
    ]

    def key(p: Path):
        # Digit-aware sort so R25-264-2 precedes R25-264-10.
        import re
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r"(\d+)", p.stem)]

    return sorted(found, key=key)


def _batch_worker(args) -> dict[str, Any]:
    """One slide, in its own process, with its console output sent to a log.

    Slides share nothing -- each writes its own output directory -- so the only
    reason this is not embarrassingly parallel is the per-slide chatter, which
    goes to `outputs/<slide>/run.log` instead of interleaving on the terminal.
    """
    import contextlib

    cfg_dict, path = args
    slide_cfg = Config.from_dict(cfg_dict)
    slide_cfg.slide_path = path
    stem = Path(path).stem
    log_path = Path(slide_cfg.output_dir) / stem / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "w") as log, contextlib.redirect_stdout(log):
            return run_slide(slide_cfg, verbose_tiles=0, detail=False)
    except Exception as exc:
        return {"slide": stem, "error": f"{type(exc).__name__}: {exc}"}


def run_batch(
    cfg: Config,
    folder: str | Path,
    pattern: str = "*",
    verbose_tiles: int = 0,
    workers: int = 1,
) -> pd.DataFrame:
    """Run the full pipeline over every slide in a folder.

    Each slide is processed independently and its own outputs are written as
    usual; a failure on one slide is recorded and does not stop the rest.
    """
    slides = find_slides(folder, pattern)
    if not slides:
        raise RuntimeError(
            f"no whole-slide images in {folder} "
            f"(looked for {', '.join(SLIDE_EXTENSIONS)})"
        )

    print(f"batch: {len(slides)} slide(s) in {folder}")
    for p in slides:
        print(f"  - {p.name}  ({p.stat().st_size/1e6:.0f} MB)")
    print()

    summaries: list[dict[str, Any]] = []
    t_start = time.time()

    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        print(f"running {workers} slides at a time; "
              f"per-slide output -> {cfg.output_dir}/<slide>/run.log\n")
        jobs = [(cfg.to_dict(), str(p)) for p in slides]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_batch_worker, j): j[1] for j in jobs}
            for i, fut in enumerate(as_completed(futures), 1):
                s = fut.result()
                summaries.append(s)
                if "error" in s:
                    print(f"  [{i}/{len(slides)}] {s['slide']}  FAILED: {s['error']}",
                          flush=True)
                else:
                    print(f"  [{i}/{len(slides)}] {s['slide']:<20} "
                          f"macro {s['macro_fat_fraction']*100:6.2f}%  "
                          f"{s['tiles_processed']:>5} tiles  "
                          f"{s['seconds']:.0f}s", flush=True)
        # as_completed returns in finish order; restore the folder's order.
        order = {p.stem: i for i, p in enumerate(slides)}
        summaries.sort(key=lambda s: order.get(s["slide"], 1 << 30))
    else:
        for i, path in enumerate(slides, 1):
            print("=" * 84)
            print(f"[{i}/{len(slides)}] {path.name}")
            print("=" * 84)
            slide_cfg = Config.from_dict(cfg.to_dict())
            slide_cfg.slide_path = str(path)
            try:
                summaries.append(run_slide(slide_cfg, verbose_tiles, detail=False))
            except Exception as exc:  # keep going; report at the end
                print(f"  FAILED: {type(exc).__name__}: {exc}")
                summaries.append(
                    {"slide": path.stem, "error": f"{type(exc).__name__}: {exc}"}
                )
            print()

    df = pd.DataFrame(summaries)
    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    batch_csv = out_root / "batch_summary.csv"
    df.to_csv(batch_csv, index=False)

    _print_batch_table(df, time.time() - t_start, batch_csv)
    return df


def _print_batch_table(df: pd.DataFrame, elapsed: float, csv_path: Path) -> None:
    print("=" * 100)
    print("BATCH SUMMARY")
    print("=" * 100)
    ok = df[df.get("error").isna()] if "error" in df else df
    failed = df[df.get("error").notna()] if "error" in df else df.iloc[0:0]

    if not ok.empty:
        print(
            f"{'slide':<16}{'tiles':>7}{'tissue mm2':>12}{'macro n':>10}"
            f"{'macro %':>9}{'micro %':>9}{'total %':>9}{'border %':>10}{'sec':>7}"
        )
        print("-" * 100)
        for _, r in ok.iterrows():
            print(
                f"{str(r['slide']):<16}{int(r['tiles_processed']):>7}"
                f"{r['slide_tissue_area_mm2']:>12.2f}{int(r['macro_count']):>10}"
                f"{r['macro_fat_fraction']*100:>9.2f}{r['micro_fat_fraction']*100:>9.2f}"
                f"{r['total_fat_fraction']*100:>9.2f}"
                f"{r['border_fraction_of_fat']*100:>10.2f}{r['seconds']:>7.0f}"
            )
        print("-" * 100)
        tot_tissue = ok["processed_tissue_area_mm2"].sum()
        w_macro = (ok["macro_area_mm2"].sum() / tot_tissue * 100) if tot_tissue else 0
        w_micro = (ok["micro_area_mm2"].sum() / tot_tissue * 100) if tot_tissue else 0
        print(
            f"{'COHORT':<16}{int(ok['tiles_processed'].sum()):>7}"
            f"{ok['slide_tissue_area_mm2'].sum():>12.2f}"
            f"{int(ok['macro_count'].sum()):>10}{w_macro:>9.2f}{w_micro:>9.2f}"
            f"{w_macro + w_micro:>9.2f}{'':>10}{elapsed:>7.0f}"
        )
        print("\n  (cohort row is tissue-area-weighted, not a mean of slide values)")
        print(
            f"  macro fat fraction across slides: "
            f"min={ok['macro_fat_fraction'].min()*100:.2f}%  "
            f"max={ok['macro_fat_fraction'].max()*100:.2f}%  "
            f"spread={((ok['macro_fat_fraction'].max()-ok['macro_fat_fraction'].min()))*100:.2f} pts"
        )

    if not failed.empty:
        print(f"\n  {len(failed)} slide(s) FAILED:")
        for _, r in failed.iterrows():
            print(f"    {r['slide']}: {r['error']}")

    print(f"\n  batch summary CSV -> {csv_path}")


def _print_summary(
    s: dict[str, Any],
    macro: np.ndarray,
    micro: np.ndarray,
    cfg: Config,
    sampled_reports: list[str],
    csv_path: Path,
) -> None:
    print()
    print("=" * 84)
    print(f"SLIDE SUMMARY -- {s['slide']}")
    print("=" * 84)
    print(f"  tiles processed:        {s['tiles_processed']} of {s['tiles_candidate']} candidates")
    print(f"  slide tissue area:      {s['slide_tissue_area_mm2']:.2f} mm^2")
    print(f"  tissue in tiles:        {s['processed_tissue_area_mm2']:.2f} mm^2")
    print()
    print(f"  {'':<12}{'count':>10}{'area mm^2':>12}{'fat fraction':>15}")
    print(f"  {'macro':<12}{s['macro_count']:>10}{s['macro_area_mm2']:>12.4f}"
          f"{s['macro_fat_fraction']*100:>14.2f}%")
    print(f"  {'micro':<12}{s['micro_count']:>10}{s['micro_area_mm2']:>12.4f}"
          f"{s['micro_fat_fraction']*100:>14.2f}%")
    print(f"  {'TOTAL':<12}{s['macro_count']+s['micro_count']:>10}"
          f"{s['macro_area_mm2']+s['micro_area_mm2']:>12.4f}"
          f"{s['total_fat_fraction']*100:>14.2f}%")
    print()
    print(f"  microvesicular in exported labels: "
          f"{'YES' if cfg.fat.include_microvesicular else 'NO (measured, not labeled)'}")
    print()
    print("  border-touching droplets (exclude_border tradeoff):")
    print(f"    count: {s['border_count']}, area: {s['border_area_um2']/1e6:.4f} mm^2 "
          f"= {s['border_fraction_of_fat']*100:.2f}% of detected fat area")
    print(f"    exclude_border is currently {'ON' if cfg.fat.exclude_border else 'OFF'}"
          f" -> that area is {'removed' if cfg.fat.exclude_border else 'kept'}")
    print()
    print(_histogram(macro, cfg.qc.histogram_bins, "MACRO droplet size distribution"))
    print()
    print(_histogram(micro, cfg.qc.histogram_bins, "MICRO droplet size distribution"))
    if sampled_reports:
        print()
        print("-" * 84)
        print("PER-TILE DETAIL (contact-sheet tiles)")
        print("-" * 84)
        for r in sampled_reports:
            print(r)
    print()
    print(f"  per-tile CSV -> {csv_path}")
    if "contact_sheet" in s:
        print(f"  contact sheet -> {s['contact_sheet']}")
    print(f"  elapsed: {s['seconds']:.1f}s")
