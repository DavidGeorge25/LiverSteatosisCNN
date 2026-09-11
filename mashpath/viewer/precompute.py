"""One exhaustive pass over a slide at native resolution. The number of record.

WHY EXHAUSTIVE RATHER THAN SAMPLED. Every slide-level figure this project has
produced so far comes from a random sample -- 150 tiles per slide in
`dataset_v2`, 60 in the survey, roughly 6% of a section. Measured on
R25-264-1, a 200-tile sample carries a 95% interval of +/-0.74 pp around a
14.35% slide. That is invisible in a table and fatal in a viewer: the whole
argument for this project is a continuous, objective number, and a continuous
number quoted to two decimals off a 6% sample is the precise thing a reviewer
attacks. Full coverage removes the sampling term entirely -- the result stops
being an estimate and becomes a census, and two decimals become defensible.

It is also affordable. 2,487 tiles for the fattiest slide in the cohort, at
~84 ms/tile on a laptop CPU, is 3.5 minutes; the whole 260-slide cohort is
~9 CPU-hours, which is under an hour of wall clock as the `--array=1-N%15`
pattern already in `cluster/`.

WHY NATIVE RESOLUTION ONLY. The model has seen one scale and only one: every
slide in `dataset_v2` is 0.4953 um/px, and the augmentation is colour-only.
Running it on a downsampled pyramid level reports 0.06x the fat. So this pass
runs at the training scale, and the viewer downsamples the RESULT. If a slide
arrives at a different mpp -- the first one from another scanner will -- it is
resampled to the training scale here, by mpp, never by pyramid level.

WHAT THIS DOES NOT ESTABLISH. Full coverage makes the number precise, not
correct. There is still no reference segmentation (`LIMITATIONS.md` §1) and the
40-tile package at `outputs/annotation_set_v1/` is still the gate. This writes
`validated_against_reference: false` into every meta.json it produces, because
a beautiful overlay is exactly the thing that tempts a reader to assume
otherwise.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..config import Config
from ..core.slide import Slide
from ..core.tiling import iter_tiles
from ..core.tissue import detect_tissue

# The scale the model was trained at. Not a guess: every one of the 259 slides
# in dataset_v2 carries mpp 0.4953, and `UNetConfig` has no scale augmentation.
TRAINING_MPP = 0.4953

# Probabilities in this band are the ones the model is genuinely unsure about.
# Reported as a certainty figure; never used to change the mask, which is
# thresholded at 0.5 exactly as `predict_frame` does it.
AMBIGUOUS_LO, AMBIGUOUS_HI = 0.35, 0.65

# Droplet size bands, in um^2, taken from `FatConfig` so the viewer's
# distribution is comparable with the teacher's macro_pct / micro_pct rather
# than being a second, differently-defined thing.
MICRO_BAND = (3.0, 20.0)
MACRO_BAND = (20.0, 2000.0)

# The U-Net pools 4 times (`UNetConfig.depth`), so an input whose height or
# width is not a multiple of 16 loses a row to integer division on the way
# down and cannot be concatenated with its skip connection on the way up:
#
#   ConcatOp : Dimension 1 in both shapes must be equal:
#   shape[0] = [1,136,202,256] vs. shape[1] = [1,137,202,256]
#
# The census never hits this -- it always feeds exactly 512 px -- but any
# caller passing a free-form region does, which is why it surfaced from the
# viewer's live spot check rather than from the pipeline.
POOL_MULTIPLE = 16


def pad_for_unet(img: np.ndarray, multiple: int = POOL_MULTIPLE
                 ) -> tuple[np.ndarray, int, int]:
    """Pad an image up to a multiple of `multiple`. Returns (padded, h, w).

    Reflected rather than zero- or white-padded: a hard edge invents a
    boundary the model has never seen, and it would be a boundary right where
    the user is looking. The pad is cropped off the prediction afterwards, so
    it only ever affects the convolution's view near the edge.
    """
    h, w = img.shape[:2]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    return img, h, w


@dataclass
class PrecomputeResult:
    slide: str
    fat_fraction: float
    tissue_px: int
    fat_px: int
    tiles: int
    seconds: float
    meta: dict = field(default_factory=dict)


def _load_model(checkpoint: str | Path):
    import os
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import keras
    return keras.models.load_model(str(checkpoint), compile=False)


def _fold_provenance(checkpoint: str | Path) -> dict:
    """Everything about the model that changes what the number means.

    `stain_normalize` in particular: the viewer must preprocess exactly as
    training did, and the answer here is `none` for every fold. Recording it
    per-artifact means a future normalized fold cannot be served through this
    viewer without the mismatch being visible in the file.
    """
    cp = Path(checkpoint)
    out = {"checkpoint": str(cp), "fold": cp.parent.name}
    for name in ("config.json", "summary.json"):
        p = cp.parent / name
        if p.exists():
            try:
                out[name[:-5]] = json.loads(p.read_text())
            except Exception:
                pass
    cfg = out.get("config", {})
    out["stain_normalize"] = cfg.get("stain_normalize", "unknown")
    out["held_out_batch"] = cfg.get("held_out_batch",
                                    out.get("summary", {}).get("held_out_batch"))
    return out


def _droplets(mask: np.ndarray, um2_per_px: float) -> tuple[int, int, int, list]:
    """Connected components of one tile's mask, binned by physical area.

    Per tile, so a droplet straddling a tile border is counted twice. That is
    stated rather than corrected: the headline number is AREA, which is
    measured in pixels and is unaffected, and correcting the count would mean
    holding a 1.4 Gpx mask in memory to gain a figure the spec asks for only
    "if it comes cheap".
    """
    if not mask.any():
        return 0, 0, 0, []
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    micro = macro = other = 0
    areas: list[float] = []
    for i in range(1, n):
        a = float(stats[i, cv2.CC_STAT_AREA]) * um2_per_px
        areas.append(a)
        if MICRO_BAND[0] <= a < MICRO_BAND[1]:
            micro += 1
        elif MACRO_BAND[0] <= a <= MACRO_BAND[1]:
            macro += 1
        else:
            other += 1
    return micro, macro, other, areas


def precompute_slide(
    slide_path: str | Path,
    checkpoint: str | Path,
    out_dir: str | Path,
    cfg: Config | None = None,
    batch_size: int = 8,
    threshold: float = 0.5,
    limit: int = 0,
    grid: int = 48,
    progress_every: int = 100,
    verbose: bool = True,
    record_path: str | Path | None = None,
) -> PrecomputeResult:
    """Run the model over every tissue tile and write the viewer's artifact.

    `record_path` is the location written into meta.json, when that differs
    from where the pixels were actually read. On a cluster the slide is staged
    to `$SLURM_TMPDIR` for the 82x read speedup, and that directory is deleted
    when the job ends -- so recording it would leave every artifact pointing at
    a path that no longer exists, and the viewer would fail to open slides it
    had just successfully measured.
    """
    cfg = cfg or Config()
    slide_path = Path(slide_path)
    recorded = Path(record_path) if record_path else slide_path
    out_dir = Path(out_dir)
    (out_dir / "mask" / "l0").mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    sl = Slide(str(slide_path), cfg.slide)
    tissue = detect_tissue(sl, cfg.tissue)

    W, H = sl.dimensions
    mpp = float(sl.mpp_x)
    # Resample by mpp, never by pyramid level. At this cohort's 0.4953 the
    # ratio is 1.0 and nothing happens; the branch exists so the first slide
    # from another scanner is rescaled to the scale the model knows instead of
    # being measured at whatever its native pixels happen to mean.
    ratio = mpp / TRAINING_MPP
    model_tile = cfg.tiling.tile_size
    native_tile = int(round(model_tile * ratio))
    rescaled = abs(ratio - 1.0) > 0.02

    # The TILE GRID has to move with the resampling, not just the read size.
    # Enumerating on the 512 px grid while reading `native_tile` pixels would
    # leave gaps on a finer scanner (read 259, step 512) and double-count on a
    # coarser one (read 1034, step 512) -- and since the numerator and the
    # denominator are both measured per tile, the resulting percentage stays
    # plausible while covering the wrong fraction of the section.
    from dataclasses import replace
    tiling = replace(cfg.tiling, tile_size=native_tile, stride=native_tile)
    tiles = list(iter_tiles(sl, tissue, tiling, limit=limit))
    if not tiles:
        raise RuntimeError(f"{slide_path.name}: no tissue tiles found")

    if verbose:
        print(f"{slide_path.name}: {W}x{H}  mpp {mpp:.4f}  "
              f"{len(tiles)} tissue tiles  tissue {tissue.area_mm2:.1f} mm2",
              flush=True)
        if rescaled:
            print(f"  resampling {mpp:.4f} -> {TRAINING_MPP} um/px "
                  f"(reading {native_tile} px, model sees {model_tile} px)",
                  flush=True)

    model = _load_model(checkpoint)
    um2_per_px = mpp * mpp

    rows: list[dict] = []
    all_areas: list[float] = []
    total_fat = total_tissue = 0
    total_amb = 0
    prob_sum = 0.0
    ds4 = np.zeros((max(1, -(-H // 4)), max(1, -(-W // 4))), np.uint8)

    def flush(batch: list) -> None:
        nonlocal total_fat, total_tissue, total_amb, prob_sum
        if not batch:
            return
        x = np.stack([b[1] for b in batch]).astype(np.float32) / 255.0
        pb = model.predict_on_batch(x)
        for (tl, _rgb, tmask), pred in zip(batch, pb):
            p = np.asarray(pred[..., 0], dtype=np.float32)
            if rescaled:
                p = cv2.resize(p, (native_tile, native_tile),
                               interpolation=cv2.INTER_LINEAR)
            # Confined to tissue, because the teacher's labels were: the
            # detector does `white &= tissue_mask` before measuring, and glass
            # is white, so a prediction scored on the whole tile is scored on a
            # larger canvas than the one that produced the labels.
            fat = (p >= threshold) & tmask
            fat_px = int(fat.sum())
            tis_px = int(tmask.sum())
            amb = int(((p >= AMBIGUOUS_LO) & (p <= AMBIGUOUS_HI) & tmask).sum())
            mean_p = float(p[fat].mean()) if fat_px else 0.0

            micro, macro, other, areas = _droplets(fat, um2_per_px)
            all_areas.extend(areas)
            total_fat += fat_px
            total_tissue += tis_px
            total_amb += amb
            prob_sum += mean_p * fat_px

            if fat_px:
                m8 = (fat.astype(np.uint8) * 255)
                cv2.imwrite(str(out_dir / "mask" / "l0" /
                                f"{tl.x:06d}_{tl.y:06d}.png"), m8)
                small = cv2.resize(m8, (max(1, m8.shape[1] // 4),
                                        max(1, m8.shape[0] // 4)),
                                   interpolation=cv2.INTER_AREA)
                yy, xx = tl.y // 4, tl.x // 4
                sub = ds4[yy:yy + small.shape[0], xx:xx + small.shape[1]]
                if sub.size:
                    np.maximum(sub, small[:sub.shape[0], :sub.shape[1]], out=sub)

            rows.append({
                "tile_x": tl.x, "tile_y": tl.y, "tile_size": tl.size,
                "tissue_fraction": round(float(tl.tissue_fraction), 4),
                "tissue_px": tis_px, "fat_px": fat_px,
                "fat_fraction": (fat_px / tis_px) if tis_px else 0.0,
                "mean_prob": round(mean_p, 4),
                "ambiguous_px": amb,
                "micro_n": micro, "macro_n": macro, "other_n": other,
            })

    batch: list = []
    for i, tl in enumerate(tiles):
        rgb = sl.read_region((tl.x, tl.y), 0, (native_tile, native_tile))
        tmask = tissue.tile_mask(tl.x, tl.y, native_tile, native_tile)
        model_in = cv2.resize(rgb, (model_tile, model_tile),
                              interpolation=cv2.INTER_AREA) if rescaled else rgb
        batch.append((tl, model_in, tmask))
        if len(batch) >= batch_size:
            flush(batch)
            batch = []
        if verbose and progress_every and (i + 1) % progress_every == 0:
            el = time.time() - t_start
            print(f"  {i + 1}/{len(tiles)} tiles  {el:.0f}s  "
                  f"({el / (i + 1) * 1000:.0f} ms/tile)  "
                  f"running {100 * total_fat / max(total_tissue, 1):.2f}%",
                  flush=True)
    flush(batch)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tiles.csv", index=False)
    cv2.imwrite(str(out_dir / "mask" / "ds4.png"), ds4)

    fat_fraction = total_fat / total_tissue if total_tissue else 0.0
    areas = np.asarray(all_areas, dtype=np.float64)
    seconds = time.time() - t_start

    meta = {
        "slide": recorded.stem,
        "slide_path": str(recorded),
        "read_from": str(slide_path),
        "width": W, "height": H, "mpp": mpp,
        "resampled_to_training_mpp": bool(rescaled),
        "training_mpp": TRAINING_MPP,
        "mask_tile": native_tile,
        "tile_size": cfg.tiling.tile_size,
        "threshold": threshold,
        # THE number. Exact, not an estimate: every tissue tile was measured.
        "fat_fraction": fat_fraction,
        "fat_percent": 100.0 * fat_fraction,
        "coverage": "complete" if not limit else f"partial ({len(tiles)} tiles)",
        "tiles": len(tiles),
        "fat_px": total_fat,
        "tissue_px": total_tissue,
        "tissue_mm2": tissue.area_mm2,
        "tissue_threshold": float(tissue.threshold),
        "droplets": {
            "counted_per_tile": True,
            "note": ("connected components are counted within each 512 px tile, "
                     "so a droplet crossing a tile border is counted twice; "
                     "fat AREA is measured per pixel and is unaffected"),
            "total": int(len(areas)),
            "micro_n": int(df["micro_n"].sum()) if len(df) else 0,
            "macro_n": int(df["macro_n"].sum()) if len(df) else 0,
            "other_n": int(df["other_n"].sum()) if len(df) else 0,
            "micro_band_um2": list(MICRO_BAND),
            "macro_band_um2": list(MACRO_BAND),
            "area_um2": {
                "mean": float(areas.mean()) if areas.size else 0.0,
                "median": float(np.median(areas)) if areas.size else 0.0,
                "p90": float(np.percentile(areas, 90)) if areas.size else 0.0,
                "max": float(areas.max()) if areas.size else 0.0,
            },
            "histogram": _area_histogram(areas),
        },
        "certainty": {
            "mean_probability_of_called_fat": (prob_sum / total_fat) if total_fat else 0.0,
            "ambiguous_px": total_amb,
            "ambiguous_fraction_of_tissue": (total_amb / total_tissue) if total_tissue else 0.0,
            "band": [AMBIGUOUS_LO, AMBIGUOUS_HI],
            "note": ("the model's own sigmoid. NOT the nine-way pseudo-label vote "
                     "-- that exists only for the exported training tiles and "
                     "cannot be computed for a slide the ensemble never scored"),
        },
        "distribution": _zonal(df, W, H, grid),
        "model": _fold_provenance(checkpoint),
        "validated_against_reference": False,
        "validation_note": ("no reference segmentation exists; see LIMITATIONS.md "
                            "§1. This overlay is illustration, not accuracy "
                            "evidence."),
        "seconds": seconds,
        "ms_per_tile": 1000.0 * seconds / max(len(tiles), 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=float))

    if verbose:
        print(f"  DONE {slide_path.name}: {100 * fat_fraction:.2f}% fat of tissue, "
              f"{len(tiles)} tiles, {seconds:.0f}s "
              f"({1000 * seconds / len(tiles):.0f} ms/tile)", flush=True)
    sl.close()
    return PrecomputeResult(slide_path.stem, fat_fraction, total_tissue,
                            total_fat, len(tiles), seconds, meta)


def _area_histogram(areas: np.ndarray, bins: int = 24) -> dict:
    """Log-spaced, because droplet area spans three orders of magnitude.

    Linear bins put 95% of droplets in the first bucket and tell the viewer
    nothing about the macro/micro balance, which is the part that is
    biologically readable.
    """
    if areas.size == 0:
        return {"edges": [], "counts": []}
    lo = max(1.0, float(areas.min()))
    hi = max(lo * 2, float(np.percentile(areas, 99.5)))
    edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)
    counts, _ = np.histogram(areas, bins=edges)
    return {"edges": [float(e) for e in edges],
            "counts": [int(c) for c in counts]}


def _zonal(df: pd.DataFrame, width: int, height: int, grid: int) -> dict:
    """Fat as a function of position, so "10%" can be read as uniform or zonal.

    A slide averaging 10% can be evenly infiltrated or have a fatty
    periportal band and a clean centre, and the difference is real biology
    that a single number erases. This is the cheapest honest way to show it:
    the per-tile fractions the census already produced, binned onto a coarse
    grid, plus the histogram of the tile values themselves.
    """
    if df.empty:
        return {"grid": [], "cell_px": 0, "tile_histogram": {"edges": [], "counts": []}}
    cell = max(width, height) / grid
    cols = max(1, int(np.ceil(width / cell)))
    rows_n = max(1, int(np.ceil(height / cell)))
    fat = np.zeros((rows_n, cols))
    tis = np.zeros((rows_n, cols))
    ci = np.clip((df["tile_x"].to_numpy() / cell).astype(int), 0, cols - 1)
    ri = np.clip((df["tile_y"].to_numpy() / cell).astype(int), 0, rows_n - 1)
    np.add.at(fat, (ri, ci), df["fat_px"].to_numpy())
    np.add.at(tis, (ri, ci), df["tissue_px"].to_numpy())
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(tis > 0, 100.0 * fat / np.maximum(tis, 1), np.nan)
    vals = df["fat_fraction"].to_numpy() * 100.0
    hi = float(np.percentile(vals, 99.5)) if vals.size else 1.0
    counts, edges = np.histogram(vals, bins=30, range=(0.0, max(hi, 0.5)))
    return {
        "grid": [[None if np.isnan(v) else round(float(v), 3) for v in row]
                 for row in pct],
        "cell_px": float(cell),
        "cols": cols, "rows": rows_n,
        "tile_histogram": {"edges": [float(e) for e in edges],
                           "counts": [int(c) for c in counts]},
        "tile_fat_percent": {
            "mean": float(vals.mean()) if vals.size else 0.0,
            "median": float(np.median(vals)) if vals.size else 0.0,
            "p10": float(np.percentile(vals, 10)) if vals.size else 0.0,
            "p90": float(np.percentile(vals, 90)) if vals.size else 0.0,
            "max": float(vals.max()) if vals.size else 0.0,
        },
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("slide")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, help="artifact directory for this slide")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=0,
                   help="stop after N tiles -- for smoke tests ONLY; the "
                        "artifact is then marked partial and its percentage "
                        "is a sample, not a census")
    p.add_argument("--config", action="append", default=[])
    p.add_argument("--record-path", default=None,
                   help="the slide's DURABLE location, when the pixels are "
                        "being read from a staged copy. Pass the original "
                        "path here or the artifact will point at a "
                        "$SLURM_TMPDIR that no longer exists.")
    a = p.parse_args(argv)

    cfg = Config()
    for c in a.config:
        cfg = cfg.merged_with_file(c) if hasattr(cfg, "merged_with_file") else cfg
    r = precompute_slide(a.slide, a.checkpoint, a.out, cfg,
                         batch_size=a.batch_size, threshold=a.threshold,
                         limit=a.limit, record_path=a.record_path)
    print(json.dumps({"slide": r.slide, "fat_percent": 100 * r.fat_fraction,
                      "tiles": r.tiles, "seconds": r.seconds}, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
