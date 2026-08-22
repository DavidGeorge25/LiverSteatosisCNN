"""Export a stratified tile set for manual annotation.

The negative control proves the pipeline does not hallucinate fat on fat-free
tissue, but it cannot say whether the fat it finds on steatotic tissue is
measured *correctly* -- every parameter cell in the sweep separates the cohorts
perfectly while reporting fat fractions that differ by 1.6x. Choosing among
them, and reporting Dice/IoU or per-droplet precision and recall, requires
ground truth.

This module builds the annotation package: a reproducible, stratified sample of
tiles, each exported as the raw image, the current pseudo-label as an editable
starting mask, and a reference overlay.

Stratification is by predicted fat fraction, not uniform random. A uniform
sample is dominated by typical tiles and would leave the extremes -- where the
parameter cells actually disagree -- barely represented.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ...config import MashConfig as Config
from .detect import MACRO, MICRO, detect_fat
from ...core.slide import Slide
from ...core.tiling import Tile, iter_tiles
from ...core.tissue import detect_tissue
from ...core.viz import outline_mask, save_rgb


@dataclass
class Candidate:
    slide: str
    path: str
    tile: Tile
    macro_pct: float
    micro_pct: float
    macro_n: int


def scan_candidates(
    slide_path: str | Path, cfg: Config, n_tiles: int, seed: int
) -> list[Candidate]:
    """Score a random subset of a slide's tiles so they can be stratified."""
    slide = Slide(str(slide_path), cfg.slide)
    try:
        tissue = detect_tissue(slide, cfg.tissue)
        all_tiles = list(iter_tiles(slide, tissue, cfg.tiling))
        if not all_tiles:
            return []
        rng = np.random.default_rng(seed)
        idx = rng.choice(
            len(all_tiles), size=min(n_tiles, len(all_tiles)), replace=False
        )
        um2 = slide.um2_per_pixel(cfg.tiling.level)

        out = []
        for i in idx:
            t = all_tiles[int(i)]
            rgb = t.read(slide, cfg.tiling.level)
            tmask = tissue.tile_mask(t.x, t.y, t.size, t.size)
            res = detect_fat(rgb, tmask, cfg.fat, um2)
            out.append(
                Candidate(
                    slide=Path(slide_path).stem,
                    path=str(slide_path),
                    tile=t,
                    macro_pct=res.fat_fraction(MACRO) * 100,
                    micro_pct=res.fat_fraction(MICRO) * 100,
                    macro_n=res.count(MACRO),
                )
            )
        return out
    finally:
        slide.close()


def stratify(
    candidates: Sequence[Candidate], n_out: int, n_strata: int, seed: int
) -> list[Candidate]:
    """Pick `n_out` candidates spread evenly across fat-fraction strata.

    Strata are quantile-based, so they adapt to the cohort rather than assuming
    a fixed range. Within a stratum the choice is random with a fixed seed.
    """
    if not candidates:
        return []
    n_out = min(n_out, len(candidates))
    vals = np.array([c.macro_pct for c in candidates])
    edges = np.quantile(vals, np.linspace(0, 1, n_strata + 1))
    edges[-1] += 1e-9
    rng = np.random.default_rng(seed)

    picked: list[Candidate] = []
    per = max(1, n_out // n_strata)
    for lo, hi in zip(edges[:-1], edges[1:]):
        pool = [c for c in candidates if lo <= c.macro_pct < hi]
        if not pool:
            continue
        take = min(per, len(pool))
        sel = rng.choice(len(pool), size=take, replace=False)
        picked.extend(pool[int(i)] for i in sel)

    # Top up to n_out from whatever is left, so rounding never short-changes it.
    if len(picked) < n_out:
        chosen = {(c.slide, c.tile.x, c.tile.y) for c in picked}
        rest = [c for c in candidates if (c.slide, c.tile.x, c.tile.y) not in chosen]
        if rest:
            extra = rng.choice(
                len(rest), size=min(n_out - len(picked), len(rest)), replace=False
            )
            picked.extend(rest[int(i)] for i in extra)
    return picked[:n_out]


def export_package(
    picked: Sequence[Candidate], cfg: Config, out_dir: str | Path
) -> pd.DataFrame:
    """Write images, starting masks, overlays and a manifest.

    The starting mask is the current pseudo-label. Annotators correct it rather
    than drawing from scratch, which is far faster -- but it does bias them
    toward the pipeline's answer, so the manifest records that these are
    pre-seeded and the raw image is exported alongside for independent work.
    """
    out_dir = Path(out_dir)
    for sub in ("images", "masks_prefilled", "overlays"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    rows = []
    by_slide: dict[str, list[Candidate]] = {}
    for c in picked:
        by_slide.setdefault(c.path, []).append(c)

    for path, cands in by_slide.items():
        slide = Slide(path, cfg.slide)
        try:
            tissue = detect_tissue(slide, cfg.tissue)
            um2 = slide.um2_per_pixel(cfg.tiling.level)
            for c in cands:
                t = c.tile
                rgb = t.read(slide, cfg.tiling.level)
                tmask = tissue.tile_mask(t.x, t.y, t.size, t.size)
                res = detect_fat(rgb, tmask, cfg.fat, um2)
                stem = f"{c.slide}__{t.name}"

                save_rgb(out_dir / "images" / f"{stem}.png", rgb)
                mask = (res.label_mask.astype(np.uint8) * 255)
                save_rgb(
                    out_dir / "masks_prefilled" / f"{stem}_mask.png",
                    np.dstack([mask, mask, mask]),
                )
                save_rgb(
                    out_dir / "overlays" / f"{stem}_overlay.png",
                    outline_mask(rgb, res.macro_mask, (255, 0, 0), 2),
                )
                rows.append({
                    "annotation_id": stem,
                    "slide": c.slide,
                    "tile_x": t.x,
                    "tile_y": t.y,
                    "tile_size": t.size,
                    "level": cfg.tiling.level,
                    "mpp": slide.mpp_x,
                    "tissue_fraction": round(t.tissue_fraction, 4),
                    "predicted_macro_pct": round(c.macro_pct, 3),
                    "predicted_micro_pct": round(c.micro_pct, 3),
                    "predicted_macro_n": c.macro_n,
                    "mask_is_prefilled": True,
                    "annotator": "",
                    "status": "todo",
                })
        finally:
            slide.close()

    df = pd.DataFrame(rows).sort_values(["slide", "tile_y", "tile_x"])
    df.to_csv(out_dir / "manifest.csv", index=False)
    cfg.save(out_dir / "config_used.yaml")
    _write_instructions(out_dir, len(df))
    return df


def _write_instructions(out_dir: Path, n: int) -> None:
    (out_dir / "README.md").write_text(
        f"""# Annotation package -- {n} tiles

## What to do

Each tile in `images/` needs a binary mask marking **macrovesicular fat
droplet pixels**. `masks_prefilled/` holds the pipeline's current guess as a
starting point -- correct it rather than drawing from scratch. `overlays/`
shows that guess outlined on the tile for quick reference.

Save corrected masks to `masks_corrected/` keeping the same filename.
White (255) = fat, black (0) = not fat.

## What counts as fat

Include: round-to-oval clear vacuoles that displace the hepatocyte nucleus,
the classic macrovesicular droplet.

Exclude:
- vessel and sinusoidal lumens (elongated, endothelial lining, may contain
  red blood cells)
- tissue tears and processing clefts (irregular, angular)
- the gap at the tissue edge
- pale/rarefied cytoplasm without a discrete droplet border -- this is the
  pipeline's main known false positive and the single most valuable thing to
  correct

Microvesicular speckling: **leave it out** of the mask. It is measured
separately and is not part of the current label definition. If a tile is
predominantly microvesicular, mark `status = microvesicular` in the manifest
rather than annotating it.

## Uncertain tiles

Set `status = unclear` in `manifest.csv` rather than guessing. Tiles nobody
can call confidently are themselves a result worth reporting.

## Why these tiles

Sampled stratified by predicted fat fraction, not uniformly at random, so the
low and high extremes are represented. That is deliberate: the parameter
settings under evaluation disagree most at the extremes and agree in the
middle, so a uniform sample would carry little information about which is
right. It also means **this set is not a random sample of the slide** -- fat
fraction computed over it will not match the slide-level fat fraction, and it
should be used for per-tile agreement metrics only.
""",
        encoding="utf-8",
    )
