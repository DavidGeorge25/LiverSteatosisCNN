"""Nucleus segmentation with StarDist, and per-nucleus morphometry.

One tile at a time, level 0, never the whole slide.

The work splits in two so that both callers get the same measurements:
`segment_array` takes pixels and knows nothing about slides, which is the shape
`review.scan` hands a detector; `segment_tile` adds the slide read and the
margin handling, which is what the QC pass wants.

Two details matter for honest numbers out of a tiled run:

*Margins.* `segment_tile` reads each tile with `margin_px` of extra context on
every side, segments the larger image, and keeps only nuclei whose centroid
falls inside the core tile. A nucleus straddling a boundary is therefore
measured on its full extent, not on the clipped fragment, and is counted once
across the slide. Without this, every boundary nucleus is split in two and both
halves land in the small-and-dark bin -- exactly the bin the inflammation call
depends on.

*Normalization.* StarDist wants percentile-normalized input. The percentiles
come from the core tile only, so white padding at a slide edge cannot drag the
range around.

Intensity is reported two ways: mean grayscale (0-255, lower = darker) and mean
haematoxylin optical density from colour deconvolution (higher = more
chromatin). Grayscale is easier to reason about; haematoxylin is less
contaminated by surrounding eosin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.color import rgb2hed
from skimage.measure import regionprops

from ...core.segmentation import (
    load_model,
    normalize_percentile,
    predict_instances,
    read_padded,
    relabel_core,
    upscale_image,
)
from ...core.segmentation import scale_note as core_scale_note
from ...core.slide import Slide
from ...core.tissue import TissueMask
from .config import NucleiConfig

# Column order of the measurement array built below.
_COLS = (
    "x", "y", "area_um2", "circularity", "eccentricity",
    "solidity", "gray", "hematoxylin", "prob",
)


@dataclass
class TileNuclei:
    """Per-nucleus measurements for one tile, in level-0 slide coordinates."""

    x: np.ndarray  # centroid x, level-0 px
    y: np.ndarray  # centroid y, level-0 px
    area_um2: np.ndarray
    circularity: np.ndarray  # 4*pi*A / P^2, Crofton perimeter
    eccentricity: np.ndarray  # 0 = circle, 1 = line
    solidity: np.ndarray
    gray: np.ndarray  # mean grayscale, 0-255
    hematoxylin: np.ndarray  # mean haematoxylin OD
    prob: np.ndarray  # StarDist objectness
    labels: np.ndarray  # core-tile label image, renumbered 1..N to match
    rejected: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.x.size)

    @classmethod
    def empty(cls, shape: tuple[int, int]) -> "TileNuclei":
        z = np.zeros(0, dtype=np.float64)
        return cls(
            x=z, y=z.copy(), area_um2=z.copy(), circularity=z.copy(),
            eccentricity=z.copy(), solidity=z.copy(), gray=z.copy(),
            hematoxylin=z.copy(), prob=z.copy(),
            labels=np.zeros(shape, dtype=np.int32), rejected={},
        )


# ---- pixels in, measurements out -----------------------------------------


def segment_array(
    rgb: np.ndarray,
    cfg: NucleiConfig,
    um2_per_px: float,
    core: tuple[slice, slice] | None = None,
    origin: tuple[float, float] = (0.0, 0.0),
    tissue_mask: np.ndarray | None = None,
) -> TileNuclei:
    """Segment and measure nuclei in one image.

    `core` is the sub-window of `rgb` this call owns; nuclei centred outside it
    are dropped as another tile's business. `origin` is the level-0 coordinate
    of the core's top-left corner, used to put centroids in slide space.
    `tissue_mask` is a core-shaped boolean array.
    """
    model = load_model(cfg.model_name, cfg.model_dir)

    if core is None:
        core = (slice(0, rgb.shape[0]), slice(0, rgb.shape[1]))
    oy, ox = core[0].start, core[1].start
    core_h = core[0].stop - oy
    core_w = core[1].stop - ox

    # Segment at `upscale`x, then convert every measurement back to level-0
    # units below, so this setting changes recall and nothing else.
    rgb, core, u = upscale_image(rgb, core, cfg.upscale)

    norm = normalize_percentile(rgb, core, cfg.norm_low_percentile, cfg.norm_high_percentile)

    labels, probs = predict_instances(
        model, norm, cfg.prob_threshold, cfg.nms_threshold
    )

    # Intensity is measured on the ORIGINAL pixels, not the normalized ones --
    # the cutoffs are meant to be physical, not per-tile-relative.
    gray = rgb.mean(axis=2).astype(np.float32)
    hema = rgb2hed(rgb.astype(np.float32) / 255.0)[..., 0].astype(np.float32)
    intensity = np.dstack([gray, hema])

    rows: list[tuple] = []
    kept_labels: list[int] = []
    rejected = {"margin": 0, "too_small": 0, "too_large": 0, "off_tissue": 0}

    # One upscaled pixel covers 1/u^2 of the physical area of a level-0 pixel.
    um2_per_px_eff = um2_per_px / (u * u)

    for r in regionprops(labels, intensity_image=intensity):
        cy, cx = r.centroid
        # Own this nucleus only if its centroid is in the core. Half-open
        # bounds make neighbouring tiles partition the plane exactly. Local
        # coordinates are returned to level-0 units here.
        lx, ly = cx / u - ox, cy / u - oy
        if not (0 <= lx < core_w and 0 <= ly < core_h):
            rejected["margin"] += 1
            continue

        area = r.area * um2_per_px_eff
        if area < cfg.min_area_um2:
            rejected["too_small"] += 1
            continue
        if area > cfg.max_area_um2:
            rejected["too_large"] += 1
            continue

        if tissue_mask is not None and cfg.mask_to_tissue:
            if not tissue_mask[int(ly), int(lx)]:
                rejected["off_tissue"] += 1
                continue

        perim = float(r.perimeter_crofton)
        circ = min(4.0 * np.pi * r.area / (perim * perim), 1.0) if perim > 0 else 0.0
        means = np.atleast_1d(r.intensity_mean)
        prob = float(probs[r.label - 1]) if r.label - 1 < probs.size else np.nan

        rows.append((
            origin[0] + lx, origin[1] + ly, area, circ,
            float(r.eccentricity), float(r.solidity),
            float(means[0]), float(means[1]), prob,
        ))
        kept_labels.append(r.label)

    if not rows:
        out = TileNuclei.empty((core_h, core_w))
        out.rejected = rejected
        return out

    arr = np.asarray(rows, dtype=np.float64)
    core_labels = relabel_core(labels[core], kept_labels)
    if u > 1:
        core_labels = cv2.resize(core_labels, (core_w, core_h),
                                 interpolation=cv2.INTER_NEAREST)
    return TileNuclei(
        **{name: arr[:, i] for i, name in enumerate(_COLS)},
        labels=core_labels,
        rejected=rejected,
    )


def segment_tile(
    slide: Slide,
    x: int,
    y: int,
    size: int,
    cfg: NucleiConfig,
    tissue: TissueMask | None = None,
) -> tuple[TileNuclei, np.ndarray]:
    """Read one tile with margins, segment it, and return the core RGB too."""
    margin = max(0, int(cfg.margin_px))
    padded, ox, oy = read_padded(slide, x, y, size, margin, cfg.level)
    core = (slice(oy, oy + size), slice(ox, ox + size))

    tmask = None
    if tissue is not None and cfg.mask_to_tissue:
        ds = slide.level_downsamples[cfg.level]
        tmask = tissue.tile_mask(
            int(x * ds), int(y * ds), int(size * ds), int(size * ds)
        )
        if tmask.shape != (size, size):  # tile_mask works in level-0 pixels
            import cv2

            tmask = cv2.resize(
                tmask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

    nuc = segment_array(
        padded, cfg, slide.um2_per_pixel(cfg.level),
        core=core, origin=(float(x), float(y)), tissue_mask=tmask,
    )
    return nuc, padded[core]


def scale_note(slide: Slide, cfg: NucleiConfig) -> str:
    """The scale the model is being run at, including `upscale`."""
    return core_scale_note(slide, cfg.level, cfg.model_name, cfg.upscale)


