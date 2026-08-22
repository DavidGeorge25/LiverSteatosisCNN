"""Hepatocyte segmentation: StarDist nuclei, then cell bodies around them.

One tile at a time, level 0, never the whole slide.

`2D_versatile_he` segments NUCLEI, not cells. Ballooning is a property of the
cell body -- an enlarged, pale, rarefied cytoplasm around a nucleus that is
itself normal-sized and often shoved against the membrane -- so a nuclear
segmentation on its own measures almost the wrong thing. The cell body is
recovered by growing each nucleus outward:

    nuclei (StarDist seeds)
      -> territory = tissue, minus real voids, capped at max_expansion_um
      -> watershed over [distance-from-nucleus + edge_weight * eosin edges]
      -> cell = the flooded region, cytoplasm = cell minus nucleus

Cells therefore stop at sinusoids and vessel lumens, at each other, and at a
hard radius.

WHAT THIS AREA ACTUALLY IS -- read before using it. The result is a
TERRITORY, not a membrane-bounded cell, and the field is named
`territory_area_um2` to keep that visible in every CSV. Measured on this
cohort, cells cover 91-94% of the tissue and the median territory equals the
tile area divided by the seed count, which is the definition of a Voronoi
tessellation: the number is reporting how far apart nuclei are, not how big
cells are. That inflation was ~2x (860-930 um^2 measured vs
~350-500 um^2 expected) and was originally attributed here to stereology --
that in a 4-5 um section about half of hepatocyte profiles contain no nucleus
in-plane, so each detected nucleus absorbs its anucleate neighbours. That
explanation was mostly wrong, and measurably so: it was missed SEEDS. Running
StarDist at `upscale: 2` (the model is trained near 40x; these slides are 20x)
takes the median hepatocyte territory from 720 to 433 um^2 and the median
diameter from 30.3 to 23.5 um, both onto their expected values, and finds 750
non-hepatocyte nuclei per 40 tiles where upscale 1 found 18. A real
stereological component remains, but it is much smaller than the segmentation
artefact that was masking it. Ballooning does push neighbouring nuclei
apart, so the number is directionally right, but it carries stereological
noise and is deliberately downweighted in `RankingConfig.weights`. The
cytoplasm pallor and texture features do not have this problem: they describe
the pixels inside the territory, and a ballooned cell's territory is pale and
wispy wherever its boundary is drawn.

Two details matter for honest numbers out of a tiled run:

*Margins.* Each tile is read with `margin_px` of extra context on every side
and segmented at that larger size; only cells whose NUCLEAR centroid falls in
the core tile are kept. A cell straddling a seam is therefore grown against its
real surroundings and counted exactly once, and every measurement below is
taken on the padded image, so no cell area or cytoplasm is lost to the seam.

*Normalization.* StarDist wants percentile-normalized input. The percentiles
come from the core tile only, so white padding at a slide edge cannot drag the
range around.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.color import rgb2hed
from skimage.filters import gaussian, sobel
from skimage.measure import regionprops
from skimage.segmentation import watershed

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
from . import features as feat
from .config import BallooningConfig, SegmentationConfig

# Geometry columns. Cytoplasm intensity and texture columns come from
# `features.feature_names` and travel in `TileCells.measurements`.
_COLS = (
    "x", "y", "nucleus_x", "nucleus_y",
    "territory_area_um2", "nucleus_area_um2", "cyto_area_um2",
    "circularity", "solidity", "eccentricity",
    "equivalent_diameter_um", "nc_ratio", "prob",
)


@dataclass
class TileCells:
    """Per-cell measurements for one tile, in level-0 slide coordinates.

    `cell_labels` and `nucleus_labels` are core-tile label images renumbered
    1..N to match the measurement arrays, so an overlay always shows exactly
    what the numbers describe. `contours` are level-0 polygons of the full
    (unclipped) cell, kept so a review crop can outline the actual cell without
    re-running the segmentation.
    """

    x: np.ndarray  # cell centroid, level-0 px
    y: np.ndarray
    nucleus_x: np.ndarray  # nuclear centroid, level-0 px
    nucleus_y: np.ndarray
    territory_area_um2: np.ndarray  # see the module docstring before using
    nucleus_area_um2: np.ndarray
    cyto_area_um2: np.ndarray
    circularity: np.ndarray  # 4*pi*A / P^2, Crofton perimeter
    solidity: np.ndarray
    eccentricity: np.ndarray  # 0 = circle, 1 = line
    equivalent_diameter_um: np.ndarray
    nc_ratio: np.ndarray  # nuclear area / territory area
    prob: np.ndarray  # StarDist objectness of the seed nucleus
    is_hepatocyte: np.ndarray  # bool; the population ballooning is judged over
    cell_labels: np.ndarray
    nucleus_labels: np.ndarray
    measurements: dict[str, np.ndarray] = field(default_factory=dict)
    contours: list[np.ndarray] = field(default_factory=list)
    nucleus_contours: list[np.ndarray] = field(default_factory=list)
    rejected: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.x.size)

    @classmethod
    def empty(cls, shape: tuple[int, int]) -> "TileCells":
        z = np.zeros(0, dtype=np.float64)
        return cls(
            **{name: z.copy() for name in _COLS},
            is_hepatocyte=np.zeros(0, dtype=bool),
            cell_labels=np.zeros(shape, dtype=np.int32),
            nucleus_labels=np.zeros(shape, dtype=np.int32),
        )


# ---- cell bodies ----------------------------------------------------------


def expand_to_cells(
    rgb: np.ndarray,
    nucleus_labels: np.ndarray,
    cfg: SegmentationConfig,
    mpp: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Grow each nucleus into a cell body. Returns (cell_labels, void_mask).

    The elevation the watershed floods over is distance-from-nucleus plus an
    optional eosin edge term. Distance alone tessellates the tissue between
    nuclei; the edge term lets a boundary snap to a membrane or a sinusoid wall
    when there is one. Both are normalized to [0, 1] first so `edge_weight`
    means the same thing on any slide.
    """
    gray = rgb.mean(axis=2)

    # White voids: sinusoid and vessel lumens, fat droplets, tears. Cells are
    # not grown into them, which is what keeps a hepatocyte from swallowing the
    # sinusoid beside it and reading as enlarged.
    void = gray > cfg.void_intensity
    if cfg.min_void_area_um2 > 0:
        # Only regions big enough to be a real lumen count. Pale speckle inside
        # rarefied cytoplasm is the ballooning signal itself -- excluding it
        # here would delete the feature before it is ever measured.
        void = _remove_small(void, max(1, int(cfg.min_void_area_um2 / (mpp * mpp))))

    has_nucleus = nucleus_labels > 0
    dist = distance_transform_edt(~has_nucleus)

    max_expansion_px = max(1.0, cfg.max_expansion_um / mpp)
    territory = (~void) & (dist <= max_expansion_px)
    # A nucleus is always part of its own cell, even if it is dark enough or
    # bright enough to have been excluded above.
    territory |= has_nucleus

    elevation = dist / max(max_expansion_px, 1e-6)
    if cfg.edge_weight > 0:
        # Edges on the EOSIN channel. On grey, the strongest edge in the image
        # is the nuclear rim, and flooding from a marker cannot cross it -- so
        # every cell would be trapped at its own nuclear boundary. Eosin is the
        # cytoplasmic stain and is comparatively flat across nuclei, so its
        # edges are membranes and sinusoid walls: the boundaries we want.
        eosin = rgb2hed(rgb.astype(np.float32) / 255.0)[..., 1]
        sigma_px = max(cfg.edge_smoothing_um / mpp, 1e-3)
        edges = sobel(gaussian(eosin, sigma=sigma_px))
        hi = float(np.percentile(edges, 99.0))
        if hi > 0:
            elevation = elevation + cfg.edge_weight * np.clip(edges / hi, 0, 1)

    cells = watershed(elevation, markers=nucleus_labels, mask=territory)
    return cells.astype(np.int32), void


# ---- pixels in, measurements out ------------------------------------------


def segment_array(
    rgb: np.ndarray,
    cfg: BallooningConfig,
    mpp: float,
    um2_per_px: float,
    core: tuple[slice, slice] | None = None,
    origin: tuple[float, float] = (0.0, 0.0),
    tissue_mask: np.ndarray | None = None,
    measure: bool = True,
) -> TileCells:
    """Segment and measure hepatocytes in one image.

    `core` is the sub-window of `rgb` this call owns; cells whose nucleus is
    centred outside it are dropped as another tile's business. `origin` is the
    level-0 coordinate of the core's top-left corner, used to put centroids in
    slide space. `tissue_mask` is a core-shaped boolean array.

    `measure=False` skips cytoplasm intensity and texture, which is what the
    segmentation QC pass wants -- it only draws boundaries, and the texture
    maps are the expensive part.
    """
    scfg = cfg.segmentation
    model = load_model(scfg.model_name, scfg.model_dir)

    if core is None:
        core = (slice(0, rgb.shape[0]), slice(0, rgb.shape[1]))

    # Level-0 core size, fixed before any upscaling: the label images and the
    # tissue mask stay in level-0 pixels whatever the model runs at.
    core_h_l0 = core[0].stop - core[0].start
    core_w_l0 = core[1].stop - core[1].start

    # Everything from here to the coordinate conversion below is in SEGMENTATION
    # pixels, which are level-0 pixels only when upscale == 1.
    rgb, core, u = upscale_image(rgb, core, scfg.upscale)
    oy, ox = core[0].start, core[1].start
    core_h = core[0].stop - oy
    core_w = core[1].stop - ox
    mpp_seg = mpp / u
    um2_per_px_seg = um2_per_px / (u * u)

    norm = normalize_percentile(rgb, core, scfg.norm_low_percentile, scfg.norm_high_percentile)

    nuc_labels, probs = predict_instances(
        model, norm, scfg.prob_threshold, scfg.nms_threshold
    )

    rejected = {
        "nucleus_too_small": 0, "nucleus_too_large": 0, "margin": 0,
        "off_tissue": 0, "cell_too_small": 0, "cell_too_large": 0,
        "no_cytoplasm": 0,
    }

    # Drop implausible nuclei BEFORE expansion -- a speckle seed would other-
    # wise claim a territory and carve a real cell in half.
    nuc_labels = _filter_nuclei(nuc_labels, scfg, um2_per_px_seg, rejected)

    cell_labels, _void = expand_to_cells(rgb, nuc_labels, scfg, mpp_seg)

    channels = feat.build_channels(rgb, cfg.features, mpp_seg) if measure else None
    nucleus_props = {r.label: r for r in regionprops(nuc_labels)}

    rows: list[tuple] = []
    flags: list[bool] = []
    kept_labels: list[int] = []
    contours: list[np.ndarray] = []
    nuc_contours: list[np.ndarray] = []
    per_cell: list[dict[str, float]] = []

    for r in regionprops(cell_labels):
        nuc = nucleus_props.get(r.label)
        # Ownership is decided by the NUCLEUS, not the cell body: the nucleus is
        # what StarDist actually detected, and a territory can drift into the
        # margin without meaning the cell belongs to the neighbouring tile.
        if nuc is None:
            continue
        ncy, ncx = nuc.centroid
        lx, ly = ncx - ox, ncy - oy
        # Half-open bounds make neighbouring tiles partition the plane exactly.
        if not (0 <= lx < core_w and 0 <= ly < core_h):
            rejected["margin"] += 1
            continue

        if tissue_mask is not None and scfg.mask_to_tissue:
            if not tissue_mask[int(ly / u), int(lx / u)]:
                rejected["off_tissue"] += 1
                continue

        area = r.area * um2_per_px_seg
        nucleus_area = nuc.area * um2_per_px_seg
        cyto_area = area - nucleus_area

        if area < scfg.min_cell_area_um2:
            rejected["cell_too_small"] += 1
            continue
        if area > scfg.max_cell_area_um2:
            rejected["cell_too_large"] += 1
            continue
        if cyto_area < scfg.min_cytoplasm_area_um2:
            rejected["no_cytoplasm"] += 1
            continue

        cy, cx = r.centroid
        perim = float(r.perimeter_crofton)
        circ = min(4.0 * np.pi * r.area / (perim * perim), 1.0) if perim > 0 else 0.0
        prob = float(probs[r.label - 1]) if r.label - 1 < probs.size else np.nan

        # Segmentation pixels -> level 0 before `origin`, which is already
        # level 0, is added.
        rows.append((
            origin[0] + (cx - ox) / u, origin[1] + (cy - oy) / u,
            origin[0] + lx / u, origin[1] + ly / u,
            area, nucleus_area, cyto_area,
            circ, float(r.solidity), float(r.eccentricity),
            float(r.equivalent_diameter_area) * mpp_seg,
            nucleus_area / area if area > 0 else np.nan,
            prob,
        ))
        # Endothelial, Kupffer and immune cells are segmented too, and they are
        # kept as seeds so they claim their own territory rather than being
        # absorbed. They are excluded from the ballooning population here.
        flags.append(nucleus_area >= scfg.min_hepatocyte_nucleus_area_um2)
        kept_labels.append(r.label)

        r0, c0, r1, c1 = r.bbox
        cell_patch = cell_labels[r0:r1, c0:c1] == r.label
        nuc_patch = nuc_labels[r0:r1, c0:c1] == r.label
        contours.append(_contour(cell_patch, origin, (c0 - ox, r0 - oy), u))
        nuc_contours.append(_contour(nuc_patch, origin, (c0 - ox, r0 - oy), u))

        if measure:
            per_cell.append(feat.measure_cell(
                channels[r0:r1, c0:c1],
                cell_patch & ~nuc_patch,
                channels[r0:r1, c0:c1, feat.CHANNELS.index("gray")],
                cfg.features,
            ))

    if not rows:
        out = TileCells.empty((core_h_l0, core_w_l0))
        out.rejected = rejected
        return out

    arr = np.asarray(rows, dtype=np.float64)
    measurements: dict[str, np.ndarray] = {}
    if measure:
        for name in feat.feature_names(cfg.features):
            measurements[name] = np.asarray(
                [c.get(name, np.nan) for c in per_cell], dtype=np.float64
            )

    return TileCells(
        **{name: arr[:, i] for i, name in enumerate(_COLS)},
        is_hepatocyte=np.asarray(flags, dtype=bool),
        cell_labels=_to_level0(relabel_core(cell_labels[core], kept_labels),
                               core_w_l0, core_h_l0),
        nucleus_labels=_to_level0(relabel_core(nuc_labels[core], kept_labels),
                                  core_w_l0, core_h_l0),
        measurements=measurements,
        contours=contours,
        nucleus_contours=nuc_contours,
        rejected=rejected,
    )


def segment_tile(
    slide: Slide,
    x: int,
    y: int,
    size: int,
    cfg: BallooningConfig,
    tissue: TissueMask | None = None,
    measure: bool = True,
) -> tuple[TileCells, np.ndarray]:
    """Read one tile with margins, segment it, and return the core RGB too.

    `x` and `y` are level-0 pixels, which is what `core.tiling` yields.
    """
    scfg = cfg.segmentation
    if scfg.level != 0:
        # Every coordinate below (and every contour) is emitted in level-0
        # pixels. Supporting a downsampled level would mean scaling all of them
        # by the level's downsample, and nuclei do not survive a 4x downsample
        # anyway -- so refuse rather than silently emit coordinates in the
        # wrong frame.
        raise ValueError(
            f"ballooning.segmentation.level must be 0, got {scfg.level}: "
            "nuclei are a few microns across and do not survive downsampling, "
            "and cell coordinates are reported in level-0 pixels"
        )

    margin = max(0, int(scfg.margin_px))
    padded, ox, oy = read_padded(slide, x, y, size, margin)
    core = (slice(oy, oy + size), slice(ox, ox + size))

    tmask = None
    if tissue is not None and scfg.mask_to_tissue:
        tmask = tissue.tile_mask(x, y, size, size)
        if tmask.shape != (size, size):
            import cv2

            tmask = cv2.resize(
                tmask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

    mx, my = slide.mpp_at_level(0)
    cells = segment_array(
        padded, cfg, (mx + my) / 2.0, slide.um2_per_pixel(0),
        core=core, origin=(float(x), float(y)), tissue_mask=tmask, measure=measure,
    )
    return cells, padded[core]


def _contour(patch: np.ndarray, origin: tuple[float, float],
             offset: tuple[int, int], upscale: int = 1) -> np.ndarray:
    """Largest outer contour of a bbox-cropped mask, in level-0 coordinates.

    `offset` is the patch's position in SEGMENTATION pixels relative to the
    core tile; `upscale` divides that back to level 0 before `origin` -- which
    is already level 0 -- is added. Getting this order wrong would place the
    contour u times too far from the tile corner, which is invisible at
    upscale 1 and puts every review crop on the wrong cell above it.
    """
    import cv2

    cnts, _ = cv2.findContours(
        patch.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not cnts:
        return np.zeros((0, 2), dtype=np.int32)
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    u = max(1, int(upscale))
    c = (c + np.array([offset[0], offset[1]], dtype=np.float64)) / u
    return np.rint(c + np.array([origin[0], origin[1]])).astype(np.int32)


def _to_level0(labels: np.ndarray, width: int, height: int) -> np.ndarray:
    """Shrink a segmentation-resolution label image back to level-0 pixels.

    Nearest-neighbour, because these are labels: any interpolation would invent
    object ids that never existed on the boundaries between two cells. A no-op
    when the shapes already agree, which is every run at upscale 1.
    """
    if labels.shape == (height, width):
        return labels
    import cv2

    return cv2.resize(
        labels, (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(np.int32)


def _remove_small(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Drop connected components below `min_area_px`.

    Uses cv2 rather than `skimage.morphology.remove_small_objects`, whose
    keyword changed name and comparison semantics in scikit-image 0.26 (see the
    version-safe wrappers in `core/tissue.py`). Connected components have no
    such ambiguity.
    """
    import cv2

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return mask
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area_px
    return keep[labels]


def _filter_nuclei(
    labels: np.ndarray,
    cfg: SegmentationConfig,
    um2_per_px: float,
    rejected: dict[str, int],
) -> np.ndarray:
    """Zero out nuclei outside the plausible area band, in place of a seed."""
    out = labels.copy()
    for r in regionprops(labels):
        area = r.area * um2_per_px
        if area < cfg.min_nucleus_area_um2:
            rejected["nucleus_too_small"] += 1
        elif area > cfg.max_nucleus_area_um2:
            rejected["nucleus_too_large"] += 1
        else:
            continue
        out[labels == r.label] = 0
    return out


def scale_note(slide: Slide, cfg: SegmentationConfig) -> str:
    """The scale the model is being run at, including `upscale`."""
    return core_scale_note(slide, cfg.level, cfg.model_name, cfg.upscale)


