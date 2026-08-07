"""Fat droplet pseudo-labeling.

Lipid dissolves out during processing, so fat droplets are white/clear voids in
an H&E section. Thresholding for white finds them -- along with vein lumens,
sinusoids, tears, and the tissue-edge gap. Shape filters separate the two:
droplets are round and convex, the false positives are elongated or irregular.

Every filter is expressed in physical units (square microns) via the slide's
MPP, so a parameter means the same thing regardless of magnification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import disk

from .config import FatConfig

MACRO = "macro"
MICRO = "micro"


@dataclass(frozen=True)
class Droplet:
    """One accepted connected component."""

    klass: str  # MACRO | MICRO
    area_um2: float
    equivalent_diameter_um: float
    circularity: float
    solidity: float
    centroid_x: float  # tile-local pixels
    centroid_y: float
    touches_border: bool


@dataclass
class FatResult:
    """Per-tile fat detection output."""

    macro_mask: np.ndarray
    micro_mask: np.ndarray
    droplets: list[Droplet]
    tissue_area_um2: float
    threshold: float
    rejected: dict[str, int] = field(default_factory=dict)
    # Border-touching components that PASSED every other filter. Reported
    # whether or not exclude_border is on, so the tradeoff is visible.
    border_count: int = 0
    border_area_um2: float = 0.0
    border_excluded: bool = False
    include_micro: bool = False

    # ---- masks ----------------------------------------------------------

    @property
    def label_mask(self) -> np.ndarray:
        """The pseudo-label actually exported.

        `micro_mask` is always populated so QC can show both classes; whether
        micro is part of the training label is a separate decision.
        """
        return self.macro_mask | self.micro_mask if self.include_micro else self.macro_mask

    # ---- per-class aggregates -------------------------------------------

    def _of(self, klass: str) -> list[Droplet]:
        return [d for d in self.droplets if d.klass == klass]

    def count(self, klass: str | None = None) -> int:
        return len(self.droplets if klass is None else self._of(klass))

    def area_um2(self, klass: str | None = None) -> float:
        ds = self.droplets if klass is None else self._of(klass)
        return float(sum(d.area_um2 for d in ds))

    def fat_fraction(self, klass: str | None = None) -> float:
        """Droplet area as a fraction of TISSUE area in the tile."""
        if self.tissue_area_um2 <= 0:
            return 0.0
        return self.area_um2(klass) / self.tissue_area_um2

    def mean_area_um2(self, klass: str | None = None) -> float:
        ds = self.droplets if klass is None else self._of(klass)
        return float(np.mean([d.area_um2 for d in ds])) if ds else 0.0

    def median_area_um2(self, klass: str | None = None) -> float:
        ds = self.droplets if klass is None else self._of(klass)
        return float(np.median([d.area_um2 for d in ds])) if ds else 0.0

    def areas(self, klass: str | None = None) -> np.ndarray:
        ds = self.droplets if klass is None else self._of(klass)
        return np.array([d.area_um2 for d in ds], dtype=float)


def _circularity(area_px: float, perimeter_px: float) -> float:
    """4*pi*A / P^2. Dimensionless, so pixel units are fine (MPP is isotropic).
    Approaches 1 for a perfect disc; low for elongated or ragged shapes."""
    if perimeter_px <= 0:
        return 0.0
    return float(4.0 * np.pi * area_px / (perimeter_px**2))


def _equiv_diameter_px(region) -> float:
    """scikit-image 0.26 renamed `equivalent_diameter` to
    `equivalent_diameter_area`."""
    try:
        return float(region.equivalent_diameter_area)
    except AttributeError:
        return float(region.equivalent_diameter)


def _white_mask(rgb: np.ndarray, cfg: FatConfig) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if cfg.method == "fixed":
        thresh = float(cfg.white_threshold)
    elif cfg.method == "otsu":
        # Degenerate on a uniform tile; fall back to the fixed cutoff.
        thresh = float(threshold_otsu(gray)) if gray.min() != gray.max() else float(
            cfg.white_threshold
        )
    else:
        raise ValueError(f"fat.method must be 'fixed' or 'otsu', got {cfg.method!r}")
    return gray >= thresh, thresh


@dataclass(frozen=True)
class Component:
    """One connected white region, before any shape filtering."""

    label: int
    area_um2: float
    circularity: float
    solidity: float
    # Eccentricity of the best-fit ellipse: 0 is a circle, 1 a line segment.
    # Circularity (4*pi*A/P^2) is dominated by boundary raggedness and barely
    # responds to elongation -- a smooth 2.5:1 ellipse still scores ~0.87 --
    # so vessel lumens need this separate measure to be rejected on shape.
    eccentricity: float
    equivalent_diameter_um: float
    centroid_x: float
    centroid_y: float
    touches_border: bool


@dataclass
class ComponentSet:
    """All candidate components in a tile, plus the label image they index."""

    labels: np.ndarray
    components: list[Component]
    threshold: float
    tissue_area_um2: float


def extract_components(
    rgb: np.ndarray,
    tissue_mask: np.ndarray,
    cfg: FatConfig,
    um2_per_px: float,
) -> ComponentSet:
    """Threshold, clean up, and measure every white region in a tile.

    This is the expensive half of detection and depends only on the threshold
    and morphology settings -- not on the area/shape cutoffs. Parameter sweeps
    over the shape filters can therefore extract once and re-classify cheaply.
    """
    white, thresh = _white_mask(rgb, cfg)
    white &= tissue_mask

    # Opening removes speckle and pinches apart droplets joined by a thin
    # bridge; closing then repairs ragged droplet borders.
    if cfg.open_radius_px > 0:
        white = cv2.morphologyEx(
            white.astype(np.uint8), cv2.MORPH_OPEN,
            disk(cfg.open_radius_px).astype(np.uint8),
        ).astype(bool)
    if cfg.close_radius_px > 0:
        white = cv2.morphologyEx(
            white.astype(np.uint8), cv2.MORPH_CLOSE,
            disk(cfg.close_radius_px).astype(np.uint8),
        ).astype(bool)

    # Droplets often contain a compressed nucleus or debris speck.
    if cfg.fill_holes:
        white = binary_fill_holes(white)

    h, w = white.shape
    labels = label(white)
    components: list[Component] = []
    for r in regionprops(labels):
        min_row, min_col, max_row, max_col = r.bbox
        components.append(
            Component(
                label=int(r.label),
                area_um2=r.area * um2_per_px,
                circularity=_circularity(r.area, r.perimeter),
                solidity=float(r.solidity),
                eccentricity=float(r.eccentricity),
                equivalent_diameter_um=_equiv_diameter_px(r) * np.sqrt(um2_per_px),
                centroid_x=float(r.centroid[1]),
                centroid_y=float(r.centroid[0]),
                touches_border=(
                    min_row == 0 or min_col == 0 or max_row == h or max_col == w
                ),
            )
        )

    return ComponentSet(
        labels=labels,
        components=components,
        threshold=thresh,
        tissue_area_um2=float(tissue_mask.sum()) * um2_per_px,
    )


def classify(c: Component, cfg: FatConfig) -> tuple[str | None, str | None]:
    """Assign a component to MACRO / MICRO, or reject it.

    Returns (class, rejection_reason); exactly one is non-None.
    """
    if cfg.min_area_um2 <= c.area_um2 <= cfg.max_area_um2:
        if c.circularity < cfg.circularity_min:
            return None, "low_circularity"
        if c.solidity < cfg.solidity_min:
            return None, "low_solidity"
        if c.eccentricity > cfg.max_eccentricity:
            return None, "elongated"
        return MACRO, None
    if cfg.micro_min_area_um2 <= c.area_um2 < cfg.min_area_um2:
        # Below the macro cutoff -> microvesicular candidate. Always measured;
        # `include_microvesicular` only decides whether it is written into the
        # exported pseudo-label.
        if c.circularity < cfg.micro_circularity_min:
            return None, "low_circularity"
        if c.solidity < cfg.micro_solidity_min:
            return None, "low_solidity"
        return MICRO, None
    if c.area_um2 > cfg.max_area_um2:
        return None, "too_large"
    return None, "too_small"


@dataclass
class Tally:
    """Scalar aggregate of a tile's components -- no masks allocated.

    `build_result` paints every accepted component into a full-size mask, which
    is the bulk of the cost when a parameter sweep re-classifies the same tile
    dozens of times. Sweeps and cohort surveys only need the numbers, so they
    use this instead; it is additive across tiles.
    """

    tissue_area_um2: float = 0.0
    macro_count: int = 0
    micro_count: int = 0
    macro_area_um2: float = 0.0
    micro_area_um2: float = 0.0
    border_count: int = 0
    border_area_um2: float = 0.0
    tiles: int = 0
    rejected: dict[str, int] = field(default_factory=dict)

    def __iadd__(self, other: "Tally") -> "Tally":
        self.tissue_area_um2 += other.tissue_area_um2
        self.macro_count += other.macro_count
        self.micro_count += other.micro_count
        self.macro_area_um2 += other.macro_area_um2
        self.micro_area_um2 += other.micro_area_um2
        self.border_count += other.border_count
        self.border_area_um2 += other.border_area_um2
        self.tiles += other.tiles
        for k, v in other.rejected.items():
            self.rejected[k] = self.rejected.get(k, 0) + v
        return self

    def _frac(self, area: float) -> float:
        return area / self.tissue_area_um2 if self.tissue_area_um2 > 0 else 0.0

    @property
    def macro_fraction(self) -> float:
        return self._frac(self.macro_area_um2)

    @property
    def micro_fraction(self) -> float:
        return self._frac(self.micro_area_um2)

    @property
    def total_fraction(self) -> float:
        return self._frac(self.macro_area_um2 + self.micro_area_um2)


def tally(cset: ComponentSet, cfg: FatConfig) -> Tally:
    """Classify components and accumulate scalars only. Mirrors `build_result`
    exactly -- same `classify`, same border handling -- minus the masks."""
    t = Tally(tissue_area_um2=cset.tissue_area_um2, tiles=1)
    for c in cset.components:
        klass, reason = classify(c, cfg)
        if klass is None:
            t.rejected[reason] = t.rejected.get(reason, 0) + 1
            continue
        if c.touches_border:
            t.border_count += 1
            t.border_area_um2 += c.area_um2
            if cfg.exclude_border:
                t.rejected["border"] = t.rejected.get("border", 0) + 1
                continue
        if klass == MACRO:
            t.macro_count += 1
            t.macro_area_um2 += c.area_um2
        else:
            t.micro_count += 1
            t.micro_area_um2 += c.area_um2
    return t


def detect_fat(
    rgb: np.ndarray,
    tissue_mask: np.ndarray,
    cfg: FatConfig,
    um2_per_px: float,
) -> FatResult:
    """Detect fat droplets in a single tile.

    `tissue_mask` restricts the search to tissue, which removes the off-tissue
    background of edge tiles at the resolution of the tissue mask.
    """
    cset = extract_components(rgb, tissue_mask, cfg, um2_per_px)
    return build_result(cset, cfg)


def build_result(cset: ComponentSet, cfg: FatConfig) -> FatResult:
    """Apply the shape/area filters to already-measured components."""
    shape = cset.labels.shape
    macro = np.zeros(shape, dtype=bool)
    micro = np.zeros(shape, dtype=bool)
    droplets: list[Droplet] = []
    rejected = {
        "too_small": 0, "too_large": 0,
        "low_circularity": 0, "low_solidity": 0, "elongated": 0, "border": 0,
    }
    border_count = 0
    border_area = 0.0

    for c in cset.components:
        klass, reason = classify(c, cfg)
        if klass is None:
            rejected[reason] += 1
            continue

        area_um2 = c.area_um2
        touches = c.touches_border
        if touches:
            border_count += 1
            border_area += area_um2
            if cfg.exclude_border:
                rejected["border"] += 1
                continue

        (macro if klass == MACRO else micro)[cset.labels == c.label] = True
        droplets.append(
            Droplet(klass, area_um2, c.equivalent_diameter_um,
                    c.circularity, c.solidity, c.centroid_x, c.centroid_y, touches)
        )

    return FatResult(
        macro_mask=macro,
        micro_mask=micro,
        droplets=droplets,
        tissue_area_um2=cset.tissue_area_um2,
        threshold=cset.threshold,
        rejected=rejected,
        border_count=border_count,
        border_area_um2=border_area,
        border_excluded=cfg.exclude_border,
        include_micro=cfg.include_microvesicular,
    )
