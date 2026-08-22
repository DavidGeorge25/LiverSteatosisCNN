"""Tissue vs. background detection.

Background on these H&E slides is off-white/light gray glass; tissue is
pink/purple. In HSV that difference lives almost entirely in the saturation
channel, so an Otsu cut on S separates them robustly without depending on how
darkly a given slide was stained.

Detection runs at a low pyramid level (default 2, a 16x downsample) so the
whole slide fits in memory, then the mask is upsampled on demand.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import disk, remove_small_holes, remove_small_objects

from .config import TissueConfig
from .slide import Slide


@dataclass
class TissueMask:
    """A binary tissue mask plus the geometry needed to map it back to level 0."""

    mask: np.ndarray  # bool, shape (H, W) at `level`
    level: int
    downsample: float  # level-0 pixels per mask pixel
    um2_per_pixel: float  # square microns per mask pixel
    threshold: float  # the saturation cutoff actually used

    @property
    def shape(self) -> tuple[int, int]:
        return self.mask.shape  # type: ignore[return-value]

    @property
    def pixel_count(self) -> int:
        return int(self.mask.sum())

    @property
    def area_um2(self) -> float:
        return self.pixel_count * self.um2_per_pixel

    @property
    def area_mm2(self) -> float:
        return self.area_um2 / 1e6

    @property
    def coverage(self) -> float:
        """Fraction of the slide's area that is tissue."""
        return self.pixel_count / self.mask.size

    @property
    def component_count(self) -> int:
        return int(label(self.mask).max())

    def tissue_fraction(self, x0: int, y0: int, w: int, h: int) -> float:
        """Fraction of a LEVEL-0 box that is tissue.

        Used by the tiling stage to skip mostly-background tiles without
        touching level-0 pixels. The box is mapped into mask coordinates; a
        512px tile at 16x downsample covers 32x32 mask pixels, which is plenty
        of resolution for a 0.5 cutoff.
        """
        ds = self.downsample
        mx0 = int(np.floor(x0 / ds))
        my0 = int(np.floor(y0 / ds))
        mx1 = int(np.ceil((x0 + w) / ds))
        my1 = int(np.ceil((y0 + h) / ds))

        mh, mw = self.mask.shape
        mx0, my0 = max(0, mx0), max(0, my0)
        mx1, my1 = min(mw, mx1), min(mh, my1)
        if mx1 <= mx0 or my1 <= my0:
            return 0.0

        window = self.mask[my0:my1, mx0:mx1]
        return float(window.mean())

    def tile_mask(self, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        """Tissue mask for a LEVEL-0 box, upsampled to (h, w) pixels.

        Necessarily coarse -- a 512px tile covers only ~32x32 mask pixels at
        16x downsample -- so this excludes the off-tissue background of edge
        tiles at block resolution, not the fine tissue boundary.
        """
        ds = self.downsample
        mx0, my0 = int(np.floor(x0 / ds)), int(np.floor(y0 / ds))
        mx1, my1 = int(np.ceil((x0 + w) / ds)), int(np.ceil((y0 + h) / ds))
        mh, mw = self.mask.shape

        crop = np.zeros((max(1, my1 - my0), max(1, mx1 - mx0)), dtype=np.uint8)
        sx0, sy0 = max(0, mx0), max(0, my0)
        sx1, sy1 = min(mw, mx1), min(mh, my1)
        if sx1 > sx0 and sy1 > sy0:
            crop[sy0 - my0 : sy1 - my0, sx0 - mx0 : sx1 - mx0] = self.mask[
                sy0:sy1, sx0:sx1
            ]
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    def upsample_to(self, size: tuple[int, int]) -> np.ndarray:
        """Nearest-neighbour upsample of the mask to a (width, height) size."""
        w, h = size
        out = cv2.resize(
            self.mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        )
        return out.astype(bool)

    def bounding_boxes(self) -> list[tuple[int, int, int, int]]:
        """Level-0 (x, y, w, h) bounding boxes of each tissue component."""
        boxes = []
        for region in regionprops(label(self.mask)):
            min_row, min_col, max_row, max_col = region.bbox
            boxes.append(
                (
                    int(min_col * self.downsample),
                    int(min_row * self.downsample),
                    int((max_col - min_col) * self.downsample),
                    int((max_row - min_row) * self.downsample),
                )
            )
        return boxes


def _remove_small_holes(mask: np.ndarray, area_px: int) -> np.ndarray:
    """Version-safe wrapper: scikit-image 0.26 renamed `area_threshold` to
    `max_size` (and made the comparison inclusive)."""
    if "max_size" in _signature_params(remove_small_holes):
        return remove_small_holes(mask, max_size=area_px)
    return remove_small_holes(mask, area_threshold=area_px)


def _remove_small_objects(mask: np.ndarray, area_px: int) -> np.ndarray:
    """Version-safe wrapper: scikit-image 0.26 renamed `min_size` to `max_size`
    (it now drops objects smaller than *or equal to* the value)."""
    if "max_size" in _signature_params(remove_small_objects):
        return remove_small_objects(mask, max_size=area_px)
    return remove_small_objects(mask, min_size=area_px)


@lru_cache(maxsize=None)
def _signature_params(func) -> frozenset[str]:
    return frozenset(inspect.signature(func).parameters)


def _um_to_px(radius_um: float, mpp: float) -> int:
    """Convert a micron radius to an integer pixel radius (at least 1 if the
    caller asked for a nonzero size, so the operation is never a silent no-op)."""
    if radius_um <= 0:
        return 0
    return max(1, int(round(radius_um / mpp)))


def detect_tissue(slide: Slide, cfg: TissueConfig) -> TissueMask:
    """Detect tissue on `slide` and return a binary mask at `cfg.level`."""
    level = slide.resolve_level(cfg.level)
    rgb = slide.read_level(level)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    if cfg.method == "otsu":
        thresh = float(threshold_otsu(saturation))
    elif cfg.method == "fixed":
        thresh = float(cfg.saturation_threshold)
    else:
        raise ValueError(
            f"tissue.method must be 'otsu' or 'fixed', got {cfg.method!r}"
        )

    mask = saturation > thresh

    # Bright glass can pick up a little chromatic noise; force it to background.
    if cfg.max_value < 255:
        mask &= value <= cfg.max_value

    mpp_x, mpp_y = slide.mpp_at_level(level)
    mpp = (mpp_x + mpp_y) / 2.0
    um2_px = slide.um2_per_pixel(level)

    # Close first (bridge gaps within tissue), then open (drop thin bridges and
    # small specks). Both radii are specified in microns.
    close_r = _um_to_px(cfg.close_radius_um, mpp)
    if close_r:
        mask = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_CLOSE, disk(close_r).astype(np.uint8)
        ).astype(bool)

    open_r = _um_to_px(cfg.open_radius_um, mpp)
    if open_r:
        mask = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_OPEN, disk(open_r).astype(np.uint8)
        ).astype(bool)

    # Fill interior holes, then drop dust and debris outside the section.
    if cfg.max_hole_area_um2 > 0:
        mask = _remove_small_holes(mask, max(1, int(cfg.max_hole_area_um2 / um2_px)))

    if cfg.min_object_area_um2 > 0:
        mask = _remove_small_objects(mask, max(1, int(cfg.min_object_area_um2 / um2_px)))

    if cfg.keep_largest_n > 0:
        mask = _keep_largest(mask, cfg.keep_largest_n)

    return TissueMask(
        mask=mask,
        level=level,
        downsample=float(slide.level_downsamples[level]),
        um2_per_pixel=um2_px,
        threshold=thresh,
    )


def _keep_largest(mask: np.ndarray, n: int) -> np.ndarray:
    labels = label(mask)
    regions = sorted(regionprops(labels), key=lambda r: r.area, reverse=True)
    keep = {r.label for r in regions[:n]}
    if not keep:
        return mask
    return np.isin(labels, list(keep))
