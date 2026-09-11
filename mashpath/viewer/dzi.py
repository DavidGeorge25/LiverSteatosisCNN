"""DeepZoom geometry, shared by the slide and its overlay.

The viewer draws two pyramids on top of each other: the tissue, and the fat
mask the model predicted. They only line up if both answer to the SAME tile
grid, so the geometry lives here once and both tile sources are built from it.
A slide pyramid from openslide's own `DeepZoomGenerator` and a mask pyramid
computed some other way would agree at most zoom levels and drift at the
edges, which reads as "the model is slightly off" rather than as a bug in the
viewer.

THE OVERLAY IS NEVER PRODUCED BY RUNNING THE MODEL AT A LOW PYRAMID LEVEL.
The U-Net was trained on 512 px level-0 tiles at 0.4953 um/px and has no scale
augmentation -- `train/augment.py` is colour only, and geometry is flips and
90-degree rotations. Run it 4x downsampled and it reports 0.06x the fat
(measured: 14.54% -> 0.81% on identical windows). So inference happens once, at
native resolution, in `precompute.py`; what this module downsamples for a
zoomed-out view is the MASK, never the input. Display scale and inference scale
are decoupled, and that is the whole reason the number on screen is stable as
the user zooms.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from ..core.slide import Slide

TILE_SIZE = 512
OVERLAP = 0

# Level-0 grid the precompute pass writes its full-resolution mask tiles on.
# Same 512 as `TilingConfig.tile_size`, because those mask tiles ARE the tiles
# the model was run on -- see `precompute.py`.
MASK_TILE = 512


class DeepZoom:
    """Standard DeepZoom pyramid geometry over a level-0 (width, height).

    Level `max_level` is full resolution; each level down halves both axes,
    ending at a 1x1 level 0. This is the layout OpenSeadragon's `.dzi` reader
    expects, and re-deriving it here (rather than importing openslide's
    generator) is what lets the mask share it.
    """

    def __init__(self, width: int, height: int, tile_size: int = TILE_SIZE,
                 overlap: int = OVERLAP):
        self.width = int(width)
        self.height = int(height)
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        self.max_level = max(1, math.ceil(math.log2(max(self.width, self.height))))

    @property
    def level_count(self) -> int:
        return self.max_level + 1

    def scale(self, level: int) -> float:
        """Level-0 pixels per pixel at `level`. 1.0 at full resolution."""
        return float(2 ** (self.max_level - level))

    def level_dimensions(self, level: int) -> tuple[int, int]:
        s = self.scale(level)
        return (max(1, math.ceil(self.width / s)), max(1, math.ceil(self.height / s)))

    def tile_counts(self, level: int) -> tuple[int, int]:
        w, h = self.level_dimensions(level)
        return (math.ceil(w / self.tile_size), math.ceil(h / self.tile_size))

    def tile_box(self, level: int, col: int, row: int
                 ) -> tuple[int, int, int, int, int, int]:
        """Where a tile lives, in both coordinate systems.

        Returns `(x0, y0, w0, h0, tw, th)`: the level-0 box to read, and the
        pixel size the tile must be rendered at. The two differ by `scale`,
        and keeping both explicit is what stops an off-by-one at the right and
        bottom edges, where the last tile is a partial one.
        """
        s = self.scale(level)
        lw, lh = self.level_dimensions(level)
        lx0 = col * self.tile_size
        ly0 = row * self.tile_size
        if lx0 >= lw or ly0 >= lh or col < 0 or row < 0:
            raise IndexError(f"tile ({col},{row}) is outside level {level}")
        tw = min(self.tile_size, lw - lx0)
        th = min(self.tile_size, lh - ly0)
        x0 = int(lx0 * s)
        y0 = int(ly0 * s)
        # Clamp to the slide: the last tile of a level usually overhangs.
        x1 = min(self.width, int(math.ceil((lx0 + tw) * s)))
        y1 = min(self.height, int(math.ceil((ly0 + th) * s)))
        return x0, y0, max(1, x1 - x0), max(1, y1 - y0), tw, th

    def dzi_xml(self, fmt: str = "jpg") -> str:
        """The .dzi descriptor.

        `fmt` is used by OpenSeadragon as the literal file EXTENSION, not as a
        media type -- `Format="jpeg"` makes it request `0_0.jpeg`. So this must
        match the extension actually written on disk. The dynamic server hid
        this by stripping the extension before dispatching, which meant the
        mismatch only appeared in the static export, where it showed up as a
        blank tissue layer under a perfectly good overlay.
        """
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"'
            f' Format="{fmt}" Overlap="{self.overlap}"'
            f' TileSize="{self.tile_size}">'
            f'<Size Width="{self.width}" Height="{self.height}"/>'
            "</Image>"
        )


# ---- the tissue pyramid ---------------------------------------------------


class SlideTileSource:
    """DeepZoom tiles read straight from the slide.

    Reads from the coarsest native pyramid level that is still finer than the
    requested scale, then downsamples the rest of the way. Reading level 0 for
    a zoomed-out tile would be correct and unusably slow -- a level-0 read of
    the whole slide is 1.4 Gpx -- and reading a level coarser than requested
    would upsample, which looks soft and hides exactly the detail the demo is
    about.
    """

    def __init__(self, slide: Slide, tile_size: int = TILE_SIZE):
        self.slide = slide
        w, h = slide.dimensions
        self.dz = DeepZoom(w, h, tile_size)

    def _best_level(self, scale: float) -> int:
        """Finest native level whose downsample does not exceed `scale`."""
        best, best_ds = 0, 1.0
        for i, ds in enumerate(self.slide.level_downsamples):
            if ds <= scale + 1e-6 and ds >= best_ds:
                best, best_ds = i, ds
        return best

    def tile(self, level: int, col: int, row: int) -> np.ndarray:
        x0, y0, w0, h0, tw, th = self.dz.tile_box(level, col, row)
        s = self.dz.scale(level)
        lv = self._best_level(s)
        ds = self.slide.level_downsamples[lv]
        rw = max(1, int(round(w0 / ds)))
        rh = max(1, int(round(h0 / ds)))
        rgb = self.slide.read_region((x0, y0), lv, (rw, rh))
        if (rgb.shape[1], rgb.shape[0]) != (tw, th):
            # INTER_AREA is the right kernel going down: it averages the pixels
            # that are being discarded instead of point-sampling them, so a
            # zoomed-out view shows the mean colour of the tissue rather than
            # whichever pixel happened to land on the sample grid.
            interp = cv2.INTER_AREA if rw >= tw else cv2.INTER_LINEAR
            rgb = cv2.resize(rgb, (tw, th), interpolation=interp)
        return rgb


# ---- the overlay pyramid --------------------------------------------------


class MaskTileSource:
    """DeepZoom tiles of the predicted fat mask, from the precomputed store.

    Two sources behind one interface, chosen by scale:

      full resolution   the per-tile PNGs the precompute pass wrote, on the
                        512 px level-0 grid it ran the model on. Used when the
                        user is zoomed in far enough to see individual
                        droplets, which is where a blocky overlay would be
                        obvious and would undersell the model.
      the ds4 map       one downsampled coverage image for the whole slide,
                        held in memory. Used for everything zoomed out, where
                        assembling thousands of full-resolution tiles per
                        screen would be the bottleneck.

    The seam between them is at scale 4, where both agree to within a pixel.
    """

    def __init__(self, store: str | Path, width: int, height: int,
                 tile_size: int = TILE_SIZE, mask_tile: int = MASK_TILE):
        self.store = Path(store)
        self.dz = DeepZoom(width, height, tile_size)
        self.width, self.height = int(width), int(height)
        # The level-0 grid the mask PNGs were written on. Usually 512, but a
        # slide resampled to the training scale is tiled on its own native
        # spacing -- `precompute.py` records it as `mask_tile` in meta.json,
        # and reading it back is what keeps the two in step.
        self.mask_tile = int(mask_tile)
        self._ds4: np.ndarray | None = None

    # -- sources --

    @property
    def ds4(self) -> np.ndarray:
        if self._ds4 is None:
            p = self.store / "mask" / "ds4.png"
            a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None
            if a is None:
                a = np.zeros((max(1, self.height // 4), max(1, self.width // 4)),
                             np.uint8)
            self._ds4 = a
        return self._ds4

    @lru_cache(maxsize=512)
    def _l0_tile(self, tx: int, ty: int) -> np.ndarray | None:
        p = self.store / "mask" / "l0" / f"{tx:06d}_{ty:06d}.png"
        if not p.exists():
            return None
        return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

    def _read_full(self, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        """Assemble a level-0 region from the per-tile mask PNGs.

        A tile with no fat was never written -- most of a clean slide -- and a
        missing file means zero, not an error. That sparsity is why a whole
        slide's full-resolution mask is a few megabytes rather than 180.
        """
        out = np.zeros((h, w), np.uint8)
        t = self.mask_tile
        for ty in range((y0 // t) * t, y0 + h, t):
            for tx in range((x0 // t) * t, x0 + w, t):
                a = self._l0_tile(tx, ty)
                if a is None:
                    continue
                sx0, sy0 = max(x0, tx), max(y0, ty)
                sx1 = min(x0 + w, tx + a.shape[1])
                sy1 = min(y0 + h, ty + a.shape[0])
                if sx1 <= sx0 or sy1 <= sy0:
                    continue
                out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                    a[sy0 - ty:sy1 - ty, sx0 - tx:sx1 - tx]
        return out

    def _read_ds4(self, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        a = self.ds4
        ah, aw = a.shape
        cx0, cy0 = min(aw, x0 // 4), min(ah, y0 // 4)
        cx1 = min(aw, max(cx0 + 1, math.ceil((x0 + w) / 4)))
        cy1 = min(ah, max(cy0 + 1, math.ceil((y0 + h) / 4)))
        return a[cy0:cy1, cx0:cx1]

    # -- the tile --

    def tile_gray(self, level: int, col: int, row: int) -> np.ndarray:
        """The mask for one tile, as 0..255 coverage at the tile's own size."""
        x0, y0, w0, h0, tw, th = self.dz.tile_box(level, col, row)
        s = self.dz.scale(level)
        src = self._read_full(x0, y0, w0, h0) if s <= 4.0 \
            else self._read_ds4(x0, y0, w0, h0)
        if src.size == 0:
            return np.zeros((th, tw), np.uint8)
        if (src.shape[1], src.shape[0]) != (tw, th):
            # INTER_AREA again, and it is doing real work here: a droplet that
            # is a quarter of a pixel at this zoom becomes a quarter-intensity
            # pixel instead of vanishing or becoming a whole one. That is what
            # makes the zoomed-out overlay show where the fat IS rather than a
            # thresholded caricature of it.
            interp = cv2.INTER_AREA if src.shape[1] >= tw else cv2.INTER_NEAREST
            src = cv2.resize(src, (tw, th), interpolation=interp)
        return src

    def tile_rgba(self, level: int, col: int, row: int,
                  colour: tuple[int, int, int] = (255, 32, 32)) -> np.ndarray:
        """The overlay tile OpenSeadragon draws: colour where fat, clear where not.

        Alpha carries the mask, so opacity is a property of the layer and the
        slider costs nothing. Opaque colour with a transparent background is
        also what makes the on/off toggle read as "the model found exactly
        this" instead of as a colour-balance change.
        """
        g = self.tile_gray(level, col, row)
        h, w = g.shape
        out = np.zeros((h, w, 4), np.uint8)
        out[..., 0] = colour[2]  # OpenCV writes BGRA
        out[..., 1] = colour[1]
        out[..., 2] = colour[0]
        out[..., 3] = g
        return out
