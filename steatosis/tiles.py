"""Level-0 tiling over the detected tissue.

Tiles are enumerated lazily and read one at a time via openslide's region
interface -- level 0 (35856 x 40087 here) is never held in memory. Tissue
fraction is answered from the low-resolution tissue mask, so background tiles
are skipped without any level-0 read at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .config import TilingConfig
from .slide import Slide
from .tissue import TissueMask


@dataclass(frozen=True)
class Tile:
    """One tile's geometry. Coordinates are level-0 pixels."""

    x: int
    y: int
    size: int
    tissue_fraction: float

    @property
    def name(self) -> str:
        """Filename stem with coordinates encoded."""
        return f"tile_x{self.x:06d}_y{self.y:06d}"

    def read(self, slide: Slide, level: int = 0) -> np.ndarray:
        return slide.read_region((self.x, self.y), level, (self.size, self.size))


def iter_tiles(
    slide: Slide, tissue: TissueMask, cfg: TilingConfig, limit: int = 0
) -> Iterator[Tile]:
    """Yield tiles whose tissue fraction meets the threshold, in raster order.

    `limit` > 0 stops after that many *accepted* tiles, which is what makes
    --limit useful for iteration: it bounds work, not just output.
    """
    w, h = slide.dimensions
    size, stride = cfg.tile_size, cfg.stride
    if stride <= 0:
        raise ValueError(f"tiling.stride must be > 0, got {stride}")

    emitted = 0
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            frac = tissue.tissue_fraction(x, y, size, size)
            if frac < cfg.min_tissue_fraction:
                continue
            yield Tile(x=x, y=y, size=size, tissue_fraction=frac)
            emitted += 1
            if limit and emitted >= limit:
                return


def count_candidate_tiles(
    slide: Slide, tissue: TissueMask, cfg: TilingConfig
) -> int:
    """Total tiles passing the tissue filter, without reading any pixels."""
    w, h = slide.dimensions
    size, stride = cfg.tile_size, cfg.stride
    n = 0
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            if tissue.tissue_fraction(x, y, size, size) >= cfg.min_tissue_fraction:
                n += 1
    return n
