"""Whole-slide reading.

Thin wrapper over openslide that keeps the physical scale (microns per pixel)
attached to the image, so downstream filters can work in microns instead of
pixels. Nothing here ever reads level 0 in full -- only region requests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openslide


class Slide:
    """A whole-slide image with its pyramid and physical scale."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.osr = openslide.OpenSlide(str(self.path))

        mpp_x = self.osr.properties.get(openslide.PROPERTY_NAME_MPP_X)
        mpp_y = self.osr.properties.get(openslide.PROPERTY_NAME_MPP_Y)
        if mpp_x is None or mpp_y is None:
            raise ValueError(
                f"{self.path.name}: no microns-per-pixel in slide metadata; "
                "area filters in microns cannot be computed"
            )
        self.mpp_x = float(mpp_x)
        self.mpp_y = float(mpp_y)

    # ---- basic properties ----------------------------------------------

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def dimensions(self) -> tuple[int, int]:
        """Level-0 (width, height)."""
        return self.osr.dimensions

    @property
    def level_count(self) -> int:
        return self.osr.level_count

    @property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        return self.osr.level_dimensions

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        return self.osr.level_downsamples

    @property
    def objective_power(self) -> str | None:
        return self.osr.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)

    @property
    def vendor(self) -> str | None:
        return self.osr.properties.get(openslide.PROPERTY_NAME_VENDOR)

    def mpp_at_level(self, level: int) -> tuple[float, float]:
        """Microns per pixel at a given pyramid level."""
        ds = self.osr.level_downsamples[level]
        return self.mpp_x * ds, self.mpp_y * ds

    def um2_per_pixel(self, level: int) -> float:
        """Area of one pixel in square microns at the given level."""
        mx, my = self.mpp_at_level(level)
        return mx * my

    def resolve_level(self, level: int) -> int:
        """Clamp a requested level to what the slide actually has."""
        if level < 0:
            raise ValueError(f"level must be >= 0, got {level}")
        return min(level, self.osr.level_count - 1)

    # ---- pixel access ---------------------------------------------------

    def read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> np.ndarray:
        """Read a region as an RGB uint8 array.

        `location` is in LEVEL 0 coordinates (openslide's convention), `size`
        is in pixels at `level`. Alpha is dropped -- Aperio JPEG slides are
        opaque, and openslide fills out-of-bounds requests with black.
        """
        img = self.osr.read_region(location, level, size)
        return np.asarray(img.convert("RGB"))

    def read_level(self, level: int) -> np.ndarray:
        """Read an entire pyramid level as RGB. Only safe for small levels --
        callers should use level >= 1 on 35k x 40k slides."""
        level = self.resolve_level(level)
        w, h = self.osr.level_dimensions[level]
        return self.read_region((0, 0), level, (w, h))

    def thumbnail(self, max_dim: int = 2000) -> np.ndarray:
        """A downscaled RGB overview of the whole slide."""
        w, h = self.osr.dimensions
        scale = min(max_dim / w, max_dim / h, 1.0)
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return np.asarray(self.osr.get_thumbnail(size).convert("RGB"))

    # ---- lifecycle ------------------------------------------------------

    def close(self) -> None:
        self.osr.close()

    def __enter__(self) -> "Slide":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def describe(self) -> str:
        w, h = self.dimensions
        lines = [
            f"slide:            {self.name}",
            f"vendor:           {self.vendor}",
            f"level 0:          {w} x {h}",
            f"levels:           {self.level_count}",
            f"level dims:       {self.level_dimensions}",
            f"downsamples:      {tuple(round(d, 3) for d in self.level_downsamples)}",
            f"mpp:              {self.mpp_x:.4f} x {self.mpp_y:.4f} um/px",
            f"objective:        {self.objective_power}x",
        ]
        return "\n".join(lines)
