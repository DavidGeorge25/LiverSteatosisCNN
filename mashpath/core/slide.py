"""Whole-slide reading.

Keeps the physical scale (microns per pixel) attached to the image, so every
downstream filter works in microns instead of pixels. Nothing here ever reads
level 0 in full -- only region requests.

Two backends behind one class:

  openslide   SVS, NDPI, MRXS, SCN, and tiled pyramidal TIFF. Lazy region
              reads at any level.
  tifffile    TIFFs openslide will not open (untiled, odd photometric
              layouts, some OME-TIFF). Region reads are lazy only when `zarr`
              is installed; without it a level is decoded whole, which is
              refused above `slide.max_in_memory_megapixels` rather than
              quietly exhausting RAM.

MPP policy: a slide's own embedded scale wins. When the file carries none --
routine for plain TIFF -- the reader raises and names the config keys to set.
It never infers MPP from objective power or image size, because a wrong guess
silently rescales every area threshold in the project and the numbers would
still look plausible.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import openslide

from .config import SlideConfig

# Formats openslide handles natively, plus TIFF variants we can fall back on.
SLIDE_EXTENSIONS = (
    ".svs", ".tif", ".tiff", ".ndpi", ".vms", ".vmu",
    ".scn", ".mrxs", ".svslide", ".bif", ".ome.tif", ".ome.tiff",
)


class SlideFormatError(RuntimeError):
    """The file cannot be opened by any available backend."""


class MissingScaleError(ValueError):
    """The slide carries no microns-per-pixel and none was configured."""


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------


class _OpenSlideBackend:
    name = "openslide"

    def __init__(self, path: Path):
        self.osr = openslide.OpenSlide(str(path))

    @property
    def dimensions(self) -> tuple[int, int]:
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
    def vendor(self) -> str | None:
        return self.osr.properties.get(openslide.PROPERTY_NAME_VENDOR)

    @property
    def objective_power(self) -> str | None:
        return self.osr.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)

    def embedded_mpp(self) -> tuple[float | None, float | None]:
        x = self.osr.properties.get(openslide.PROPERTY_NAME_MPP_X)
        y = self.osr.properties.get(openslide.PROPERTY_NAME_MPP_Y)
        return (float(x) if x else None, float(y) if y else None)

    def read_region(self, location, level, size) -> np.ndarray:
        img = self.osr.read_region(location, level, size)
        return np.asarray(img.convert("RGB"))

    def thumbnail(self, size) -> np.ndarray:
        return np.asarray(self.osr.get_thumbnail(size).convert("RGB"))

    def close(self) -> None:
        self.osr.close()


class _TiffBackend:
    """tifffile fallback for TIFFs openslide rejects.

    A pyramidal TIFF exposes its levels as `series.levels`; a flat TIFF has a
    single level, and downsampling requests are served by cropping level 0.
    """

    name = "tifffile"

    def __init__(self, path: Path, max_megapixels: float):
        import tifffile

        self._tf = tifffile.TiffFile(str(path))
        self._max_mp = max_megapixels
        if not self._tf.series:
            raise SlideFormatError(f"{path.name}: TIFF contains no image series")
        self._series = self._tf.series[0]
        self._levels = list(getattr(self._series, "levels", []) or [self._series])

        self._hw: list[tuple[int, int]] = []
        for lv in self._levels:
            axes = getattr(lv, "axes", "YX")
            shape = lv.shape
            try:
                h = shape[axes.index("Y")]
                w = shape[axes.index("X")]
            except ValueError as exc:  # no Y/X axis label to go on
                raise SlideFormatError(
                    f"{path.name}: cannot locate Y/X axes in TIFF series "
                    f"(axes={axes!r}, shape={shape})"
                ) from exc
            self._hw.append((int(h), int(w)))

        # Largest level first, as openslide orders them.
        order = sorted(range(len(self._levels)), key=lambda i: -self._hw[i][1])
        self._levels = [self._levels[i] for i in order]
        self._hw = [self._hw[i] for i in order]

        self._zarr: Any = None
        try:  # lazy region reads when zarr is available
            import zarr  # noqa: F401

            store = self._series.aszarr()
            self._zarr = zarr.open(store, mode="r")
        except Exception:
            self._zarr = None
        self._cache: dict[int, np.ndarray] = {}

    @property
    def dimensions(self) -> tuple[int, int]:
        h, w = self._hw[0]
        return (w, h)

    @property
    def level_count(self) -> int:
        return len(self._levels)

    @property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        return tuple((w, h) for h, w in self._hw)

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        w0 = self._hw[0][1]
        return tuple(w0 / w for _, w in self._hw)

    @property
    def vendor(self) -> str | None:
        return "generic-tiff"

    @property
    def objective_power(self) -> str | None:
        return None

    def embedded_mpp(self) -> tuple[float | None, float | None]:
        """Physical scale from OME metadata or the TIFF resolution tags.

        Both are genuine embedded scale, not inference. A ResolutionUnit of
        "none" means the numbers are an aspect ratio, not a physical size, so
        it is treated as absent.
        """
        ome = getattr(self._tf, "ome_metadata", None)
        if ome:
            x = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', ome)
            y = re.search(r'PhysicalSizeY="([0-9.eE+-]+)"', ome)
            if x and y:
                unit = re.search(r'PhysicalSizeXUnit="([^"]+)"', ome)
                scale = {"nm": 1e-3, "µm": 1.0, "um": 1.0, "mm": 1e3}.get(
                    unit.group(1) if unit else "µm", 1.0
                )
                return float(x.group(1)) * scale, float(y.group(1)) * scale

        page = self._levels[0].pages[0]
        tags = page.tags
        unit = tags.get("ResolutionUnit")
        unit_val = getattr(unit, "value", None)
        # 1 = no absolute unit, 2 = inch, 3 = centimetre.
        per_unit_um = {2: 25_400.0, 3: 10_000.0}.get(int(unit_val or 1))
        if per_unit_um is None:
            return (None, None)

        def _mpp(tag_name: str) -> float | None:
            tag = tags.get(tag_name)
            if tag is None or tag.value is None:
                return None
            v = tag.value
            res = float(v[0]) / float(v[1]) if isinstance(v, tuple) else float(v)
            return per_unit_um / res if res > 0 else None

        return _mpp("XResolution"), _mpp("YResolution")

    def _level_array(self, level: int) -> Any:
        if self._zarr is not None:
            return self._zarr[level] if hasattr(self._zarr, "__getitem__") and \
                not isinstance(self._zarr, np.ndarray) else self._zarr
        if level not in self._cache:
            h, w = self._hw[level]
            mp = h * w / 1e6
            if mp > self._max_mp:
                raise SlideFormatError(
                    f"level {level} is {w}x{h} ({mp:.0f} MP), above the "
                    f"{self._max_mp:.0f} MP limit for the tifffile backend. "
                    "Install `zarr` for lazy region reads, raise "
                    "slide.max_in_memory_megapixels, or convert the slide to a "
                    "tiled pyramidal TIFF that openslide can read directly."
                )
            self._cache.clear()  # one level at a time
            self._cache[level] = np.asarray(self._levels[level].asarray())
        return self._cache[level]

    def read_region(self, location, level, size) -> np.ndarray:
        x0, y0 = location  # level-0 coordinates, openslide's convention
        w, h = size
        ds = self.level_downsamples[level]
        lx, ly = int(round(x0 / ds)), int(round(y0 / ds))

        arr = self._level_array(level)
        lh, lw = self._hw[level]
        out = np.zeros((h, w, 3), dtype=np.uint8)  # out-of-bounds reads as black

        sx0, sy0 = max(0, lx), max(0, ly)
        sx1, sy1 = min(lw, lx + w), min(lh, ly + h)
        if sx1 <= sx0 or sy1 <= sy0:
            return out

        crop = np.asarray(arr[sy0:sy1, sx0:sx1])
        if crop.ndim == 2:
            crop = np.dstack([crop] * 3)
        elif crop.shape[2] > 3:
            crop = crop[:, :, :3]
        if crop.dtype != np.uint8:  # 16-bit or float TIFF
            crop = _to_uint8(crop)
        out[sy0 - ly : sy1 - ly, sx0 - lx : sx1 - lx] = crop
        return out

    def thumbnail(self, size) -> np.ndarray:
        import cv2

        level = self.level_count - 1
        arr = np.asarray(self._level_array(level))
        if arr.ndim == 2:
            arr = np.dstack([arr] * 3)
        elif arr.shape[2] > 3:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = _to_uint8(arr)
        return cv2.resize(arr, size, interpolation=cv2.INTER_AREA)

    def close(self) -> None:
        self._cache.clear()
        self._tf.close()


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    hi = float(a.max()) or 1.0
    return np.clip(a / hi * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# public slide
# --------------------------------------------------------------------------


class Slide:
    """A whole-slide image with its pyramid and physical scale."""

    def __init__(self, path: str | Path, cfg: SlideConfig | None = None):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.cfg = cfg or SlideConfig()
        self._backend = self._open_backend()
        self.mpp_x, self.mpp_y, self.mpp_source = self._resolve_mpp()

    def _open_backend(self):
        try:
            return _OpenSlideBackend(self.path)
        except Exception as openslide_exc:
            suffix = self.path.name.lower()
            if not suffix.endswith((".tif", ".tiff")):
                raise SlideFormatError(
                    f"{self.path.name}: openslide cannot open this file "
                    f"({type(openslide_exc).__name__}: {openslide_exc})"
                ) from openslide_exc
            try:
                return _TiffBackend(self.path, self.cfg.max_in_memory_megapixels)
            except Exception as tiff_exc:
                raise SlideFormatError(
                    f"{self.path.name}: not readable by openslide "
                    f"({openslide_exc}) or tifffile ({tiff_exc})"
                ) from tiff_exc

    def _resolve_mpp(self) -> tuple[float, float, str]:
        emb_x, emb_y = self._backend.embedded_mpp()
        cfg_x, cfg_y = self.cfg.mpp_x, self.cfg.mpp_y
        configured = cfg_x is not None and cfg_y is not None

        if configured and (self.cfg.override_mpp or emb_x is None or emb_y is None):
            if emb_x and emb_y and self.cfg.override_mpp:
                drift = max(abs(cfg_x - emb_x) / emb_x, abs(cfg_y - emb_y) / emb_y)
                if drift > 0.01:
                    print(
                        f"  WARNING {self.path.name}: slide.override_mpp is on and "
                        f"the configured scale ({cfg_x:.4f} x {cfg_y:.4f} um/px) "
                        f"differs from the embedded scale "
                        f"({emb_x:.4f} x {emb_y:.4f} um/px) by {drift*100:.1f}%; "
                        "every micron threshold now means a different size."
                    )
            return float(cfg_x), float(cfg_y), "config"

        if emb_x is not None and emb_y is not None:
            return float(emb_x), float(emb_y), "embedded"

        raise MissingScaleError(
            f"{self.path.name}: no microns-per-pixel in the file "
            f"({self._backend.name} backend) and none configured. Every filter "
            "in this pipeline is specified in microns, so the scale cannot be "
            "guessed. Set it explicitly:\n"
            "    slide:\n"
            "      mpp_x: 0.4953\n"
            "      mpp_y: 0.4953\n"
            "or pass --mpp 0.4953 on the command line."
        )

    # ---- basic properties ----------------------------------------------

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def backend(self) -> str:
        return self._backend.name

    @property
    def dimensions(self) -> tuple[int, int]:
        """Level-0 (width, height)."""
        return self._backend.dimensions

    @property
    def level_count(self) -> int:
        return self._backend.level_count

    @property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        return self._backend.level_dimensions

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        return self._backend.level_downsamples

    @property
    def objective_power(self) -> str | None:
        return self._backend.objective_power

    @property
    def vendor(self) -> str | None:
        return self._backend.vendor

    def mpp_at_level(self, level: int) -> tuple[float, float]:
        """Microns per pixel at a given pyramid level."""
        ds = self.level_downsamples[level]
        return self.mpp_x * ds, self.mpp_y * ds

    def um2_per_pixel(self, level: int) -> float:
        """Area of one pixel in square microns at the given level."""
        mx, my = self.mpp_at_level(level)
        return mx * my

    def um_to_px(self, microns: float, level: int = 0) -> int:
        """Convert a length in microns to whole pixels at `level`.

        The one place a micron threshold becomes a pixel count, so changing
        pyramid level can never silently change what a filter means.
        """
        mx, my = self.mpp_at_level(level)
        return max(1, int(round(microns / ((mx + my) / 2.0))))

    def resolve_level(self, level: int) -> int:
        """Clamp a requested level to what the slide actually has."""
        if level < 0:
            raise ValueError(f"level must be >= 0, got {level}")
        return min(level, self.level_count - 1)

    # ---- pixel access ---------------------------------------------------

    def read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> np.ndarray:
        """Read a region as an RGB uint8 array.

        `location` is in LEVEL 0 coordinates (openslide's convention), `size`
        is in pixels at `level`. Alpha is dropped -- Aperio JPEG slides are
        opaque, and out-of-bounds requests are filled with black.
        """
        return self._backend.read_region(location, level, size)

    def read_level(self, level: int) -> np.ndarray:
        """Read an entire pyramid level as RGB. Only safe for small levels --
        callers should use level >= 1 on 35k x 40k slides."""
        level = self.resolve_level(level)
        w, h = self.level_dimensions[level]
        return self.read_region((0, 0), level, (w, h))

    def thumbnail(self, max_dim: int = 2000) -> np.ndarray:
        """A downscaled RGB overview of the whole slide."""
        w, h = self.dimensions
        scale = min(max_dim / w, max_dim / h, 1.0)
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return self._backend.thumbnail(size)

    # ---- lifecycle ------------------------------------------------------

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> "Slide":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def describe(self) -> str:
        w, h = self.dimensions
        lines = [
            f"slide:            {self.name}",
            f"reader:           {self.backend}",
            f"vendor:           {self.vendor}",
            f"level 0:          {w} x {h}",
            f"levels:           {self.level_count}",
            f"level dims:       {self.level_dimensions}",
            f"downsamples:      {tuple(round(d, 3) for d in self.level_downsamples)}",
            f"mpp:              {self.mpp_x:.4f} x {self.mpp_y:.4f} um/px "
            f"({self.mpp_source})",
            f"objective:        {self.objective_power}x",
        ]
        return "\n".join(lines)


def open_slide(path: str | Path, cfg: SlideConfig | None = None) -> Slide:
    """Open a slide with whichever backend can read it."""
    return Slide(path, cfg)


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
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r"(\d+)", p.stem)]

    return sorted(found, key=key)
