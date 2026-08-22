"""The StarDist layer both candidate features sit on.

Ballooning and inflammation each need nuclei out of a tile, and before this
module existed they each had their own copy of the machinery to get them:
model loading, percentile normalization, padded reads, core relabelling. Six
functions, two of them byte-identical and the rest differing only in wording
-- which meant the two features could drift apart on the parameter that
decides what StarDist sees, and did. Inflammation measured that
`2D_versatile_he` needs `upscale: 2` at this cohort's 20x; ballooning had no
upscale setting at all and ran at 1.

The lru_cache matters more than it looks. Two separately-cached `load_model`s
put two copies of the same weights on the device when one process runs both
features -- on a laptop that is wasted seconds, on an H100 it is wasted VRAM
and a second CUDA context. One cache here, keyed on (name, dir), and the two
features share the instance.

Everything here is pixels-and-labels. Nothing knows what a hepatocyte is.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from .slide import Slide

# The pretrained H&E model both features use. Trained on 40x-ish material, so
# at 20x the small dark nuclei fall below what it resolves -- see `upscale`.
DEFAULT_MODEL = "2D_versatile_he"


def tf_quiet() -> None:
    """Silence TensorFlow's startup banner before it is ever imported."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")


@lru_cache(maxsize=4)
def load_model(model_name: str, model_dir: str | None = None):
    """Load a StarDist model once per process, shared across features.

    Imported lazily so `--help`, config loading, the tissue stage and the whole
    steatosis pipeline never pay TensorFlow's multi-second import cost.

    `model_dir` points at a directory containing the weights, for clusters with
    no outbound network -- `from_pretrained` would otherwise try to download.
    """
    tf_quiet()
    from stardist.models import StarDist2D

    if model_dir:
        p = Path(model_dir)
        return StarDist2D(None, name=p.name, basedir=str(p.parent))
    return StarDist2D.from_pretrained(model_name)


def read_padded(
    slide: Slide, x: int, y: int, size: int, margin: int, level: int = 0
) -> tuple[np.ndarray, int, int]:
    """Read a tile plus `margin` on each side, clamped to the slide.

    Returns the padded RGB and the (col, row) offset of the core tile within
    it. Regions past the slide edge are filled with white -- that is what is
    physically there (glass), and unlike a black fill it cannot read as a dense
    nucleus.

    `x`/`y`/`size` are in `level` pixels; the read is issued in level-0
    coordinates because that is what openslide expects.
    """
    w, h = slide.level_dimensions[level]
    x0, y0 = x - margin, y - margin
    x1, y1 = x + size + margin, y + size + margin

    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(w, x1), min(h, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return np.full((y1 - y0, x1 - x0, 3), 255, dtype=np.uint8), margin, margin

    ds = slide.level_downsamples[level]
    region = slide.read_region(
        (int(cx0 * ds), int(cy0 * ds)), level, (cx1 - cx0, cy1 - cy0)
    )

    out = np.full((y1 - y0, x1 - x0, 3), 255, dtype=np.uint8)
    out[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0] = region
    return out, x - x0, y - y0


def normalize_percentile(
    img: np.ndarray, core: tuple[slice, slice], pmin: float, pmax: float
) -> np.ndarray:
    """Percentile-normalize per channel, with the range taken from the core.

    Taking the range from the core rather than the padded array keeps a tile's
    normalization independent of how much glass its margin happened to catch.
    Two adjacent tiles then normalize the same tissue the same way, which is
    what lets their nuclei be compared at all.
    """
    ref = img[core]
    lo = np.percentile(ref, pmin, axis=(0, 1), keepdims=True)
    hi = np.percentile(ref, pmax, axis=(0, 1), keepdims=True)
    return (img.astype(np.float32) - lo) / np.maximum(hi - lo, 1e-6)


def upscale_image(
    rgb: np.ndarray, core: tuple[slice, slice], upscale: int
) -> tuple[np.ndarray, tuple[slice, slice], int]:
    """Resize a tile for segmentation. Returns (rgb, core, effective upscale).

    `upscale <= 1` is a no-op and returns the inputs untouched, so a caller
    that has not opted in pays nothing and gets byte-identical behaviour.

    The model was trained near 40x. At 20x a 5 um nucleus is only ~10 px
    across, which is below what StarDist resolves -- it finds the large pale
    hepatocyte nuclei and misses the small dark ones entirely, so any
    downstream split into "large" and "small" populations starts with one of
    them already empty. Upscaling costs ~u^2 compute and buys back that
    population. Every measurement taken afterwards must be divided back down:
    areas by u^2, distances by u.
    """
    u = max(1, int(upscale))
    if u == 1:
        return rgb, core, 1
    import cv2

    oy, ox = core[0].start, core[1].start
    ch = core[0].stop - oy
    cw = core[1].stop - ox
    big = cv2.resize(
        rgb, (rgb.shape[1] * u, rgb.shape[0] * u), interpolation=cv2.INTER_LINEAR
    )
    return big, (slice(oy * u, (oy + ch) * u), slice(ox * u, (ox + cw) * u)), u


def predict_instances(
    model,
    norm: np.ndarray,
    prob_threshold: float | None = None,
    nms_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run StarDist. Returns (int32 labels, per-object probability).

    `None` for either threshold means the model's own trained default, which is
    what both features want unless they are being tuned -- passing the default
    explicitly would silently pin it if a future model shipped a different one.
    """
    kwargs = {}
    if prob_threshold is not None:
        kwargs["prob_thresh"] = prob_threshold
    if nms_threshold is not None:
        kwargs["nms_thresh"] = nms_threshold
    labels, details = model.predict_instances(norm, **kwargs)
    return (
        np.asarray(labels, dtype=np.int32),
        np.asarray(details.get("prob", []), dtype=np.float32),
    )


def relabel_core(core_labels: np.ndarray, kept: list[int]) -> np.ndarray:
    """Renumber a core label image to 1..N, matching the measurement arrays.

    Objects dropped by the margin/area/tissue filters become background, so an
    overlay shows exactly what the measurements describe -- an overlay drawn
    from unfiltered labels would show the reviewer objects that no row in the
    CSV accounts for.
    """
    out = np.zeros(core_labels.shape, dtype=np.int32)
    if not kept:
        return out
    lut = np.zeros(int(core_labels.max()) + 1, dtype=np.int32)
    for new, old in enumerate(kept, start=1):
        if old < lut.size:
            lut[old] = new
    return np.take(lut, np.clip(core_labels, 0, lut.size - 1))


def scale_note(
    slide: Slide, level: int, model_name: str = DEFAULT_MODEL, upscale: int = 1
) -> str:
    """The scale the model is actually being run at.

    `2D_versatile_he` was trained near 40x. This cohort scans at 20x, so the
    effective resolution -- level downsample and upscale together -- is worth
    printing on every run rather than assumed, because it is the single setting
    that decides which nuclei exist at all downstream.
    """
    mx, my = slide.mpp_at_level(level)
    u = max(1, int(upscale))
    note = f"level {level}: {mx:.4f} x {my:.4f} um/px"
    if u > 1:
        note += f"; segmenting at {u}x -> {mx / u:.4f} um/px effective"
    return f"{note}  [{model_name}]"
