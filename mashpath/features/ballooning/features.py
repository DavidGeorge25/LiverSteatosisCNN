"""Per-cell morphometry, cytoplasm intensity and cytoplasm texture.

Everything here is measured on the PADDED tile, using the uncropped label
images, so a cell whose nucleus sits near a tile seam is still measured over
its whole body instead of the part that happens to fall inside the core.

Two conventions worth stating once:

*The cytoplasm compartment is the cell minus its nucleus.* Every intensity and
texture number below is computed over that compartment alone. Including the
nucleus would put a dense haematoxylin blob into every measurement and swamp
the thing being measured -- a ballooned cell and a normal one differ in their
cytoplasm, not in their chromatin.

*Texture maps are computed once per tile, not once per cell.* Local standard
deviation and local entropy are sliding-window operations; running them per
cell would recompute the same pixels for every overlapping window and cost
tens of times more for identical numbers. Only the GLCM is genuinely per cell,
because it needs that cell's own pixels and nothing else's.

The texture features exist to capture "rarefied": ballooned cytoplasm is wispy
and heterogeneous -- cleared-out strands with pale gaps -- where normal
hepatocyte cytoplasm is smoothly and evenly eosinophilic. So the signal is a
LOSS of smoothness, which is why local std, entropy and GLCM contrast all move
up together while homogeneity moves down.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.color import rgb2hed
from skimage.feature import graycomatrix, graycoprops
from skimage.filters.rank import entropy as local_entropy
from skimage.morphology import disk

from .config import FeatureConfig

# Channels every cell is measured over, in order. Kept as one stack so a cell's
# pixels are gathered once and reduced in a single pass.
CHANNELS = (
    "r", "g", "b",          # raw RGB, for the "each channel" requirement
    "gray",                 # mean of RGB, 0-255; higher = paler
    "hematoxylin_od",       # colour-deconvolved; chromatin
    "eosin_od",             # colour-deconvolved; the stain rarefaction removes
    "local_std",            # texture: within-window spread
    "entropy",              # texture: within-window information
)


def build_channels(
    rgb: np.ndarray, cfg: FeatureConfig, mpp: float
) -> np.ndarray:
    """Stack every per-pixel channel a cell is measured over. Shape (H, W, C).

    Computed once for the whole tile; cells then index into it.
    """
    f = rgb.astype(np.float32)
    gray = f.mean(axis=2)

    if cfg.use_color_deconvolution:
        hed = rgb2hed(f / 255.0)
        hema, eos = hed[..., 0].astype(np.float32), hed[..., 1].astype(np.float32)
    else:
        hema = eos = np.zeros_like(gray)

    win = max(3, int(round(cfg.texture_window_um / mpp)) | 1)  # odd, >= 3
    # E[x^2] - E[x]^2, clipped: float error can push a flat window slightly
    # negative and sqrt would produce nan.
    mean = uniform_filter(gray, size=win)
    mean_sq = uniform_filter(gray * gray, size=win)
    lstd = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))

    ent = local_entropy(
        np.clip(gray, 0, 255).astype(np.uint8), disk(max(1, win // 2))
    ).astype(np.float32)

    return np.dstack([f[..., 0], f[..., 1], f[..., 2], gray, hema, eos, lstd, ent])


def measure_cell(
    channels: np.ndarray,
    cyto_mask: np.ndarray,
    gray_patch: np.ndarray,
    cfg: FeatureConfig,
) -> dict[str, float]:
    """Mean/std of every channel over the cytoplasm, plus Haralick features.

    `cyto_mask`, `gray_patch` and `channels` are all already cropped to the
    cell's bounding box.
    """
    out: dict[str, float] = {}
    vals = channels[cyto_mask]  # (n_px, C)
    if vals.size == 0:
        return {f"cyto_{c}_{s}": np.nan for c in CHANNELS for s in ("mean", "std")}

    means = vals.mean(axis=0)
    stds = vals.std(axis=0)
    for i, name in enumerate(CHANNELS):
        out[f"cyto_{name}_mean"] = float(means[i])
        out[f"cyto_{name}_std"] = float(stds[i])

    out.update(_haralick(gray_patch, cyto_mask, cfg))
    return out


def _haralick(
    gray: np.ndarray, mask: np.ndarray, cfg: FeatureConfig
) -> dict[str, float]:
    """GLCM properties over the cytoplasm only.

    `graycomatrix` has no mask argument, so the standard trick applies: quantize
    the cytoplasm into levels 1..L, leave 0 for everything outside it, then
    delete row 0 and column 0 of the matrix. Pairs involving an outside pixel
    are discarded instead of being counted as a real grey-level transition,
    which is what would otherwise put a huge false "contrast" on every cell
    boundary -- the exact feature ballooning is being scored on.
    """
    levels = int(cfg.glcm_levels)
    keys = [
        f"cyto_glcm_{p}_d{d}" for d in cfg.glcm_distances for p in cfg.glcm_properties
    ]
    if mask.sum() < 16:
        return dict.fromkeys(keys, np.nan)

    vals = gray[mask]
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        return dict.fromkeys(keys, np.nan)

    q = np.zeros(gray.shape, dtype=np.uint8)
    scaled = (vals - lo) / (hi - lo) * (levels - 1)
    q[mask] = scaled.astype(np.uint8) + 1  # 1..levels; 0 means "not cytoplasm"

    out: dict[str, float] = {}
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    for d in cfg.glcm_distances:
        if min(gray.shape) <= d:
            for p in cfg.glcm_properties:
                out[f"cyto_glcm_{p}_d{d}"] = np.nan
            continue
        glcm = graycomatrix(
            q, distances=[int(d)], angles=angles,
            levels=levels + 1, symmetric=True, normed=False,
        ).astype(np.float64)
        glcm = glcm[1:, 1:, :, :]  # drop every pair touching a non-cytoplasm pixel
        total = glcm.sum(axis=(0, 1), keepdims=True)
        if not np.all(total > 0):
            for p in cfg.glcm_properties:
                out[f"cyto_glcm_{p}_d{d}"] = np.nan
            continue
        glcm /= total
        for p in cfg.glcm_properties:
            # Average over angles: ballooning has no orientation, so a
            # direction-dependent number would just add noise.
            out[f"cyto_glcm_{p}_d{d}"] = float(graycoprops(glcm, p).mean())
    return out


def feature_names(cfg: FeatureConfig) -> list[str]:
    """Every per-cell measurement name, in a stable order."""
    names = [f"cyto_{c}_{s}" for c in CHANNELS for s in ("mean", "std")]
    names += [
        f"cyto_glcm_{p}_d{d}"
        for d in cfg.glcm_distances
        for p in cfg.glcm_properties
    ]
    return names
