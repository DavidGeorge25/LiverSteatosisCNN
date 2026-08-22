"""Stain normalization: Macenko and Reinhard. Configurable, off by default.

OFF BY DEFAULT, deliberately. Normalization is usually presented as free — it
can only help, so turn it on — and that is exactly the assumption this project
cannot afford to make. Staining batch is perfectly aliased with experiment here,
so if normalization is on from the first run there is no measurement of what it
did: a model that generalises across batches and a model that does not will both
be reported through the same normalized pixels. The switch exists so the effect
can be MEASURED — same labels, same architecture, normalization on and off, and
the held-out-batch metric in `evaluate.py` says which was better.

There is also a specific reason to distrust it here. Macenko estimates the stain
vectors from the image it is given; on a section with heavy macrovesicular
steatosis a large fraction of the tissue area is white droplet void, which
carries no stain and is discarded by the optical-density floor. The estimate is
then made from less tissue on exactly the slides whose biology we care about.
That is not a reason to avoid it, it is a reason to look at the numbers.

Colour augmentation (`augment.py`) is the other half of this and is often the
stronger half: normalization tries to make every batch look the same, whereas
augmentation makes the model stop relying on how a batch looks. They compose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Ruifrok-style H&E reference vectors in optical density, and the reference
# 99th-percentile concentrations they go with. Columns are haematoxylin, eosin.
DEFAULT_STAIN_MATRIX = np.array([[0.5626, 0.2159],
                                 [0.7201, 0.8012],
                                 [0.4062, 0.5581]])
DEFAULT_MAX_C = np.array([1.9705, 1.0308])
IO = 240.0          # assumed transmitted light intensity


def rgb_to_od(rgb: np.ndarray, io: float = IO) -> np.ndarray:
    """(..., 3) uint8 RGB -> optical density. +1 keeps log finite at pure black."""
    x = np.asarray(rgb, dtype=np.float64)
    return -np.log((x + 1.0) / io)


def od_to_rgb(od: np.ndarray, io: float = IO) -> np.ndarray:
    return np.clip(io * np.exp(-od), 0, 255).astype(np.uint8)


def macenko_stain_matrix(rgb: np.ndarray, io: float = IO, beta: float = 0.15,
                         alpha: float = 1.0) -> np.ndarray:
    """Estimate the 3x2 H&E stain matrix from one image.

    Raises rather than guessing when there is too little stained tissue to
    estimate from -- a silent fallback to the reference matrix would make a
    mostly-empty tile look like it had been normalized when it had not.
    """
    od = rgb_to_od(rgb, io).reshape(-1, 3)
    tissue = od[~np.any(od < beta, axis=1)]
    if len(tissue) < 100:
        raise ValueError(
            f"only {len(tissue)} pixels above the optical-density floor "
            f"(beta={beta}); not enough stained tissue to estimate stain vectors")
    _, vecs = np.linalg.eigh(np.cov(tissue.T))
    plane = vecs[:, 1:3]                       # the two leading directions
    proj = tissue @ plane
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, alpha), np.percentile(phi, 100 - alpha)
    v_lo = plane @ np.array([np.cos(lo), np.sin(lo)])
    v_hi = plane @ np.array([np.cos(hi), np.sin(hi)])
    # Eigenvectors have arbitrary sign, and optical density cannot be negative,
    # so each vector is FLIPPED into the positive orthant rather than passed
    # through abs(): abs() would mirror individual components independently and
    # silently invent a stain vector pointing somewhere neither eigenvector did.
    v_lo = -v_lo if v_lo.sum() < 0 else v_lo
    v_hi = -v_hi if v_hi.sum() < 0 else v_hi
    # Haematoxylin is the one with the larger red-channel OD.
    he = np.stack([v_lo, v_hi], axis=1) if v_lo[0] > v_hi[0] \
        else np.stack([v_hi, v_lo], axis=1)
    he = np.clip(he, 0.0, None)
    norms = np.linalg.norm(he, axis=0, keepdims=True)
    if not np.all(norms > 1e-8):
        raise ValueError("degenerate stain vectors; the optical-density cloud "
                         "has no two distinguishable directions")
    return he / norms


def stain_concentrations(rgb: np.ndarray, stain_matrix: np.ndarray,
                         io: float = IO) -> np.ndarray:
    """(2, N) concentration of each stain at each pixel."""
    od = rgb_to_od(rgb, io).reshape(-1, 3)
    c, *_ = np.linalg.lstsq(stain_matrix, od.T, rcond=None)
    return c


@dataclass
class MacenkoNormalizer:
    """Match an image's stain vectors and intensities to a reference."""

    stain_matrix: np.ndarray = field(default_factory=lambda: DEFAULT_STAIN_MATRIX.copy())
    max_c: np.ndarray = field(default_factory=lambda: DEFAULT_MAX_C.copy())
    io: float = IO
    beta: float = 0.15
    alpha: float = 1.0

    def fit(self, rgb: np.ndarray) -> "MacenkoNormalizer":
        """Take the reference from a real image instead of the literature.

        Worth doing when one batch is the intended target appearance; the
        literature vectors are a compromise across scanners and stains and are
        not obviously closer to this lab's H&E than one of its own batches is.
        """
        self.stain_matrix = macenko_stain_matrix(rgb, self.io, self.beta, self.alpha)
        c = stain_concentrations(rgb, self.stain_matrix, self.io)
        self.max_c = np.percentile(c, 99, axis=1)
        return self

    def transform(self, rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb)
        h, w = rgb.shape[:2]
        src = macenko_stain_matrix(rgb, self.io, self.beta, self.alpha)
        c = stain_concentrations(rgb, src, self.io)
        max_c = np.percentile(c, 99, axis=1)
        max_c = np.where(max_c <= 0, 1.0, max_c)
        c = c * (self.max_c / max_c)[:, None]
        return od_to_rgb((self.stain_matrix @ c).T, self.io).reshape(h, w, 3)


@dataclass
class ReinhardNormalizer:
    """Match per-channel mean and standard deviation in LAB.

    Cruder than Macenko and more robust for it: it never has to estimate a
    stain vector, so a tile that is 60% fat void does not throw it. When the two
    disagree on a steatotic slide, that is usually why.
    """

    mean: np.ndarray = field(default_factory=lambda: np.array([65.0, 15.0, -5.0]))
    std: np.ndarray = field(default_factory=lambda: np.array([15.0, 8.0, 5.0]))

    @staticmethod
    def _lab(rgb: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB
                            ).astype(np.float64)

    def fit(self, rgb: np.ndarray) -> "ReinhardNormalizer":
        lab = self._lab(rgb).reshape(-1, 3)
        self.mean, self.std = lab.mean(0), lab.std(0)
        return self

    def transform(self, rgb: np.ndarray) -> np.ndarray:
        import cv2
        lab = self._lab(rgb)
        m, s = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0)
        s = np.where(s < 1e-6, 1.0, s)
        out = (lab - m) / s * self.std + self.mean
        out = np.clip(out, [0, 0, 0], [255, 255, 255]).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


class NullNormalizer:
    """The default. Returns the image untouched, so `if normalizer:` is never
    the thing that decides whether normalization happened."""

    def fit(self, rgb: np.ndarray) -> "NullNormalizer":
        return self

    def transform(self, rgb: np.ndarray) -> np.ndarray:
        return np.asarray(rgb)


METHODS = {"none": NullNormalizer, "macenko": MacenkoNormalizer,
           "reinhard": ReinhardNormalizer}


def make_normalizer(method: str = "none", reference: np.ndarray | None = None):
    """`method` is a name from METHODS; unknown names raise rather than no-op."""
    if method not in METHODS:
        raise ValueError(f"unknown stain normalization method {method!r}; "
                         f"expected one of {sorted(METHODS)}")
    norm = METHODS[method]()
    if reference is not None:
        norm.fit(reference)
    return norm
