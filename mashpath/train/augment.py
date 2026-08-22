"""Colour augmentation, aggressively.

WHY THIS RATHER THAN NORMALIZATION ALONE. Normalization tries to make every
batch look the same and then asks the model to trust that it worked. Augmentation
does the opposite: it shows the model the same tissue under colours no scanner
ever produced, so that colour stops being a usable feature at all. When batch is
confounded with cohort — as it is here, one-to-one — that difference matters,
because a normalizer that leaves any residual batch signature leaves a shortcut
the model is free to take, whereas an augmenter that destroys colour information
removes the shortcut whether or not any residue was left.

They are not alternatives. Normalize if it measures better (see `stain.py` and
`evaluate.py`), augment either way.

HOW AGGRESSIVE. The defaults here are stronger than typical ImageNet recipes and
that is intentional: the nine staining batches in this collection differ from
each other by more than a mild jitter spans, so a jitter narrower than the real
between-batch variation trains the model to be invariant over a range that does
not include the range it will meet. The morphology this task depends on — a cell
enlarged relative to its neighbours, rounded, with rarefied cytoplasm — survives
strong colour distortion because it is a matter of size, shape and texture. The
one thing to keep an eye on is that haematoxylin-vs-eosin contrast never inverts
so far that a nucleus stops reading as a nucleus; `stain_jitter` is bounded for
that reason.

GEOMETRY IS NOT HERE. Flips and 90-degree rotations are free and correct for
histology (there is no canonical orientation), but they are not part of the
batch problem and belong wherever the training loop lives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stain import DEFAULT_STAIN_MATRIX, od_to_rgb, rgb_to_od, stain_concentrations


@dataclass
class ColourJitter:
    """Hue, saturation, brightness and contrast, in HSV then in RGB.

    Ranges are FULL widths: `hue=0.10` means the hue is shifted by up to
    +/-0.05 turns. Every parameter can be set to 0 to disable that term, which
    is how an ablation is run.
    """

    hue: float = 0.10           # +/- 18 degrees; batches differ by about this
    saturation: float = 0.60    # +/- 30%
    brightness: float = 0.50
    contrast: float = 0.50
    gamma: float = 0.40
    probability: float = 0.9    # some tiles pass through untouched

    def __call__(self, rgb: np.ndarray, rng: np.random.Generator | None = None
                 ) -> np.ndarray:
        import cv2
        rng = rng or np.random.default_rng()
        img = np.asarray(rgb, dtype=np.uint8)
        if rng.random() > self.probability:
            return img

        def spread(width: float) -> float:
            return float(rng.uniform(-width / 2, width / 2))

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        if self.hue:
            # OpenCV hue is 0..179, so a "turn" is 180 units.
            hsv[..., 0] = (hsv[..., 0] + spread(self.hue) * 180.0) % 180.0
        if self.saturation:
            hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + spread(self.saturation)), 0, 255)
        if self.brightness:
            hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + spread(self.brightness)), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        if self.contrast:
            mean = out.mean()
            out = (out - mean) * (1.0 + spread(self.contrast)) + mean
        if self.gamma:
            g = float(np.exp(spread(self.gamma)))
            out = 255.0 * np.power(np.clip(out, 0, 255) / 255.0, g)
        return np.clip(out, 0, 255).astype(np.uint8)


@dataclass
class StainJitter:
    """Perturb haematoxylin and eosin concentrations independently.

    The pathology-specific version of colour jitter: instead of moving the whole
    image through HSV, it decomposes into stain concentrations and scales and
    shifts each stain on its own. That reproduces what actually varies between
    staining runs — how much haematoxylin took, how much eosin took — rather
    than a generic colour transform that also produces colours no H&E slide can
    have.

    Bounded so the two stains cannot swap dominance: a nucleus that stops being
    the darkest thing in the field is no longer the object the label is about.
    """

    sigma: float = 0.25         # multiplicative, per stain
    bias: float = 0.05          # additive, per stain, in OD units
    probability: float = 0.9

    def __call__(self, rgb: np.ndarray, rng: np.random.Generator | None = None
                 ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        img = np.asarray(rgb, dtype=np.uint8)
        if rng.random() > self.probability:
            return img
        h, w = img.shape[:2]
        try:
            c = stain_concentrations(img, DEFAULT_STAIN_MATRIX)
        except np.linalg.LinAlgError:
            return img
        alpha = 1.0 + rng.uniform(-self.sigma, self.sigma, size=(2, 1))
        beta = rng.uniform(-self.bias, self.bias, size=(2, 1))
        c = np.clip(c * alpha + beta, 0.0, None)
        out = od_to_rgb((DEFAULT_STAIN_MATRIX @ c).T).reshape(h, w, 3)
        return out


@dataclass
class Compose:
    """Apply several augmenters in order, sharing one generator."""

    steps: tuple

    def __call__(self, rgb: np.ndarray, rng: np.random.Generator | None = None
                 ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        for step in self.steps:
            rgb = step(rgb, rng)
        return rgb


def default_pipeline(strength: str = "strong") -> Compose:
    """The recommended stack. `strength` exists so an ablation is one word.

    "none" returns a Compose with no steps rather than None, so calling code
    never has to branch on whether augmentation is on.
    """
    if strength == "none":
        return Compose(())
    if strength == "mild":
        return Compose((ColourJitter(hue=0.04, saturation=0.25, brightness=0.25,
                                     contrast=0.25, gamma=0.2),))
    if strength == "strong":
        return Compose((StainJitter(), ColourJitter()))
    raise ValueError(f"unknown strength {strength!r}; "
                     f"expected none, mild or strong")
