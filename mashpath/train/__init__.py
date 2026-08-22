"""Everything the model will need that is not a model.

Built before there are any labels, on purpose: the pathologist is labelling now,
and the parts of a training pipeline that decide whether a result means anything
— how the data is split, what is normalized, what is jittered, how it is scored
— are the parts that are hardest to change once numbers exist and easiest to get
wrong under deadline. None of this trains anything. It is tested against
synthetic labels, which is enough to prove the plumbing and the guards.

    splits.py     batch-aware splitting; a tile-level split is not expressible
    evaluate.py   leave-one-batch-out, reported per unseen batch
    stain.py      Macenko / Reinhard normalization, off by default
    augment.py    colour and stain jitter, on by default

The two defaults are opposite for a reason spelled out in each file: normalizing
is an intervention that has to prove it helped, augmenting is a shortcut removal
that has to justify being switched off.
"""

from __future__ import annotations

from .augment import ColourJitter, Compose, StainJitter, default_pipeline
from .evaluate import (auc, batch_separability, binary_metrics,
                       held_out_batch_eval, per_batch_metrics, report)
from .splits import (BatchSplit, LeakySplitError, assert_no_leakage,
                     leave_one_batch_out, split_by_batch)
from .stain import (MacenkoNormalizer, NullNormalizer, ReinhardNormalizer,
                    make_normalizer)

__all__ = [
    "BatchSplit", "LeakySplitError", "assert_no_leakage", "leave_one_batch_out",
    "split_by_batch", "auc", "batch_separability", "binary_metrics",
    "held_out_batch_eval", "per_batch_metrics", "report", "ColourJitter",
    "Compose", "StainJitter", "default_pipeline", "MacenkoNormalizer",
    "NullNormalizer", "ReinhardNormalizer", "make_normalizer",
    "normalizer_from_config", "augmenter_from_config",
]


def normalizer_from_config(cfg):
    """Build a normalizer from a `StainConfig`. Disabled means a real no-op
    object, not None, so callers never branch on it."""
    from .stain import NullNormalizer, make_normalizer
    if not getattr(cfg, "enabled", False) or cfg.method == "none":
        return NullNormalizer()
    reference = None
    if cfg.reference_image:
        import cv2
        img = cv2.imread(cfg.reference_image)
        if img is None:
            raise FileNotFoundError(f"stain.reference_image {cfg.reference_image!r} "
                                    f"could not be read")
        reference = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    norm = make_normalizer(cfg.method, reference)
    for attr in ("beta", "alpha"):
        if hasattr(norm, attr) and hasattr(cfg, attr):
            setattr(norm, attr, getattr(cfg, attr))
    return norm


def augmenter_from_config(cfg):
    """Build an augmenter from an `AugmentConfig`."""
    from .augment import ColourJitter, Compose, StainJitter
    if cfg.strength == "none":
        return Compose(())
    jitter = ColourJitter(hue=cfg.hue, saturation=cfg.saturation,
                          brightness=cfg.brightness, contrast=cfg.contrast,
                          gamma=cfg.gamma, probability=cfg.probability)
    if cfg.strength == "mild":
        return Compose((jitter,))
    if cfg.strength == "strong":
        return Compose((StainJitter(sigma=cfg.stain_sigma, bias=cfg.stain_bias,
                                    probability=cfg.probability), jitter))
    raise ValueError(f"unknown augment.strength {cfg.strength!r}; "
                     f"expected none, mild or strong")
