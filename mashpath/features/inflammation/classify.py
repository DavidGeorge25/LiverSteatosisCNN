"""Heuristic hepatocyte / immune split on nuclear size and intensity.

This is a first pass, and deliberately a blunt one. It exists to answer a
prior question -- do small-dark and large-pale nuclei actually form separable
populations on these slides? -- before any clustering is built on top of it.
So it reports an explicit `ambiguous` class instead of forcing a decision, and
it ships a `suggest_cutoffs` helper that reads thresholds back off the observed
distribution rather than off anyone's prior.

What it cannot do, by construction: separate lymphocytes from Kupffer and
endothelial nuclei. Those are all small and dark. Shape helps a little --
endothelial nuclei are flattened against the sinusoid and score high on
eccentricity, lymphocytes are round -- so eccentricity is measured and carried
through, but no cutoff is applied to it here. Sinusoidal-lining cells are a
background that the focus-detection stage has to reject by density, not
something this split can remove.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ClassifyConfig

DEBRIS = 0
IMMUNE = 1
HEPATOCYTE = 2
AMBIGUOUS = 3

CLASS_NAMES = {
    DEBRIS: "debris",
    IMMUNE: "immune",
    HEPATOCYTE: "hepatocyte",
    AMBIGUOUS: "ambiguous",
}


def intensity_for(
    cfg: ClassifyConfig, gray: np.ndarray, hematoxylin: np.ndarray
) -> np.ndarray:
    """The intensity array the configured channel refers to."""
    if cfg.intensity_channel == "gray":
        return gray
    if cfg.intensity_channel == "hematoxylin":
        return hematoxylin
    raise ValueError(
        f"classify.intensity_channel must be 'gray' or 'hematoxylin', "
        f"got {cfg.intensity_channel!r}"
    )


def is_dark(cfg: ClassifyConfig, intensity: np.ndarray) -> np.ndarray:
    """Dark-enough-to-be-immune, with the comparison direction per channel."""
    if cfg.intensity_channel == "gray":
        return intensity <= cfg.immune_max_intensity
    return intensity >= cfg.immune_min_hematoxylin


def classify(
    area_um2: np.ndarray,
    gray: np.ndarray,
    hematoxylin: np.ndarray,
    cfg: ClassifyConfig,
) -> np.ndarray:
    """Assign a class code to every nucleus. See `ClassifyConfig` for the rule."""
    n = area_um2.size
    out = np.full(n, AMBIGUOUS, dtype=np.int8)
    if n == 0:
        return out

    intensity = intensity_for(cfg, gray, hematoxylin)
    small = area_um2 <= cfg.immune_max_area_um2
    immune = small & is_dark(cfg, intensity) if cfg.require_dark else small

    # Order matters: immune wins over hepatocyte where the two bands overlap,
    # and debris wins over everything.
    out[area_um2 >= cfg.hepatocyte_min_area_um2] = HEPATOCYTE
    out[immune] = IMMUNE
    out[area_um2 < cfg.immune_min_area_um2] = DEBRIS
    return out


@dataclass
class ClassCounts:
    """Class tallies for a tile or a whole slide."""

    counts: dict[int, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def scored(self) -> int:
        """Everything except debris -- the denominator that matters."""
        return self.total - self.counts.get(DEBRIS, 0)

    def get(self, code: int) -> int:
        return self.counts.get(code, 0)

    def fraction(self, code: int) -> float:
        return self.get(code) / self.scored if self.scored else 0.0

    def __add__(self, other: "ClassCounts") -> "ClassCounts":
        merged = dict(self.counts)
        for k, v in other.counts.items():
            merged[k] = merged.get(k, 0) + v
        return ClassCounts(merged)

    def line(self) -> str:
        return "  ".join(
            f"{CLASS_NAMES[c]} {self.get(c)} ({self.fraction(c) * 100:.1f}%)"
            for c in (IMMUNE, HEPATOCYTE, AMBIGUOUS)
        ) + f"  debris {self.get(DEBRIS)}"


def tally(classes: np.ndarray) -> ClassCounts:
    vals, cnts = np.unique(classes, return_counts=True)
    return ClassCounts({int(v): int(c) for v, c in zip(vals, cnts)})


def suggest_cutoffs(
    area_um2: np.ndarray, intensity: np.ndarray, channel: str
) -> dict[str, float]:
    """Read candidate cutoffs off the observed distributions.

    Otsu on each marginal, plus the percentiles around it. These are printed as
    suggestions only -- nothing is applied automatically, because an Otsu split
    of a unimodal distribution still returns a number, and that number would
    look just as authoritative as a real one. Compare `otsu_area` against the
    histogram before trusting it.
    """
    from skimage.filters import threshold_otsu

    out: dict[str, float] = {}
    if area_um2.size >= 32:
        out["otsu_area_um2"] = float(threshold_otsu(area_um2))
        for p in (5, 25, 50, 75, 95):
            out[f"area_p{p}"] = float(np.percentile(area_um2, p))
    if intensity.size >= 32:
        out[f"otsu_{channel}"] = float(threshold_otsu(intensity))
        for p in (5, 25, 50, 75, 95):
            out[f"{channel}_p{p}"] = float(np.percentile(intensity, p))
    return out


def bimodality_coefficient(values: np.ndarray) -> float:
    """Sarle's bimodality coefficient: (skew^2 + 1) / kurtosis.

    Above ~0.555 (the value for a uniform distribution) suggests the sample is
    not coming from a single mode. A weak signal, not a test -- but it puts a
    number on "are these two populations or one smear", which is the actual
    question at this stage.
    """
    v = values[np.isfinite(values)]
    n = v.size
    if n < 32:
        return float("nan")
    m = v.mean()
    sd = v.std()
    if sd == 0:
        return float("nan")
    z = (v - m) / sd
    skew = float((z**3).mean())
    excess_kurt = float((z**4).mean()) - 3.0
    # Sarle / SAS formulation: excess kurtosis, plus the sample-size correction.
    num = skew**2 + 1.0
    den = excess_kurt + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    return num / den if den else float("nan")
