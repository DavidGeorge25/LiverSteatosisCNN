"""Parameter-ensemble pseudo-labeling with per-pixel confidence.

No annotations exist, and none are coming -- that is the premise of a weakly
supervised design, not a gap to be filled. But it does mean there is no way to
identify a single "correct" parameter setting: every cell in the sweep grid
separates steatotic from fat-free tissue perfectly while reporting fat
fractions that differ by 1.6x.

Rather than pick one arbitrarily and hide the choice, run several plausible
settings and use their agreement. Pixels every setting calls fat are reliable;
pixels only some call fat are where the labeling is genuinely uncertain. That
yields two things a single setting cannot:

  * a consensus mask that is more robust than any individual member, and
  * a per-pixel confidence map, which a weakly supervised model can use to
    down-weight contested pixels in its loss instead of treating every
    pseudo-label as equally certain.

Implementation note: the expensive step (threshold + morphology + measure)
depends only on `white_threshold`, and the shape filters only relabel existing
components. So components are extracted once per distinct white value, and each
member's verdict becomes a per-label lookup table. Voting is then one fancy-
index per white value rather than one segmentation per member.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

from .config import FatConfig
from .detect import MACRO, MICRO, classify, extract_components


@dataclass(frozen=True)
class EnsembleMember:
    name: str
    cfg: FatConfig


def default_ensemble(base: FatConfig) -> list[EnsembleMember]:
    """A spread over the axes that actually move the measurement.

    Chosen to span the plausible region rather than to cluster near one
    setting: if the members agreed by construction, their agreement would mean
    nothing. `white_threshold` controls droplet boundaries, `min_area_um2` the
    small-droplet cutoff, `max_eccentricity` how aggressively elongated lumens
    are rejected -- the three that drove the sweep.
    """
    grid = [
        (200, 20.0, 0.80),
        (200, 40.0, 0.85),
        (200, 60.0, 0.75),
        (210, 20.0, 0.85),
        (210, 40.0, 0.80),
        (210, 60.0, 0.80),
        (220, 20.0, 0.75),
        (220, 40.0, 0.80),
        (220, 60.0, 0.85),
    ]
    return [
        EnsembleMember(
            name=f"w{w}_a{int(a)}_e{e:.2f}",
            cfg=replace(
                base,
                white_threshold=w,
                min_area_um2=a,
                micro_max_area_um2=a,  # min_area doubles as the macro/micro split
                max_eccentricity=e,
            ),
        )
        for w, a, e in grid
    ]


@dataclass
class ConsensusResult:
    votes: np.ndarray  # per-pixel count of members calling it fat
    n_members: int
    tissue_area_um2: float
    um2_per_px: float
    per_member_pct: dict[str, float] = field(default_factory=dict)

    @property
    def confidence(self) -> np.ndarray:
        """Vote fraction in [0, 1] -- the per-pixel label confidence."""
        return self.votes.astype(np.float32) / max(1, self.n_members)

    def mask_at(self, agree: float = 0.5) -> np.ndarray:
        """Consensus mask: pixels at least `agree` of members call fat."""
        return self.votes >= np.ceil(agree * self.n_members)

    @property
    def unanimous_mask(self) -> np.ndarray:
        return self.votes == self.n_members

    @property
    def any_mask(self) -> np.ndarray:
        return self.votes > 0

    def fat_fraction(self, agree: float = 0.5) -> float:
        """Consensus fat area as a fraction of tissue area in the tile."""
        if self.tissue_area_um2 <= 0:
            return 0.0
        px = float(self.mask_at(agree).sum())
        return px * self.um2_per_px / self.tissue_area_um2

    def agreement_stats(self, agree: float = 0.5) -> dict[str, float]:
        """How much of the labeled area is contested rather than certain."""
        any_px = float(self.any_mask.sum())
        cons_px = float(self.mask_at(agree).sum())
        unan_px = float(self.unanimous_mask.sum())
        return {
            "any_px": any_px,
            "consensus_px": cons_px,
            "unanimous_px": unan_px,
            # Of everything at least one member calls fat, how much is certain?
            "unanimous_of_any": unan_px / any_px if any_px else 0.0,
            # Of the exported label, how much is unanimous?
            "unanimous_of_consensus": unan_px / cons_px if cons_px else 0.0,
            "contested_of_consensus": 1 - (unan_px / cons_px) if cons_px else 0.0,
        }


def consensus(
    rgb: np.ndarray,
    tissue_mask: np.ndarray,
    members: Sequence[EnsembleMember],
    um2_per_px: float,
) -> ConsensusResult:
    """Vote a tile across ensemble members."""
    votes = np.zeros(rgb.shape[:2], dtype=np.uint8)
    per_member: dict[str, float] = {}
    tissue_area = float(tissue_mask.sum()) * um2_per_px

    by_white: dict[int, list[EnsembleMember]] = {}
    for m in members:
        by_white.setdefault(int(m.cfg.white_threshold), []).append(m)

    for white, group in by_white.items():
        cset = extract_components(
            rgb, tissue_mask, replace(group[0].cfg, white_threshold=white), um2_per_px
        )
        if cset.labels.size == 0:
            continue
        max_label = int(cset.labels.max())
        lut = np.zeros(max_label + 1, dtype=np.uint8)

        for m in group:
            accepted = 0.0
            for c in cset.components:
                klass, _ = classify(c, m.cfg)
                if klass is None:
                    continue
                if klass == MICRO and not m.cfg.include_microvesicular:
                    continue
                if c.touches_border and m.cfg.exclude_border:
                    continue
                lut[c.label] += 1
                accepted += c.area_um2
            per_member[m.name] = (
                accepted / tissue_area * 100 if tissue_area > 0 else 0.0
            )

        votes += lut[cset.labels]

    return ConsensusResult(
        votes=votes,
        n_members=len(members),
        tissue_area_um2=tissue_area,
        um2_per_px=um2_per_px,
        per_member_pct=per_member,
    )
