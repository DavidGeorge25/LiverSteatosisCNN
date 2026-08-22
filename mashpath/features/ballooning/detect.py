"""Hepatocyte ballooning candidate generation -- per-tile detector contract.

NOT IMPLEMENTED as a per-tile call, and deliberately so. See `pipeline.py`.

`review.scan.scan_slide` hands a detector one tile at a time and expects
candidates back from that tile alone. Ballooning cannot answer on those terms.
The judgment is *relative* -- a hepatocyte is ballooned because it is larger
and paler than the hepatocytes around it -- and the neighbourhood that defines
"around it" is ~150 um in radius, while a 512 px tile is only ~254 um across.
Normalizing inside a single tile would mean:

  - every tile gets its own reference population, so the same cell scores
    differently depending on where the tile grid happened to fall;
  - a tile that is *uniformly* ballooned normalizes its own signal away and
    reports nothing, which is exactly backwards on the worst tissue;
  - cells near a tile edge are compared against a half-neighbourhood.

So ballooning runs as two passes over the slide instead: segment and measure
every cell tile by tile (streaming, level 0 never held whole), then z-score
each cell against its true spatial neighbours across tile boundaries, then
rank. `pipeline.py` does that, and `cli.py` is its entry point:

    python -m mashpath.features.ballooning.cli segment --slide X.svs ...
    python -m mashpath.features.ballooning.cli run     --slide X.svs ...

This module stays so the feature registry in `mashpath.cli` keeps importing,
and raises with that explanation rather than silently producing per-tile
z-scores that would look plausible and be wrong.
"""

from __future__ import annotations

import numpy as np

from ...review.candidates import Candidate
from .config import BallooningConfig

FEATURE = "ballooning"


def detect_candidates(
    rgb: np.ndarray,
    tissue_mask: np.ndarray,
    cfg: BallooningConfig,
    um2_per_px: float,
    slide_id: str = "",
    tile_x: int = 0,
    tile_y: int = 0,
) -> list[Candidate]:
    """Not available per tile -- ballooning needs a slide-wide neighbourhood."""
    raise NotImplementedError(
        "ballooning has no per-tile detector: the call is relative to a "
        f"~{cfg.normalization.radius_um:.0f} um neighbourhood, which is wider "
        "than one tile, and normalizing within a tile would erase the signal "
        "on uniformly ballooned tissue. Use the two-pass pipeline instead:\n"
        "  python -m mashpath.features.ballooning.cli run --slide <path.svs> "
        "--config configs/core.yaml --config configs/ballooning.yaml"
    )
