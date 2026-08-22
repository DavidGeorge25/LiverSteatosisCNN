"""Steatosis detection parameters.

Kept in the feature that uses them; the shared sections (tissue, tiling, qc)
live in `mashpath.core.config`. In YAML this is the `fat:` block, unchanged
from before the restructure so every tuned config and every `config_used.yaml`
already on disk still loads.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FatConfig:
    """Fat droplet pseudo-labeling."""

    method: str = "fixed"  # "fixed" | "otsu"
    white_threshold: int = 200

    fill_holes: bool = True
    open_radius_px: int = 2
    close_radius_px: int = 2

    # Macrovesicular droplets ~5-50 um diameter -> area in um^2.
    min_area_um2: float = 20.0
    max_area_um2: float = 2000.0

    circularity_min: float = 0.65
    solidity_min: float = 0.85

    # Best-fit-ellipse eccentricity ceiling (0 = circle, 1 = line). Rejects
    # elongated vessel/sinusoidal lumens, which circularity does not catch:
    # circularity responds to boundary raggedness, not to elongation, so a
    # smooth 2.5:1 lumen scores ~0.87 and sails through. Macro class only --
    # the micro band is too small for a stable ellipse fit. 1.0 disables it.
    max_eccentricity: float = 0.80

    # Microvesicular steatosis: fine speckling below the macro min-area cutoff.
    include_microvesicular: bool = False
    micro_min_area_um2: float = 3.0
    micro_max_area_um2: float = 20.0
    micro_circularity_min: float = 0.5
    micro_solidity_min: float = 0.8

    # Drop components touching the tile border -- usually tears or the
    # tissue-edge gap rather than whole droplets.
    exclude_border: bool = False
