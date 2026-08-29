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

    # ---- splitting touching droplets ----------------------------------
    #
    # In dense regions adjacent droplets threshold into ONE white component.
    # That used only to inflate droplet size; since `max_eccentricity` arrived
    # it is worse than that -- a merged pair is elongated, so the pair is
    # rejected outright and its area is lost. It is most of the 6-7% signal
    # cost the eccentricity filter carries on severe slides, which is exactly
    # backwards: the filter exists to remove vessel lumens, not fat.
    #
    # OFF BY DEFAULT, and that is the deliberate half. Splitting is an
    # intervention on the boundaries that ARE the exported pseudo-label, and
    # the failure mode is silent: over-splitting one ragged droplet into three
    # raises the count, lowers the mean size, and leaves the fat FRACTION
    # almost unchanged, so the headline number cannot detect it. The first
    # numbers must be the unsplit baseline any later claim has to beat --
    # measured on the CCl4 floor and on the R22-354 diet contrast, not on
    # whether the fat fraction went up.
    watershed: bool = False
    # Minimum depth, in microns of distance-transform, by which a peak must
    # exceed the saddle joining it to a taller one before it counts as a
    # separate droplet. This is the over-splitting guard: a single droplet with
    # a scalloped border carries many shallow maxima and must yield exactly one
    # seed. Depth rather than distance because it does not care how far apart
    # two bumps are, only how deep the notch between them is.
    watershed_min_depth_um: float = 1.5
    # Only attempt a split on components at least this large. A component the
    # size of one droplet cannot be two droplets, and trying costs accuracy on
    # the class that is already correct. Default 0 means 2x min_area_um2,
    # resolved against the config at call time so it tracks a retune.
    watershed_min_area_um2: float = 0.0

    # Drop components touching the tile border -- usually tears or the
    # tissue-edge gap rather than whole droplets.
    exclude_border: bool = False
