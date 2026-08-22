"""Lobular inflammation candidate generation.

NOT IMPLEMENTED YET -- this module defines the contract the rest of the
pipeline already builds on, so implementing it is a local change.

Stages 1-2 ARE implemented, in `nuclei.py` and `classify.py`, and are driven by
`python -m mashpath.features.inflammation.cli nuclei`. What they established on
R26-122-1_HE_70 (60 tiles, 3.76 mm^2, 8,779 nuclei) shapes what is left:

  * StarDist must run at `nuclei.upscale: 2`. The `2D_versatile_he` weights
    were trained near 40x; at this slide's 20x, upscale 1 finds 1,215
    nuclei/mm^2 and detects essentially only hepatocyte nuclei -- the entire
    small-dark population is invisible, so the immune class is empty by
    construction. Upscale 2 finds 2,337/mm^2 and recovers it.

  * Mean intensity does not discriminate. Grayscale within the nuclear mask
    runs p5-p95 = 67-109 for every class; any cutoff either admits ~all nuclei
    or none. The hepatocyte/immune split is currently area alone, and nuclear
    area is unimodal (bimodality coefficient 0.27) -- the cutoff bisects one
    continuum rather than separating two populations.

  * The resulting "immune" class is 55% round / 45% eccentricity > 0.70, and
    appears in 60 of 60 tiles at CV 0.33. That is the resident sinusoidal
    lining population -- endothelial and Kupffer nuclei -- not lymphocytes.

Which changes step 4 below from what it looked like before those numbers
existed: absolute density thresholds cannot work on top of a ~1,100/mm^2
uniform floor of resident cells. The plan:

  1. Segment nuclei at `nuclei.upscale` and keep the size band
     (`nuclei.min_area_um2` / `max_area_um2`).  [done -- nuclei.py]
  2. Classify hepatocyte vs immune (`classify.*`). Shape is the missing
     discriminator: lymphocyte nuclei are round, sinusoidal lining nuclei are
     flattened. `eccentricity` is already measured per nucleus and carried
     into nuclei.csv; it is not yet a cutoff.  [partly done -- classify.py]
  3. Detect portal tracts -- sustained high nuclear density over
     `portal.min_area_um2`, and/or large clear lumens -- and dilate them by
     `portal.exclusion_radius_um`. Report the excluded fraction of tissue
     (`portal.max_excluded_fraction` bounds it), because an exclusion that
     eats the parenchyma is the failure mode that looks like success.
  4. Cluster the survivors on LOCAL EXCESS over the resident baseline, not on
     absolute counts: DBSCAN at `foci.cluster_radius_um` / `foci.min_cells`
     against a background rate estimated from the same slide.
  5. Emit each focus as a `Candidate` scored on cell count and density.

Every threshold is in `InflammationConfig` (config.py); nothing belongs
hardcoded below. Note that `mashpath/config.py` currently composes the
`inflammation:` YAML block from three sub-configs by hand -- switch it to
`InflammationConfig` to pick up `portal:` and `foci:`.
"""

from __future__ import annotations

import numpy as np

from ...review.candidates import Candidate
from .config import InflammationConfig

FEATURE = "inflammation"


def detect_candidates(
    rgb: np.ndarray,
    tissue_mask: np.ndarray,
    cfg: InflammationConfig,
    um2_per_px: float,
    slide_id: str = "",
    tile_x: int = 0,
    tile_y: int = 0,
) -> list[Candidate]:
    """Propose lobular inflammatory foci in one tile.

    Returns candidates in LEVEL-0 coordinates, highest score first, capped at
    `cfg.foci.max_candidates_per_tile`.

    `nuclei.segment_array` already takes exactly this shape of input -- pixels
    plus a tile-local tissue mask -- so steps 1-2 plug in directly here.
    """
    raise NotImplementedError(
        "lobular inflammation candidate generation is not implemented yet. "
        "Stages 1-2 are: run `python -m mashpath.features.inflammation.cli "
        "nuclei --slide <path>` for segmentation, the hepatocyte/immune split "
        "and its QC. What is missing is steps 3-5 in this module's docstring "
        "-- portal tract exclusion and focus clustering. Those were held back "
        "deliberately: on this cohort the immune class is currently a uniform "
        "resident population, so clustering it would produce foci everywhere. "
        "Run with --plan to exercise the tile plan without calling this."
    )
