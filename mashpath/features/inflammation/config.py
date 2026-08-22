"""Lobular inflammation parameters.

In YAML this is the `inflammation:` block of the root config, alongside `fat:`
and `ballooning:`; the shared `tissue:`, `tiling:` and `qc:` sections come from
`mashpath.core.config` and behave exactly as they do for steatosis.

Every cutoff here is a first guess, not a calibrated value. Mouse hepatocyte
nuclei run roughly 7-10 um across (~40-80 um^2) and lymphocyte nuclei roughly
5-7 um (~20-38 um^2), so the two populations overlap in the 35-45 um^2 band by
construction -- which is why `classify` carries an explicit `ambiguous` class
rather than forcing every nucleus to one side of a line, and why the defaults
below are meant to be replaced by numbers read off a real distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NucleiConfig:
    """StarDist nucleus segmentation over level-0 tiles."""

    # Pretrained model. `2D_versatile_he` takes RGB H&E directly and was
    # trained near 20x, which matches these slides (0.4953 um/px).
    model_name: str = "2D_versatile_he"
    # Local model directory, for offline runs. None = download and cache via
    # StarDist's own pretrained registry.
    model_dir: str | None = None

    # Pyramid level to segment at. Level 0 -- nuclei are a few microns across
    # and do not survive a 4x downsample.
    level: int = 0

    # Detection thresholds. None uses the values shipped with the model, which
    # is the right starting point; raise prob to cut false positives.
    prob_threshold: float | None = None
    nms_threshold: float | None = None

    # Tiles are read with this much extra context on every side, and only
    # nuclei whose CENTROID lands in the core tile are kept, so a nucleus on a
    # tile boundary is measured whole and counted once. 64 px = ~32 um,
    # comfortably larger than any nucleus. Costs ~28% more pixels per tile at
    # 512; set to 0 to disable (and accept split nuclei at every seam).
    margin_px: int = 64

    # Integer upsampling applied before segmentation. `2D_versatile_he` was
    # trained on ~40x H&E; these slides are 20x (0.4953 um/px), so a 5 um
    # lymphocyte nucleus arrives ~10 px across where the model expects ~20 and
    # gets missed. Segmenting at 2x recovers that population. Areas and
    # centroids are converted back to level-0 units, so every measurement stays
    # in the same frame regardless of this setting. Costs ~4x the compute.
    upscale: int = 2

    # Percentile normalization fed to StarDist, computed on the CORE tile only
    # so white padding at the slide edge cannot skew the range.
    norm_low_percentile: float = 1.0
    norm_high_percentile: float = 99.8

    # Detections outside this band are dropped before classification: below is
    # debris and segmentation speckle, above is a merged clump.
    min_area_um2: float = 4.0
    max_area_um2: float = 400.0

    # Drop nuclei whose centroid falls outside the tissue mask. Coarse -- the
    # mask is a 16x downsample -- but it clears the background of edge tiles.
    mask_to_tissue: bool = True


@dataclass
class ClassifyConfig:
    """Heuristic hepatocyte / immune split on nuclear size and intensity.

    Applied per nucleus, in this order:

        area <  immune_min_area_um2                  -> dropped (debris)
        area <= immune_max_area_um2 and dark enough   -> immune
        area >= hepatocyte_min_area_um2               -> hepatocyte
        otherwise                                     -> ambiguous

    Setting `hepatocyte_min_area_um2 == immune_max_area_um2` with
    `require_dark: false` collapses this to a plain two-way split on area.
    """

    # Which per-nucleus intensity to threshold on.
    #   "gray"        mean grayscale 0-255, lower = darker. Intuitive.
    #   "hematoxylin" mean haematoxylin optical density from colour
    #                 deconvolution, higher = more chromatin. Less
    #                 contaminated by surrounding eosin, but the numbers are
    #                 unitless ODs -- retune the cutoff when switching.
    intensity_channel: str = "gray"

    # Direction is handled per channel: on "gray" a nucleus is dark when
    # intensity <= immune_max_intensity; on "hematoxylin" when
    # intensity >= immune_min_hematoxylin.
    immune_max_intensity: float = 110.0
    immune_min_hematoxylin: float = 0.45

    immune_min_area_um2: float = 6.0
    immune_max_area_um2: float = 35.0
    hepatocyte_min_area_um2: float = 40.0

    # Require small AND dark for the immune call. False ignores intensity
    # entirely and splits on area alone -- the honest fallback if the intensity
    # modes turn out not to separate.
    require_dark: bool = True


@dataclass
class NucleusQCConfig:
    """QC for the segmentation and the class split."""

    # RGB triples. Checked for colour-vision separation as a categorical set:
    # worst adjacent pair dE 25.4 protan / 14.7 tritan. Green reads clearly
    # against H&E pink, which is why the class of interest gets it.
    immune_color: tuple[int, int, int] = (0, 176, 80)
    hepatocyte_color: tuple[int, int, int] = (21, 101, 192)
    ambiguous_color: tuple[int, int, int] = (230, 81, 0)
    outline_thickness: int = 1
    fill_alpha: float = 0.30  # 0 = outline only

    # Tiles drawn into the class-overlay contact sheet.
    sample_tiles: int = 12
    cols: int = 3
    # Also write each sampled tile at full size -- nuclei are ~15 px across and
    # are not legible once a 512 px tile is shrunk into a grid cell.
    save_full_size_panels: bool = True

    # Points kept for the area-vs-intensity scatter. A whole slide produces
    # millions of nuclei; the plot saturates long before that.
    scatter_max_points: int = 150_000
    scatter_dpi: int = 150

    # Write the full per-nucleus table. Off for whole-slide runs unless you
    # want a multi-hundred-MB CSV.
    save_nucleus_csv: bool = True


# --- the sections below are scaffolded but not yet wired up ---------------
# Stages 3-5 (portal exclusion, focus clustering) are deliberately not
# implemented until the size/intensity split above is shown to separate on real
# tissue. The parameters live here so the YAML is complete and the plan is
# reviewable, but nothing reads them yet.


@dataclass
class PortalConfig:
    """Portal tract detection, so tract immune cells are not called lobular.

    This is the part that decides whether the whole detector is usable. A
    portal tract contains lymphocytes *normally*, so a detector that skips this
    fires on healthy anatomy on every slide. Two strategies, cheapest first;
    `method` selects which runs.

      "density"  sustained high nuclear density over an area -- what a tract
                 looks like once you stop resolving individual structures.
      "vessel"   find large clear lumens (portal vein / artery / duct) and
                 exclude a radius around them.
      "both"     union of the two.

    Whichever runs, the excluded area is reported as a fraction of tissue, so
    over-aggressive exclusion is visible rather than silent.
    """

    enabled: bool = True
    method: str = "density"  # density | vessel | both

    # --- density strategy ---
    # Nuclei per mm^2 sustained over `min_area_um2` marks a tract.
    nucleus_density_min: float = 6000.0
    min_area_um2: float = 10_000.0  # ~113 um across; smaller is a focus
    smoothing_um: float = 40.0

    # --- vessel strategy ---
    # A portal vein lumen is large and empty; sinusoids are far smaller.
    vessel_min_area_um2: float = 2000.0
    vessel_max_eccentricity: float = 0.95
    vessel_white_threshold: int = 215

    # Everything within this distance of a detected tract is excluded.
    # Generous on purpose: it costs recall at the tract margin and buys
    # specificity, the right trade when a pathologist reviews the output.
    exclusion_radius_um: float = 150.0

    # Refuse to proceed if exclusion eats more than this fraction of tissue --
    # that means the detector is finding tracts everywhere and the run would be
    # meaningless. Reported either way.
    max_excluded_fraction: float = 0.40


@dataclass
class FociConfig:
    """Clustering immune nuclei into candidate inflammatory foci."""

    # DBSCAN over immune-classified centroids, in microns.
    # How far above the slide's own resident baseline a region must be before
    # its nuclei can seed a focus. This is THE parameter that separates a
    # focus from the sinusoidal lining, and it exists because absolute density
    # cutoffs cannot work here: the resident small-dark population runs
    # ~1,100/mm^2 uniformly across every tile of a control slide, so any
    # absolute threshold either calls the whole slide inflamed or nothing.
    # 2.5x is a starting point, NOT tuned -- it is the first thing to move
    # once pathologist verdicts exist to move it against.
    min_excess_ratio: float = 2.5

    cluster_radius_um: float = 30.0
    min_cells: int = 5  # NASH-CRN counts foci, not cells

    # A focus this dense is more likely a tract the portal stage missed.
    max_cells: int = 200

    # Padding added around the cluster's extent for the review crop.
    bbox_pad_um: float = 20.0

    # Cap per tile before review export, highest scoring first.
    max_candidates_per_tile: int = 20


@dataclass
class InflammationConfig:
    """The `inflammation:` block: everything the feature needs, and nothing else."""

    nuclei: NucleiConfig = field(default_factory=NucleiConfig)
    classify: ClassifyConfig = field(default_factory=ClassifyConfig)
    nucleus_qc: NucleusQCConfig = field(default_factory=NucleusQCConfig)
    portal: PortalConfig = field(default_factory=PortalConfig)
    foci: FociConfig = field(default_factory=FociConfig)
