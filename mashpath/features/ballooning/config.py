"""Hepatocyte ballooning parameters.

Ballooning has no void and no clean geometric signature, and the call is
*relative* -- a hepatocyte is ballooned when it is markedly larger and paler
than its neighbours, which differs slide to slide and even lobule to lobule.
So nothing here is an absolute detection threshold. The absolute bounds exist
only to say what could physically be one hepatocyte; the actual ballooning
signal is a z-score against a local neighbourhood, and what comes out is a
CANDIDATE for pathologist review, not a detection.

Sizes are in microns. A normal mouse hepatocyte is ~20-25 um across (~350-500
um^2 in section); ballooned cells are typically >=1.5-2x that in diameter.

The sections mirror the pipeline stages:

    segmentation -> features -> normalization -> ranking -> crops
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SegmentationConfig:
    """Nuclei via StarDist, then cell bodies by constrained expansion.

    There is no pretrained model that gives hepatocyte *cell* boundaries on
    H&E -- `2D_versatile_he` segments nuclei. Ballooning is a property of the
    cell body, not the nucleus (a ballooned hepatocyte's nucleus is normal
    sized, just displaced), so nuclei are used as watershed seeds and grown
    outward into the cytoplasm, stopping at image edges, at each other, at
    white voids, and at a hard radius cap.
    """

    # --- StarDist ---
    # `2D_versatile_he` takes RGB H&E directly. It was trained near 40x; these
    # slides scan at 20x (0.4953 um/px), so see `upscale` below.
    model_name: str = "2D_versatile_he"
    # Local model directory, for offline runs. None = download and cache via
    # StarDist's own pretrained registry.
    model_dir: str | None = None

    # Pyramid level to segment at. Level 0 -- nuclei are a few microns across
    # and do not survive a 4x downsample.
    level: int = 0

    # None uses the thresholds shipped with the model, which is the right
    # starting point; raise prob to cut false positives.
    prob_threshold: float | None = None
    nms_threshold: float | None = None

    # Segment at `upscale`x, then convert every measurement back to level-0
    # units, so this changes recall and nothing else.
    #
    # Inflammation measured on this cohort that at upscale 1 StarDist finds
    # 1,215 nuclei/mm^2 and at upscale 2 it finds 2,337/mm^2 -- the small dark
    # population is simply invisible at 1. That matters differently here than
    # it does there. Ballooning does not measure nuclei; it uses them as
    # watershed SEEDS, and a missed seed does not just cost a row, it hands
    # that cell's territory to the neighbour that did get a seed. The
    # neighbour then reads as enlarged, which is the ballooning signal itself.
    #
    # Measured both ways on the same 40 tiles of R25-264-1, hepatocytes only:
    #
    #                        upscale 1   upscale 2   expected (mouse)
    #   median territory       720 um2     433 um2     350-500 um2
    #   median diameter       30.3 um     23.5 um       20-25 um
    #   non-hepatocyte nuclei       18         750     ~20% of nuclei
    #   seeds / mm2                760        1472
    #   runtime, 40 tiles          23 s        89 s
    #
    # Upscale 1 inflates the territory ~1.7x and finds 18 non-hepatocyte
    # nuclei in 1,954 cells -- i.e. it does not resolve the sinusoidal lining
    # population at all, so those cells' territory is absorbed by whichever
    # hepatocyte did get a seed. Upscale 2 lands every morphometric on its
    # expected value.
    #
    # The ~4x compute is NOT free on a cluster, contrary to what this comment
    # used to claim. Measured on an H100: StarDist inference is ~57 ms/tile
    # while the whole tile takes ~4-6 s, and during a real run the GPU sits at
    # 0% utilisation with one core at 95%. The cost of upscale 2 is the CPU
    # work -- watershed, GLCM and local entropy over 4x the pixels -- not the
    # inference. Ballooning is a CPU-bound job that happens to need a GPU
    # briefly, which is worth knowing before requesting one per array task.
    upscale: int = 2

    # Per-tile wall-clock ceiling in seconds; 0 disables the watchdog.
    # Tiles run ~2.5-6s, so 300 only fires on something genuinely stuck. See
    # `pipeline._tile_budget_seconds` for why this exists at all.
    tile_timeout_s: int = 300

    # Tiles are read with this much extra context on every side, and only
    # cells whose nuclear CENTROID lands in the core tile are kept -- so a cell
    # on a tile boundary is segmented against its real surroundings and counted
    # exactly once, with no area lost to the seam. 64 px = ~32 um; a ballooned
    # cell can be ~70 um across, so the margin also has to cover the expansion
    # radius (see `max_expansion_um`) for the cell body to close properly.
    margin_px: int = 96

    # Percentile normalization fed to StarDist, computed on the CORE tile only
    # so white padding at a slide edge cannot skew the range.
    norm_low_percentile: float = 1.0
    norm_high_percentile: float = 99.8

    # --- nucleus sanity bounds ---
    # Below is segmentation speckle and debris; above is a merged clump.
    min_nucleus_area_um2: float = 8.0
    max_nucleus_area_um2: float = 250.0

    # --- cell expansion ---
    # Hard cap on how far a cell body may grow from its nucleus boundary. A
    # ballooned hepatocyte ~70 um across with a central nucleus needs ~30 um;
    # this bounds how much empty space one isolated nucleus can claim.
    max_expansion_um: float = 22.0

    # Pixels brighter than this are treated as void (sinusoid lumen, vessel,
    # fat droplet, tear) and cells are not grown into them. This is what stops
    # a hepatocyte from swallowing the sinusoid next to it -- and it is
    # deliberately near steatosis' `white_threshold`, so a macrovesicular
    # droplet reads as void rather than as very pale cytoplasm.
    void_intensity: int = 215

    # ...but only where the bright region is big enough to BE a lumen. Without
    # an area floor the void mask shatters on the pale speckle inside rarefied
    # cytoplasm -- which is precisely the ballooning signal -- and excludes it
    # from the cytoplasm compartment before it can ever be measured. A
    # macrovesicular droplet starts around 20 um^2 (steatosis' min_area_um2);
    # anything smaller is texture, not a void.
    min_void_area_um2: float = 15.0

    # How much the cell boundary follows image edges rather than sitting
    # halfway between two nuclei. 0 gives a pure distance tessellation (each
    # cell takes the territory nearest its own nucleus, bounded by voids and
    # by the radius cap), which is predictable but ignores membranes. The edge
    # map is built on the EOSIN channel, not grey -- grey is dominated by the
    # nuclear rim, and an edge there would trap every cell inside its own
    # nucleus instead of letting it grow into the cytoplasm.
    edge_weight: float = 0.35

    # Gaussian smoothing applied before the edge map that guides the
    # watershed. Enough to kill pixel noise, not enough to blur membranes.
    edge_smoothing_um: float = 0.75

    # Drop cells whose centroid falls outside the tissue mask. Coarse -- the
    # mask is a 16x downsample -- but it clears the background of edge tiles.
    mask_to_tissue: bool = True

    # A cell with less cytoplasm than this has no measurable cytoplasmic
    # compartment; its texture and pallor features would be noise.
    min_cytoplasm_area_um2: float = 30.0

    # --- which segmented cells are hepatocytes at all ---
    # Endothelial, Kupffer, stellate and immune cells all get segmented too.
    # They are kept as watershed seeds (so they claim their own territory
    # rather than being absorbed into a hepatocyte) but are excluded from the
    # ballooning population by these bounds.
    min_cell_area_um2: float = 100.0
    max_cell_area_um2: float = 5000.0
    # Small dark nuclei with almost no cytoplasm are immune/endothelial.
    min_hepatocyte_nucleus_area_um2: float = 20.0


@dataclass
class FeatureConfig:
    """Per-cell morphometry and cytoplasmic texture.

    The texture measures target the "rarefied" appearance: ballooned cytoplasm
    is wispy and heterogeneous -- cleared-out strands rather than the smooth
    even eosinophilia of a normal hepatocyte. Smoothness is the signal that
    inverts, so all three of these are computed on the cytoplasm compartment
    only, never on the whole cell (the nucleus would dominate every one).
    """

    # Window for local standard deviation and local entropy, in microns.
    # Should sit well inside a cell so it measures within-cytoplasm texture
    # rather than straddling the cell boundary.
    texture_window_um: float = 2.5

    # Grey levels the cytoplasm is quantized to before the GLCM. 32 keeps the
    # matrix populated at the small pixel counts one cell provides.
    glcm_levels: int = 32
    # Offsets (in pixels) and angles for the Haralick features. One short
    # offset is enough at this scale; longer ones mostly measure the cell edge.
    glcm_distances: tuple[int, ...] = (1, 3)
    # Which Haralick properties to keep. Contrast rises and homogeneity falls
    # as cytoplasm becomes wispy, so the pair brackets the effect.
    glcm_properties: tuple[str, ...] = ("contrast", "homogeneity", "correlation")

    # Colour deconvolution gives an eosin channel; rarefied cytoplasm loses
    # eosin specifically, which is cleaner than overall brightness because it
    # is not confounded by haematoxylin from the displaced nucleus.
    use_color_deconvolution: bool = True


@dataclass
class NormalizationConfig:
    """Local z-scoring -- the step that makes ballooning measurable.

    Raw absolute values are dominated by staining batch and by zonation
    (pericentral hepatocytes are larger and paler than periportal ones in a
    normal liver). Comparing each cell to the cells physically around it
    removes both at once, which is also how a pathologist actually makes the
    call.
    """

    # Radius of the neighbourhood each cell is compared against, in microns.
    # ~150 um is a few cell diameters out -- inside one zone of one lobule, but
    # containing enough hepatocytes for a stable median.
    radius_um: float = 150.0

    # A neighbourhood with fewer cells than this cannot support a z-score; the
    # cell is kept with its raw features and flagged rather than scored off a
    # sample of three.
    min_neighbours: int = 25

    # Use median and a robust scale (1.4826 * MAD) instead of mean and standard
    # deviation. Ballooned cells are outliers by construction, so a plain
    # standard deviation is inflated by exactly the cells being looked for --
    # which shrinks their own z-scores. Robust statistics avoid that.
    robust: bool = True

    # Floor on the scale estimate, as a fraction of the neighbourhood centre.
    # Stops a uniform neighbourhood (MAD ~ 0) from turning a trivial deviation
    # into an enormous z-score.
    min_scale_fraction: float = 0.05

    # Only hepatocytes form the reference population. Including immune and
    # endothelial cells would drag the local median area down and make every
    # hepatocyte look enlarged.
    reference_hepatocytes_only: bool = True


@dataclass
class RankingConfig:
    """Turning normalized features into a reviewable, unbiased sample.

    The score is a weighted sum of local z-scores on the three axes that define
    ballooning. Weights are a starting point, not a calibration -- there is no
    ground truth to fit them to yet. That is the point of the review round.
    """

    # Feature -> weight, applied to LOCAL z-scores. Positive means "more of
    # this looks more ballooned".
    #
    # Pallor and texture carry the score, not size. That is a deliberate
    # response to what the segmentation actually produces: the territory area
    # is a nuclear-spacing estimate rather than a measured cell boundary (see
    # `segment`'s module docstring), so it carries stereological noise on the
    # very axis that would otherwise dominate. The cytoplasm features do not
    # have that problem -- they describe the pixels inside the territory, and a
    # ballooned cell's cytoplasm is pale and wispy wherever the boundary lands.
    weights: dict[str, float] = field(
        default_factory=lambda: {
            # --- primary: the cytoplasm itself ---
            "z_cyto_eosin_od_mean": -1.0,       # rarefaction removes eosin
            "z_cyto_local_std_mean": 0.9,       # wispy, not smooth
            "z_cyto_gray_mean": 0.8,            # brighter grey = paler
            "z_cyto_entropy_mean": 0.7,
            "z_cyto_glcm_contrast_d1": 0.5,
            "z_cyto_glcm_homogeneity_d1": -0.4,
            # --- secondary: size, downweighted and named for what it is ---
            "z_territory_area_um2": 0.4,
            "z_nn_distance_um": 0.4,            # nuclear spacing, stated plainly
            "z_circularity": 0.3,               # ballooned cells round up
        }
    )

    # --- necessary conditions ---
    # Vetoes, not score contributions. Ballooning is *enlargement* with
    # rarefaction, so a cell that is not enlarged and rounded cannot be a
    # candidate however pale and heterogeneous its cytoplasm is.
    #
    # This split is what makes the noisy territory area usable. As a weighted
    # score term, area noise competes directly with the pallor signal; as a
    # coarse gate it only has to answer "bigger than its neighbours, yes or
    # no", which it can do reliably. Ranking within the survivors is then led
    # by the cytoplasm features, which are not affected by where exactly the
    # territory boundary fell.
    #
    # Without these, the top of the ranking fills with hepatocytes compressed
    # into thin slivers between fat droplets: their remaining cytoplasm is
    # genuinely pale and genuinely heterogeneous (it is a rim against a white
    # void), so it scores high on every cytoplasm axis while being SMALLER
    # than its neighbours -- the opposite of ballooning.
    # 2.0x, from a pathologist. Reviewing our top candidates on 2026-08-16 she
    # rejected every one, with the reason: "they're not enlarged 2-3x". This
    # threshold was 1.3x, and measured across our top-ranked cells the median
    # was 1.54x with only 22% reaching 2.0x -- so roughly four fifths of what
    # we proposed could never have met the criterion. Not a tuning nicety: the
    # detector was answering a different question from the one being asked.
    min_area_ratio: float = 2.0
    # Deliberately below the ~1.5-2x a florid balloon shows: the territory is
    # a spacing estimate and carries noise, and a candidate generator should
    # over-propose and let the pathologist reject.
    min_solidity: float = 0.80      # a sliver is ragged; a balloon is convex
    max_eccentricity: float = 0.85  # a sliver is elongated; a balloon is round

    # A cell many times its local median is a segmentation merge, not a cell.
    # Tightened 6.0 -> 4.0. The same review flagged one candidate as "not even
    # a single cell" -- a watershed merge of neighbouring hepatocytes passing as
    # one huge cell. A genuine balloon tops out around 3x its neighbours, so
    # anything past 4x is a segmentation artefact rather than a rare finding.
    max_area_ratio: float = 4.0
    # Above this the "cytoplasm" is a fat void, not pale cytoplasm -- keeps
    # macrovesicular droplets from being ranked as balloons.
    max_cyto_gray_mean: float = 205.0
    # Ballooned hepatocytes keep their nucleus, often displaced against the
    # membrane. Requiring one separates them from empty space and from voids.
    require_nucleus: bool = True

    # --- what the pathologist actually receives ---
    # Top-scoring cells per slide.
    # The top band is a percentile WINDOW, not "the first N by rank". See
    # `rank.stratified_select` for why: as an absolute count it silently ate
    # the bands below it whenever the scorable population was small.
    top_percentile_min: float = 80.0
    top_n: int = 150
    # Cells drawn from the middle of the distribution, and from the bottom.
    # Without these the review set is all extremes, the pathologist never sees
    # a near-miss, and the classifier trained on the confirmed set learns a
    # decision boundary that was never actually shown to a human.
    mid_n: int = 75
    low_n: int = 75
    # The middle band is sampled from this percentile range of the score.
    mid_percentile_range: tuple[float, float] = (40.0, 60.0)
    # The low band comes from below this percentile.
    low_percentile_max: float = 20.0

    # Candidates drawn from cells the detector VETOED, so the vetoes are
    # reviewed too. The vetoes discard ~70% of segmented cells on this cohort
    # -- `not_enlarged` alone accounts for most of it -- and nothing currently
    # checks whether that 70% contains real ballooning. Set to 0 only once it
    # has been checked at least once.
    vetoed_n: int = 40

    seed: int = 0

    # Cap per tile, applied before slide-level ranking. Ballooning is rare; a
    # tile proposing dozens is misconfigured, not informative.
    max_candidates_per_tile: int = 20


@dataclass
class CropConfig:
    """The image each candidate is reviewed from."""

    # Padding around the cell's bounding box, in microns. The judgment is
    # comparative, so the crop must contain the neighbours it is being called
    # against -- ~40 um on each side puts roughly two rings of normal
    # hepatocytes in frame.
    padding_um: float = 40.0

    # Draw the segmented cell boundary on the crop. The unmarked crop is always
    # saved alongside, because an outline is itself a claim the pathologist may
    # want to judge without.
    draw_outline: bool = True
    outline_color: tuple[int, int, int] = (255, 210, 0)
    nucleus_color: tuple[int, int, int] = (0, 140, 255)
    outline_thickness: int = 2
    draw_nucleus: bool = True

    # Save the same crop unmarked, so the reviewer can look without the
    # detector's opinion drawn on top.
    save_unmarked: bool = True

    # Contact sheet for the operator's own sanity check -- not for review.
    contact_sheet_n: int = 60
    contact_sheet_cols: int = 10
    contact_sheet_thumb_px: int = 220


@dataclass
class QCConfig:
    """Segmentation overlays, for confirming the cells are real before any
    feature is built on them."""

    # Full-resolution per-tile overlays. Downscaled contact sheets hide exactly
    # the boundary errors this pass exists to catch.
    tiles: int = 8
    save_native_resolution: bool = True
    nucleus_color: tuple[int, int, int] = (0, 140, 255)
    cell_color: tuple[int, int, int] = (255, 210, 0)
    hepatocyte_color: tuple[int, int, int] = (0, 220, 120)
    thickness: int = 1
    # Also write a side-by-side of raw H&E next to the overlay, which is the
    # only honest way to judge whether a boundary is real.
    side_by_side: bool = True


@dataclass
class BallooningConfig:
    """Candidate generation for ballooned hepatocytes."""

    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    crops: CropConfig = field(default_factory=CropConfig)
    qc: QCConfig = field(default_factory=QCConfig)
