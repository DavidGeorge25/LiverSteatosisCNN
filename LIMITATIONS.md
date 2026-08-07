# Limitations of the pseudo-labeling pipeline

Written to be usable as a draft Limitations section. Every claim here is backed
by an artifact in `outputs/`; where a number is quoted, the file is named.

The pipeline is a **weakly supervised** design: pseudo-labels are generated
programmatically because no pixel-level annotations exist for this dataset and
none are expected. That is the premise, not a defect. But it bounds what can be
claimed, and the bounds below should be stated rather than discovered by a
reviewer.

---

## 1. No annotation-based accuracy metric exists

**What this means.** There is no reference segmentation, so Dice, IoU, and
per-droplet precision/recall cannot be computed. No statement of the form "the
pseudo-labels are X% accurate" is supported by anything in this repository.

**Why it bites harder than it looks.** Parameter selection has no ground truth
to optimise against. A 360-cell parameter sweep over 14 steatotic and 37
fat-free slides (`outputs/sweep_ecc_14v37_per_slide.csv`) found that **all 360
cells separate the two cohorts perfectly** — slide-level AUC = 1.000, in-sample
and on every one of 200 held-out splits (`outputs/holdout_repeats.csv`). Yet
those same cells report mean fat fraction anywhere from **6.82% to 10.91%**
(1.6x), and the weakest slide from 1.67% to 5.16% (3.1x).

So the negative control confirms the pipeline does not invent fat on fat-free
tissue, but it **cannot identify which setting measures fat correctly**. The
separation metric is saturated and carries no information for choosing
parameters.

**Consequence for reporting.** Absolute fat fractions from this pipeline are
uncalibrated. Relative comparisons between slides processed with identical
parameters are far better supported than any absolute value. A reported
percentage should carry the parameter-sensitivity range alongside it.

**Mitigation applied.** Rather than pick one setting and conceal the choice,
the exported labels are a consensus of 9 members spanning the plausible
parameter region, with a per-pixel confidence map (`steatosis/ensemble.py`).
This does not create ground truth; it makes the uncertainty explicit and
usable as a loss weight.

**What would remove this limitation.** 20–40 annotated tiles. A ready-to-
annotate, stratified package is at `outputs/annotation_set_v1/` (40 tiles,
0.31%–22.2% predicted fat, prefilled masks, written criteria). This is the
single highest-value missing input.

## 2. Specificity is measured; sensitivity is not

The CCl4 cohort supports a false-positive estimate because tissue known to be
fat-free should yield nothing. There is **no corresponding way to measure false
negatives.** A droplet class that the pipeline systematically misses — small,
pale-rimmed, or merged into neighbours — would be invisible to every check
currently in place, including the negative control.

Sensitivity is therefore assumed from visual inspection only.

## 3. The negative control's validity is assumed, not verified

Treating the CCl4 cohort as fat-free rests on the model type (carbon
tetrachloride is a fibrosis/centrilobular-necrosis model) and on visual review
of QC overlays. **No study metadata was available** to confirm diet, treatment
arm, or timepoint for these animals.

If any of those animals also received a steatogenic diet, the false-positive
estimates are inflated and the tuning derived from them is biased. Since the
entire specificity argument rests on this cohort, the assumption is
load-bearing and should be confirmed before publication.

## 4. Small number of biological replicates

14 steatotic and 37 fat-free slides. Tile counts are large (42,335 and 41,040)
but tiles within a slide are not independent — they share staining, scanner
state, and tissue.

Resampling **slides** (the correct independent unit) gives a wide interval:
mean macro fat **9.37%, 95% CI [7.18, 11.40]**
(`outputs/bootstrap_shipped_config.csv`). Any per-group biological claim from
14 animals will be underpowered.

## 5. Parameter selection used all available slides

Parameters were chosen by scoring every cell on every slide, then reporting the
winner's score on those same slides — circular in principle.

Held-out analysis (`steatosis/validate.py`) shows the practical impact is
small: selection is stable (one cell wins 72% of 200 splits; only 8 of 360 are
ever selected) and held-out AUC is 1.000 in every split. **But** the `margin`
metric (min positive − max negative) is a min/max statistic and therefore
sample-size dependent; held-out margin came out *higher* than tune margin
(4.75 vs 3.72), which is an artifact of fewer slides in the test set, not
evidence of generalisation. **Do not quote `margin` as a generalisation
estimate.** Report AUC or a mean-based effect size.

Note also that the shipped configuration (`configs/tuned.yaml`) ranks 9th of
360 by margin and was never the top pick in any split. The differences are
within noise — which is exactly the point of limitation 1.

## 6. A systematic false positive remains

Pale or rarefied cytoplasm — glycogen accumulation, hydropic/ballooning change
— is the dominant residual artifact. It was diagnosed directly: on the fat-free
cohort the detector fires on angular cytoplasmic clearing rather than on
droplets (`outputs/R26-122-10_HE_79/qc/tiles/`).

This error is **systematic, not random**, which matters for the weakly
supervised premise. Random label noise is averaged out by a CNN trained over
many tiles; a reproducible error is learned faithfully. A model trained on
these labels will learn some amount of "pale cytoplasm = fat."

Consensus voting reduces but does not eliminate it (see §9 for the measured
reduction).

## 7. Droplet merging is not addressed

Confluent droplets merge into a single connected component. This inflates mean
droplet size, deflates droplet counts, and distorts the size distribution.
Area-based fat fraction is affected less than count-based statistics.
No watershed or similar separation step is implemented.

## 8. Tile-border effects

12.7–19.0% of detected fat area touches a tile edge. `exclude_border` is
disabled because enabling it removes legitimate droplets clipped by tiling;
overlapping tiles (`--stride` < `--tile-size`) are supported but were not used
for the reported runs. Droplets spanning tile boundaries are therefore counted
as two partial droplets.

## 9. Microvesicular steatosis is measured but not labeled

Across 14 steatotic and 37 fat-free slides, the microvesicular channel
separates the cohorts only ~1.4–1.6:1, versus ~55:1 for macrovesicular; before
tuning it read *higher* on fat-free tissue (0.85:1). It carries no usable
signal in its current definition and is excluded from exported labels.

**Scope consequence:** this pipeline quantifies **macrovesicular steatosis
only.** Microvesicular change is biologically relevant in MASH, and its
exclusion should be stated explicitly in any methods description rather than
left implicit.

## 10. Coarse tissue mask within tiles

Tissue detection runs at pyramid level 2 (16x downsample). A 512 px tile covers
only ~32x32 mask pixels, so the tissue boundary inside a tile is block-
resolution. This is adequate for excluding background from edge tiles but does
not finely delineate the tissue edge, and the tissue-edge gap remains a
false-positive source in edge tiles.

## 11. Single stain, scanner, and site

All 51 slides are Aperio SVS, 20x, 0.4953 µm/px, H&E, from one source. There is
**no evidence of robustness** to other scanners, magnifications, stain
protocols, or laboratories. Thresholds specified in absolute grayscale units
(`white_threshold`) are the most likely component to fail under stain
variation; the physical-unit filters (µm²) should transfer better.

## 12. One slide is unreadable

`R26-122-14_HE.svs` is truncated — its TIFF header points to an image directory
at byte 258,784,520 in a 258,503,100-byte file. It is excluded from all
analyses, so the negative cohort is 37 of 38 slides.

## 13. The confidence map is informative but weak and uncalibrated

Per-pixel confidence is the fraction of ensemble members voting for a pixel.
Measured across all 51 slides (`outputs/agreement_analysis.csv`, 20 tiles each):

| | steatotic (n=14) | fat-free (n=37) |
| --- | --- | --- |
| unanimous share of flagged area | 57.6% | 19.3% |
| contested share of exported label | 23.9% | 49.5% |

So agreement does carry signal — half the false-positive label area is
contested versus a quarter on genuinely steatotic tissue — but the effect is
**modest: a 3.0:1 ratio, slide-level AUC 0.973 on unanimity alone.** It is not
a substitute for the fat-fraction measurement itself (AUC 1.000).

Consensus voting versus a single configuration: **false positives on fat-free
tissue fall 9.8%** at a cost of **2.1% of signal** on steatotic tissue. A
favourable but small trade — this is a refinement, not a fix for §6.

The map is **not a calibrated probability**. A confidence of 0.5 does not mean
a 50% chance the pixel is fat. Use it for relative loss weighting only.

*Caution on sampling:* an early 4-slide pilot put fat-free unanimity at 3.2%,
implying an 18:1 ratio. The full 51-slide measurement gives 19.3% and 3.0:1.
The pilot was badly optimistic; the full-cohort numbers above are the ones to
cite.

---

## What the design does support

- **Relative ranking** of slides by macrovesicular steatosis burden, with
  perfect cohort-level separation that survives held-out validation.
- **Specificity**: a measured false-positive rate on tissue known to be
  fat-free (0.17% mean, worst slide 0.44%).
- **Confidence-weighted weak supervision**: labels plus an explicit per-pixel
  uncertainty signal (modest but measurable, §13), rather than pseudo-labels
  presented as ground truth.
- **Reproducibility**: every output directory carries the resolved config,
  seeds, package versions and git state (`steatosis/provenance.py`).

## Priority order for removing these limitations

1. Annotate the 40-tile package → removes §1, quantifies §2 and §6.
2. Confirm the CCl4 study design → removes §3.
3. Pathologist steatosis grades (NAS 0–3) per animal → orthogonal validation of
   §1 without pixel work.
4. Watershed separation → removes §7.
5. More biological replicates → narrows §4.
