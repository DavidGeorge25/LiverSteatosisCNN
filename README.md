# Hepatic steatosis pseudo-labeling pipeline

Programmatic pseudo-labeling of fat droplets in H&E whole slide images of mouse
liver (MASH model), as weak supervision for a downstream U-Net. Phase 1 of the
project: classical CV mask generation, no model training.

## Status

| Stage | Status |
| --- | --- |
| 1. Slide loading + tissue detection | **done, verified on 259 slides** |
| 2. Level-0 tiling | **done** |
| 3. Fat droplet pseudo-labeling | **done, tuned against a negative control** |
| 4. Visual QC / contact sheets | **done** |
| 5. Slide-level summary + CSV | **done** |
| 6. Cohort survey + parameter sweep | **done** |
| 7. Pseudo-label export for training | not started (one flag: drop `--no-save-tiles`) |
| 8. Cross-batch validation, 9 staining runs | **done, unretuned** — [docs/STEATOSIS_NINE_BATCHES.md](docs/STEATOSIS_NINE_BATCHES.md) |
| 9. U-Net training on the pseudo-labels | **first fold trained** — holds the CCl4 floor, reproduces the diet validation at AUC 1.000 — [docs/STEATOSIS_TRAINING.md](docs/STEATOSIS_TRAINING.md) §7 |

## Setup

The native OpenSlide library is a prerequisite (the Python package is only a
binding):

```bash
brew install openslide
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Data

| Folder | Cohort | n | Role |
| --- | --- | --- | --- |
| `data/R25-264_MASH_HE/` | R25-264, MASH model | 14 | positives — fat expected |
| `data/2026-04-20_CCl4_HE/` | R26-122, CCl4 model | 38 | negative control — fibrosis, not steatosis |

All 52 slides are Aperio SVS, JPEG-compressed, 3 pyramid levels (1× / 4× / 16×),
0.4953 µm/px, 20× objective. Level-0 dimensions vary per slide and are read from
each file.

**CCl4 is not a steatosis cohort** and is not meant to be one — carbon
tetrachloride is a fibrosis / centrilobular-necrosis model. It earns its keep as
a *known-negative*: any fat detected there is by definition a false positive,
which is what makes quantitative tuning possible at all.

**One CCl4 file is truncated:** `R26-122-14_HE.svs` — its TIFF header points to
the first image directory at byte 258,784,520 but the file is 258,503,100 bytes,
~281 KB short. OpenSlide raises `OpenSlideUnsupportedFormatError`. It needs
re-downloading. Every batch/survey command skips it and reports it rather than
dying, so the other 37 are unaffected.

## Usage

```bash
# slide metadata
.venv/bin/python -m mashpath.cli info --slide data/R25-264_MASH_HE/R25-264-1.svs

# tissue detection + QC overlay
.venv/bin/python -m mashpath.cli tissue \
    --slide data/R25-264_MASH_HE/R25-264-1.svs --config configs/tuned.yaml

# full pipeline on one slide: tissue -> tiles -> fat -> contact sheet + CSV
.venv/bin/python -m mashpath.cli steatosis \
    --slide data/R25-264_MASH_HE/R25-264-1.svs \
    --config configs/tuned.yaml --limit 100     # 0/omitted = whole slide

# a whole folder, 5 slides at a time
.venv/bin/python -m mashpath.cli batch --slide-dir data/R25-264_MASH_HE \
    --config configs/tuned.yaml --workers 5 --no-save-tiles
```

`run` writes `results/tiles.csv`, `qc/contact_sheet.png`, `qc/tiles/*_qc.png`,
and (unless `--no-save-tiles`) `tiles/*.png` + `masks/*_mask.png` — the latter
two *are* the pseudo-label dataset. Outputs land in `outputs/<slide-name>/`,
alongside `config_used.yaml`, the fully resolved parameters that produced them.
Under `--workers > 1` each slide's console output goes to `run.log` in its own
output directory instead of interleaving on the terminal.

### Cohort survey and parameter sweep

A full run reads every tissue tile (2,000–4,600 per slide, minutes each). That is
right for producing labels and wrong for answering "does this cohort have fat in
it" or "which threshold separates positives from negatives" — both are estimation
problems that a random sample answers in seconds.

```bash
# per-slide fat estimate from 60 random tiles each, 52 slides in ~16 s
.venv/bin/python -m mashpath.cli survey \
    --group MASH=data/R25-264_MASH_HE --group CCl4=data/2026-04-20_CCl4_HE \
    --config configs/tuned.yaml --tiles 60 --workers 8

# score a parameter grid on how well it separates the two cohorts
.venv/bin/python -m mashpath.cli sweep \
    --group MASH=data/R25-264_MASH_HE --group CCl4=data/2026-04-20_CCl4_HE \
    --config configs/tuned.yaml --tiles 60 --workers 8 \
    --white 200,210,220 --circ 0.55,0.60,0.65 --sol 0.85,0.90 \
    --min-area 20,40,60,80 --ecc 0.70,0.75,0.80,0.85,1.0
```

Sampling is seeded per slide name, so a slide's sample is stable across runs and
independent of which other slides are in the folder. Only `white_threshold`
changes what gets segmented; every other axis is a filter on already-measured
components, so a 360-cell grid over 52 slides costs 64 s — barely more than the
3 threshold values it actually has to re-segment for.

**The survey is accurate enough to tune on.** Against the full 14-slide run,
60-tile estimates have a mean absolute error of 3.2% relative (max 14.5%) and a
Spearman rank correlation of 0.978.

## Tuning

Every threshold lives in `configs/*.yaml`; CLI flags override the YAML file,
which overrides the built-in defaults in `mashpath/features/steatosis/config.py`.
The resolved
config is written next to the outputs, so any result directory records exactly
which parameters produced it.

`configs/tuned.yaml` is the current recommendation and carries its full
derivation in comments. It was retuned (revision 2) once the cohort grew from 5
to 14 MASH slides.

### Why revision 1 had to be redone

Revision 1 was tuned on 5 MASH slides that were all moderate-to-severe
(7.7–15.2% macro). The 9 slides added on 2026-08-06 span 3.7–12.4%, and on the
mild end revision 1 was visibly **missing large, obvious, round droplets** —
`white_threshold` 220 plus `solidity_min` 0.90 rejected any droplet with a
slightly ragged or scalloped border. Severe slides have enough textbook-clean
droplets that this never showed up. Tuning a detector only on severe cases hides
exactly the failure that matters for grading mild ones.

| metric (14 MASH vs 37 CCl4, 60 tiles each) | rev 1 | rev 2 |
| --- | --- | --- |
| MASH macro, mean | 8.87% | **9.37%** |
| MASH macro, weakest slide | 2.48% | **3.69%** |
| CCl4 macro, mean (false positive) | 0.239% | **0.17%** |
| CCl4 macro, worst slide | 0.57% | **0.34%** |
| worst-case separation | 4.3:1 | **10.8:1** |

Revision 2 beats revision 1 on all four metrics at once — it recovers *more*
real signal while producing *less* false positive. 32 of the 360 grid cells did;
this is the one among them that retains the most real signal.

Those figures come from the 60-tile survey, which is what the grid was scored
on. On complete runs the separation is a little less flattering than the sample
suggested — worst case 8.3:1 rather than 10.8:1 — because sampled per-slide
minima and maxima are less extreme than true ones. See the next section for the
full-run numbers, which are the ones to quote.

### The substantive change: an explicit elongation filter

Elongated vessel and sinusoidal lumens were the dominant false positive, and
**circularity does not catch them**. `4πA/P²` responds to boundary raggedness,
not to elongation — a smooth 2.5:1 lumen scores ~0.87 and sails through. Every
attempt to squeeze them out by tightening circularity instead removed real fat
just as fast.

`max_eccentricity` (ellipse eccentricity, 0 = circle, 1 = line) targets them
directly. At 0.80 it costs 6–10% of MASH signal and removes 40–64% of CCl4 false
positives — and the MASH cost is uniform across severity, so relative slide
ranking is preserved:

| slide | no ecc filter | ecc ≤ 0.80 | lost |
| --- | --- | --- | --- |
| R25-264-12 (severe) | 15.31% | 14.24% | 7.0% |
| R25-264-11 (severe) | 16.18% | 15.23% | 5.9% |
| R25-264-33 (mild) | 4.95% | 4.44% | 10.4% |
| R25-264-34 (mild) | 3.45% | 3.16% | 8.5% |
| R26-122-10 (CCl4) | 0.66% | 0.40% | **39.8%** |
| R26-122-20 (CCl4) | 0.59% | 0.21% | **64.3%** |

The other change worth knowing: `min_area_um2` 20 → 40. False positives cluster
just above the old floor (CCl4 median component 25–28 µm²) while real droplets
sit at 56–76 µm². Unlike raising `white_threshold`, a size floor is a whole-
component decision — it does not erode the boundaries of the droplets it keeps,
which matters because those boundaries are the labels the U-Net learns.

That is also why `white_threshold` went *back down* 220 → 210: raising it shrinks
every accepted droplet and biases area down. With size and eccentricity doing the
rejection work, it is no longer needed for that job.

### Tested and rejected

An "unstained-ness" criterion (low HSV saturation) to separate true fat voids
from pale cytoplasm. Measured, and it does not work: CCl4 false positives have
median interior saturation 14.0, *lower* than real droplets on the mild MASH
slides at 15.3. Interior color cannot separate them.

**Watershed splitting of touching droplets** — implemented, measured across 85
slides, and **left off**. It is `fat.watershed` in
[configs/watershed.yaml](configs/watershed.yaml); the default path is unchanged
and byte-identical (`tests/regression_steatosis.py`).

It does what it was built to do. Merged pairs come apart: on MASH the droplet
count rises 20% while mean droplet size falls 4%, and all 40 slides gain area.
The problem is what else comes apart.

| setting | MASH macro | CCl4 macro | CCl4 worst | MASH:CCl4 | chow max | NASH min | diet gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **off** | 8.02% | **0.165%** | 0.342% | **48.6:1** | 0.20% | 7.58% | 37.5x |
| depth 1.5 µm | 9.22% (+14.9%) | 0.267% (+61.8%) | 0.499% | 34.5:1 | 0.29% | 9.49% | 32.5x |
| depth 2.5 µm | 8.72% (+8.7%) | 0.217% (+31.5%) | 0.457% | 40.1:1 | 0.21% | 8.70% | **40.8x** |
| depth 1.5, size floor 160 µm² | 9.07% (+13.1%) | 0.264% (+59.7%) | 0.495% | 34.4:1 | 0.29% | 9.21% | 31.7x |

**The false-positive floor rises about four times faster than the signal, at
every setting.** Depth is the only knob that bites — doubling the size floor
changes almost nothing — and even at its most conservative the trade is 8.7%
more MASH signal for 31.5% more CCl4. Worst-case CCl4 reaches 0.457-0.499%,
against the 0.5% below which a slide-level call is treated as zero.

The tell is CCl4's **mean droplet size going up** (173 → 190-205 µm²) while its
count goes up 15-31%. On real fat that ratio inverts, which is what MASH does.
Rising size and rising count together is not merged droplets separating; it is
elongated sinusoidal and vessel lumens being cut into rounder pieces that then
pass `max_eccentricity`. The splitter is rescuing the exact false positives the
eccentricity filter was added to remove — and a fibrosis model, with its
abnormal vasculature, is unusually rich in them.

**One result points the other way and is the reason this is "left off" rather
than "rejected".** At depth 2.5 the *chow* negative — four untreated livers,
normal vasculature — barely moves (0.202 → 0.213%) while NASH gains 15%, so the
diet gap *improves*, 37.5x → 40.8x. Whether the recovered area is real fat or
lumen is exactly the question this A/B cannot answer, because both cohorts are
scored by the same detector that is under test. **40 annotated tiles would
settle it in an afternoon** — see [LIMITATIONS.md](LIMITATIONS.md) §1.

## Results — 14 MASH slides, full runs, `configs/tuned.yaml` rev 2

42,335 tiles, 2,722 mm² of tissue, 585 s wall clock at `--workers 5`.

| Slide | Tiles | Tissue mm² | Macro n | Macro % | Micro % | Total % | Border % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R25-264-1 | 2487 | 159.92 | 91140 | 14.34 | 2.08 | 16.42 | 16.31 |
| R25-264-3 | 4554 | 293.02 | 170411 | 11.27 | 2.14 | 13.41 | 13.38 |
| R25-264-10 | 2472 | 158.87 | 69651 | 6.71 | 1.27 | 7.99 | 11.14 |
| R25-264-11 | 3345 | 214.66 | 151307 | 14.68 | 1.61 | 16.29 | 13.62 |
| R25-264-12 | 3466 | 223.28 | 148890 | 14.92 | 2.18 | 17.10 | 14.36 |
| R25-264-13 | 4145 | 266.27 | 152242 | 12.13 | 1.77 | 13.90 | 14.28 |
| R25-264-30 | 3995 | 256.49 | 153633 | 12.40 | 2.23 | 14.63 | 14.28 |
| R25-264-31 | 2402 | 153.98 | 54591 | 5.99 | 1.38 | 7.37 | 12.13 |
| R25-264-32 | 3658 | 235.35 | 130927 | 13.07 | 1.82 | 14.90 | 14.82 |
| R25-264-33 | 2077 | 133.69 | 46175 | 5.03 | 0.97 | 6.00 | 11.46 |
| R25-264-34 | 2449 | 157.69 | 39056 | 3.66 | 1.95 | 5.61 | 11.02 |
| R25-264-35 | 2513 | 161.66 | 58854 | 5.50 | 1.00 | 6.50 | 11.46 |
| R25-264-36 | 2389 | 153.93 | 45361 | 3.95 | 1.08 | 5.02 | 10.87 |
| R25-264-37 | 2383 | 153.30 | 57469 | 8.98 | 1.02 | 10.00 | 15.49 |
| **COHORT** | 42335 | 2722.11 | 1369707 | **10.17** | **1.69** | **11.86** | — |

Cohort row is tissue-area-weighted. Macro spans 3.66–14.92%, a genuine 4×
severity range — the 9 new slides roughly doubled the cohort and extended it well
below the old floor of 6.7%.

Micro % is **measured but excluded from exported labels**; see open problems.

## Results — nine staining batches, 259 slides, nothing retuned

Full record: **[docs/STEATOSIS_NINE_BATCHES.md](docs/STEATOSIS_NINE_BATCHES.md)**.

The pipeline was tuned on 2 staining runs. On 2026-08-29 it was run, unchanged,
on all 9 — 259 slides, 60 seeded tiles each.

**It separates known diet with stain held fixed.** `R22-354` is the only
accession whose filenames record diet and whose slides were all stained in one
run. Four chow against four NASH, genotype balanced, detector never previously
run on any of them: **AUC 1.000, p = 0.0286, lowest NASH 37.5x the highest
chow** (7.58% against 0.20%).
Chow reads 0.07-0.20%, inside the CCl4 false-positive range of 0.04-0.34% — two
unrelated kinds of negative landing in the same place.

| batch | n | macro mean | range | <1% | >5% |
| --- | ---: | ---: | --- | ---: | ---: |
| 2026-04-20_CCl4 (control) | 37 | **0.17%** | 0.04-0.34 | 37 | 0 |
| 2025-07-09_Evelyn | 23 | 1.30% | 0.44-2.36 | 7 | 0 |
| 2025-04-29 | 43 | 2.18% | 0.02-14.38 | 23 | 6 |
| 2025-03-07_Jaya | 29 | 3.71% | 0.11-8.87 | 5 | 9 |
| 2025-03-28_Celina | 8 | 4.75% | 0.07-10.46 | 4 | 4 |
| 2025-09-24 | 25 | 6.69% | 1.37-13.42 | 0 | 19 |
| 2025-08-25 (MASH) | 40 | 8.02% | 0.20-14.82 | 3 | 27 |
| P273-2026-02-13 | 30 | 8.07% | 0.20-12.45 | 1 | 23 |
| 2025-11-14 | 24 | 8.64% | 0.47-16.01 | 1 | 21 |

`white_threshold` is an absolute grey level of 210 and no batch degenerates
under it. The CCl4 floor reproduces at 0.165% against the 0.173% below. And
severity now varies **within** batch — six of nine batches hold both a sub-1%
and an over-5% slide — so "predicts fat" and "recognises the staining run" are
finally different functions.

Controlled for the tissue mask: rerunning all 259 slides with per-slide Otsu
replaced by one fixed threshold moves the estimate by a median 0.6%
(Spearman 0.9966) and changes no conclusion.

Absolute fat fractions are **still uncalibrated** — see
[LIMITATIONS.md](LIMITATIONS.md) §1, which this does not touch.

## Results — cohort separation, both cohorts run in full

Not a sample: every tissue tile of all 51 readable slides, 83,375 tiles.

| | MASH (n=14) | CCl4 (n=37) |
| --- | --- | --- |
| tiles | 42,335 | 41,040 |
| macro %, mean | 9.47 | **0.173** |
| macro %, min | 3.66 | 0.09 |
| macro %, max | 14.92 | **0.44** |
| micro %, mean | 1.61 | 1.12 |

- **Every MASH slide reads above every CCl4 slide.** Weakest positive 3.66% vs
  worst false positive 0.44% — a 3.22 point gap, 8.3:1 worst case.
- Cohort means separate 54.9:1 on macro.
- Micro separates 1.4:1, i.e. essentially not at all, which is why it stays out
  of the labels.

`outputs/batch_MASH14_tuned2.csv`, `outputs/batch_CCl4_tuned2.csv`.

## Results — tissue detection

Same parameters for every slide; Otsu adapts the saturation cutoff per slide,
which matters because staining intensity varies.

| Cohort | n | Otsu cutoff range | Coverage | Components |
| --- | --- | --- | --- | --- |
| MASH R25-264 | 14 | 43–65 | 44.8–59.5% | 1–4 |
| CCl4 R26-122 | 37 | 50–64 | — | 1–4 |

Multi-component slides are genuine detached tissue fragments, not debris.
Runtime is ~0.2 s per slide after the level-2 read.

## Known open problems

**1. Microvesicular is still a noise floor, not a signal.** Across 14 MASH and
37 CCl4 slides it reads 0.74% vs 0.53% — a 1.4:1 separation, against 55:1 for
macro — and the single highest micro value in either cohort belongs to a fat-free
CCl4 slide. It stays measured and stays out of the exported labels
(`include_microvesicular: false`). Note that rev 2 widened the micro band to
3–40 µm² (it is defined as everything below `min_area_um2`), so rev 2 micro
numbers are not comparable to rev 1's.

**2. Adjacent droplets merge** into single components in dense regions. This now
interacts with the eccentricity filter: a merged pair is elongated, so instead of
inflating droplet size it gets rejected outright. That is most of the 6–7% cost
on severe slides. The real fix is watershed splitting of touching droplets, which
would recover that signal *and* fix the droplet-count statistics.

**3. `exclude_border` tradeoff.** 11–16% of detected fat area touches a tile edge
(down from 19% under rev 1, since the size floor drops small edge fragments). It
is OFF and should stay off; overlapping tiles via `--stride < --tile-size` is the
better fix and is already wired up.

**4. Whole-slide false-positive floor.** The CCl4 cohort sets it at 0.17% mean /
0.44% worst case (full runs). Any slide-level call below ~0.5% should be treated as
indistinguishable from zero.

## Design notes

**Physical units.** Filters are specified in microns / square microns, not
pixels, and converted using the slide's own MPP. Changing the detection pyramid
level therefore does not change what a filter means.

**Memory.** Level 0 (up to 53784 × 40942) is never read in full. Tissue detection
runs at level 2 (16× downsample, ~17 MB as RGB) and the mask is upsampled on
demand; tiles stream one at a time via `read_region`. Peak RSS is ~440 MB per
slide, so `--workers 5` fits comfortably.

**Tissue detection.** Background is off-white glass, tissue is pink/purple — a
difference that lives almost entirely in HSV saturation. Otsu on the saturation
channel picks the cutoff per slide. Then: morphological close (bridge
intra-tissue gaps) → open (drop thin bridges) → fill interior holes → remove
small objects (dust and debris outside the section).

Verified: it does not under-include steatotic tissue (a 4× more permissive
threshold adds only 0.36% area, all ragged section-edge fringe), and hole-filling
does not swallow fat (all filled holes are vessel lumens, median 2,858 µm², max
38,379 µm², well under the 200,000 µm² limit). Filling puts vessel lumens
*inside* the tissue mask, so they reach the fat detector and must be rejected on
shape — which is what `max_eccentricity` is for.

`TissueMask.tissue_fraction(x, y, w, h)` answers tile-level tissue queries in
level-0 coordinates straight from the low-res mask, so tiling skips background
tiles without reading level-0 pixels.

## Pre-training output: `outputs/dataset_v1/`

The deliverable of the preprocessing stage. 7,650 tiles from 51 slides, 4.2 GB.

| cohort | split | tiles | slides |
| --- | --- | --- | --- |
| positive (steatotic) | train / val / test | 1200 / 450 / 450 | 8 / 3 / 3 |
| negative (fat-free) | train / val / test | 3450 / 1050 / 1050 | 23 / 7 / 7 |

Per tile: `images/` (RGB), `labels/` (consensus mask), `confidence/` (per-pixel
vote fraction, 0-255). Plus `manifest.csv`, `splits.json`, `failures.json`,
`ensemble.json`, `config_used.yaml`, `provenance.json`.

Three properties worth knowing before training on it:

- **Labels are an ensemble consensus, not one threshold.** Nine parameter
  settings vote per pixel; the label is the majority. Mean 26% of each
  positive tile's label area is contested rather than unanimous — that is what
  `confidence/` encodes, and a weakly supervised loss should weight by it
  rather than treat every pseudo-label pixel as certain.
- **Splits are by slide and stratified by severity.** Never by tile: adjacent
  tiles share staining and often the same hepatocytes across their border.
  Stratification is not cosmetic here — a plain random split of 14 slides put
  all three fattiest slides in test (13.6-15.1%) against a 3.6-12.8% train set.
- **67% of negative tiles carry an empty mask.** Deliberate: a model that has
  never seen liver without fat will invent droplets on it.

```bash
.venv/bin/python -c "
import glob
from mashpath.config import MashConfig as Config
from mashpath.features.steatosis.dataset import build_dataset
cfg = Config.from_yaml('configs/tuned.yaml')
build_dataset({'positive': sorted(glob.glob('data/R25-264_MASH_HE/*.svs')),
               'negative': sorted(glob.glob('data/2026-04-20_CCl4_HE/*.svs'))},
              cfg, 'outputs/dataset_v1', max_tiles_per_slide=150, seed=0)"
```

**Read [LIMITATIONS.md](LIMITATIONS.md) before reporting any number from this
dataset.** The short version: absolute fat fractions are uncalibrated (all 360
parameter cells separate the cohorts perfectly while reporting 6.82-10.91%
mean fat), relative comparisons under identical parameters are well supported,
and the residual false positive on pale cytoplasm is systematic rather than
random — so it will be learned, not averaged away.

## Layout

```
mashpath/
  config.py            root config: core sections + one block per feature
  cli.py               argparse entry point for every feature
  core/                everything shared, imported by all three features
    config.py          dataclass config tree, YAML load/save, dotted overrides
    slide.py           openslide wrapper; keeps MPP attached, never reads L0 whole
    tissue.py          tissue detection -> TissueMask (mm^2, tile-fraction queries)
    tiling.py          level-0 tiling over tissue, streaming
    segmentation.py    the shared StarDist layer: one model load, padded reads,
                       percentile normalization, upscale
    viz.py / io.py     rendering primitives; output layout and index writing
    provenance.py      records versions and parameters next to every result
  features/
    steatosis/         fat droplets -- pseudo-labelled, no review needed
    ballooning/        hepatocyte ballooning -- candidates for review
    inflammation/      lobular inflammation -- stages 1-2 only, see detect.py
  review/
    candidates.py      the feature-agnostic Candidate record
    manifest.py        THE manifest schema: one writer, one reader
    verdicts.py        append-only verdict store; inter/intra-rater kappa
    app.py             the local review web app (stdlib only, localhost)
    export.py          generic crop + manifest exporter
    tileset.py         TILE-level labelling sets: frame, stratified draw, package
    tiles_app.py       the tile labelling app: zoom, context, y/n/unsure
configs/
  core.yaml        shared: slide, tissue, tiling, qc, review
  steatosis.yaml   } layer one of these over core.yaml
  ballooning.yaml  }
  inflammation.yaml}
  ballooning_tiles.yaml  the tile-labelling draw (not the detector)
  default.yaml / tuned.yaml   pre-restructure single-file configs, still load
cluster/           SLURM job scripts and Alliance setup for Fir
docs/
  BALLOONING_TILESET.md     design record for the tile-level switch + inventory
  BALLOONING_TILE_BRIEF.md  the one-page brief the pathologist receives
tests/
  test_core.py            unit tests incl. the review contract
  test_tileset.py         the tile-set draw: caps, split, repeats, leak
  regression_steatosis.py steatosis must stay byte-identical; golden/ is the baseline
data/            slides (gitignored)
outputs/         per-slide results, QC, CSVs (gitignored)
```
