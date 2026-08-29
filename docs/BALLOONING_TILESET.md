# Ballooning: tile-level supervised labelling

Design record for the switch from rule-based candidate generation to
supervised learning on pathologist labels. Decided 2026-08-18 after the
candidate review round: the score does not separate cohorts and the top band
was rejected wholesale, so the detector can no longer define what a label is.
It keeps one job — stratifying the draw — and that job is falsifiable, because
the labels will say whether its ordering carried any information.

**Steatosis is unaffected.** Pseudo-labelling works there and stays as it is.

---

## 1. What is on disk

### The whole collection, as the cluster records it

`cluster/.dripped` is the only record of which staining run a slide belongs to
— it is not recoverable from the SVS file.

| staining batch | slides |
| --- | ---: |
| 2025-04-29HE | 43 |
| 2025-08-25HE (R25-264, MASH) | 40 |
| 2026-04-20_CCl4_HE (R26-122, CCl4) | 38 |
| P273-2026-02-13_Christina HE (R26-041) | 30 |
| 2025-03-07_Jaya HE (R25-079) | 29 |
| 2025-09-24HE (R22-063) | 25 |
| 2025-11-14HE (R25-345) | 24 |
| 2025-07-09_Evelyn_HE (R25-223) | 23 |
| 2025-03-28_Celina 656D H&E (R22-354) | 8 |
| **total** | **260** |

### What is actually on this laptop

**52 files, 51 readable, from 2 of those 9 batches.**

| cohort | folder | n | tissue | tiles @512 px / level 0 / ≥50% tissue |
| --- | --- | ---: | ---: | ---: |
| MASH, R25-264 | `data/R25-264_MASH_HE/` | 14 | 2,722 mm² (median 162, range 134–293) | 42,335 (median 2,513, range 2,077–4,554) |
| CCl4, R26-122 | `data/2026-04-20_CCl4_HE/` | 37 | 2,639 mm² (median 70, range 40–91) | 41,040 (median 1,084, range 615–1,417) |
| | | **51** | **5,361 mm²** | **83,375** |

All 51 are Aperio SVS, 3 pyramid levels, 0.4953 µm/px, 20× objective.

**Fails to open: 1.** `R26-122-14_HE.svs` — truncated, ~281 KB short; the TIFF
header points past the end of the file. `OpenSlideUnsupportedFormatError`.
Known, long-standing, needs re-downloading. Every command skips and reports it.

**Duplicate:** `R26-122-1_HE_70.svs` also sits loose in the repo root, byte-identical
to the copy in `data/`. Harmless, but it is 271 MB and `git status` shows it.

### Where the detector has already run

`bal_chunks/` holds one contiguous 300-tile window per slide for **44 slides,
5 from each of the 9 batches** — a deliberately spread subset.
`R26-122-1_HE_70` had chunks but no merged `cells.csv`, from a cluster job that
died mid-run; it needed no new segmentation, only a finalize, and is now usable
(915 frame tiles, the largest of any slide here).

`bal_local/` holds **every slide that has both pixels here and a score**, 32 of
them, measured for this work: 14 MASH at ~400 frame tiles each and 18 CCl4 at
~200. That took ~1.7 h of StarDist across 61 windows.

**Windows are spaced, not contiguous.** A chunk range is an index into
raster-order tiles, so 300 consecutive tiles is one horizontal *strip* — one
lobule band, one staining gradient, one focus plane. The same tile budget spent
as four 100-tile windows at spaced offsets covers the whole section instead.
The cost is more window boundary, and boundary cells have truncated
neighbourhoods to normalize against; that was the reason to be careful, and it
turned out not to bite — **normalizable came back at ~100% on every slide**
(median 67–208 neighbours against a `min_neighbours` floor of 25). Window size
stays at 100 rather than 50 for this reason: 50 tiles is about one tile row
(~254 µm) against a 150 µm normalization radius, which would truncate
everything in it.

`sampling_report.md` prints a `patches` column per slide — how many spatially
disjoint pieces its frame covers — and names the ones that are a single strip,
so a reader can tell which slides' band percentiles describe the whole section
and which describe one band of it. Five do: the four CCl4 slides taken straight
from the cluster run, plus `R25-264-1`, whose new window landed close enough to
its cluster window to merge into one patch.

**Two configs, one frame — resolved.** The cluster run predates the 2026-08-16
review that retuned `min_area_ratio` 1.3 → 2.0 and `max_area_ratio` 6.0 → 4.0.
Comparing a cluster-scored slide against a locally-scored one therefore compares
*thresholds*, not biology. Every one of the 32 slides in the frame was
re-finalized under the current config before the draw, so the set is internally
consistent; `outputs/ballooning_cohort_scores_tuned.csv` re-derives the whole
44-slide comparison the same way, without opening slides that are not here.

---

## 2. The chow-control gap

**Four chow-fed slides exist in the entire 260-slide collection.** All four are
in one staining batch, from one experiment:

```
R22-354_1_WT-CHOW1     R22-354_3_WT-CHOW3
R22-354_16_ACLY656 CHOW1   R22-354_19_ACLY656 CHOW4
```

Three of the four have been scored (`R22-354_3_WT-CHOW3` has not). **None are
on this laptop.** No other filename in the collection encodes diet, so for 252
of 260 slides we cannot tell from what we hold whether an animal was a control.

### Why CCl4 cannot stand in

Carbon tetrachloride is a fibrosis and centrilobular-necrosis model. It causes
**hydropic degeneration** — swollen, pale hepatocytes that are a recognised
morphological mimic of ballooning. Using CCl4 as the ballooning-negative
therefore does the opposite of what a negative control is for: it does not
bound the false-positive rate, it *defines* the hardest false positives as
negative. This is the exact opposite of its role in the steatosis work, where
"any fat detected here is a false positive by construction" is sound, because
CCl4 genuinely does not cause steatosis.

### How big the gap is

Not primarily a power problem. With perfect separation, an exact two-sided
Mann–Whitney on 4 chow vs 14 MASH slides reaches p = 0.0007, which is fine.

The problem is **confounding**. All four chow slides come from one staining
batch, cut and stained by one person on one day. "Chow vs MASH" is therefore
perfectly aliased with "2025-03-28_Celina batch vs everything else". A model
that scores chow low might have learned diet, or might have learned that
batch's stain. Nothing in the current collection can separate those, and adding
more slides *from that batch* would not help.

### What this does and does not block

- **It does not block the labelling.** Her labels define ballooning at field
  level; they do not depend on having a normal liver in the set. Go ahead.
- **It blocks the headline claim.** "The model scores normal liver at zero" is
  not supportable, and neither is a clean specificity figure.
- **The fix is metadata, not compute.** The R22-063 / R25-079 / R25-151 /
  R25-223 / R25-345 / R26-041 studies almost certainly include untreated
  controls; the filenames just do not say so. Getting the diet key for those
  182 slides is the single highest-value thing to resolve, and it is a question
  for the lab, not a job for the cluster.

---

## 3. The labelling scheme

Tile-level binary, per Heinemann et al. 2019 (Sci Rep):
**class 0 = no ballooned hepatocyte in the tile, class 1 = one or more.**

- **Tile:** 512 px at level 0 = **253.6 µm** across. Same grid as the pipeline,
  so a label has pipeline tile coordinates and aggregates without a remap.
- **Context view:** 3×3 tiles = **760.8 µm**, always on screen with the judged
  field outlined. Ballooning is defined *relative to neighbouring hepatocytes*,
  so a tile judged alone is a harder question than the criteria describe.
- **Tissue floor raised to 80%** (pipeline default is 50%). A half-glass field
  costs the same 10 seconds and carries half the evidence. *Inference must use
  the same floor*, or the model meets fields at test time unlike any it saw.

Cell-level annotation on a subset remains available later and is not designed
for here. Brunt et al. 2022 found substantial disagreement among nine expert
hepatopathologists about which individual cells are ballooned, so a cell-level
ground truth is partly an artefact of who drew it.

---

## 4. How many tiles, and why

**Time available.** 3–4 sessions × ~2 h = 6–8 h gross. Pathologists do not
label for 120 minutes without pause; at 83% duty that is ~5.0–6.6 h effective.

**Per-tile decision time — the least certain number here.** Estimated
**11 s mean**, extrapolated from ~8 s per single-cell call in the candidate
round and adjusted upward because a tile-level *negative* is a search ("no
ballooned cell among ~120 hepatocytes") rather than a judgment about one
outlined object. Positives are faster — she can stop at the first one.
**This is unvalidated. Measuring it is the pilot's main job.**

| gross | 9 s/tile | 11 s/tile | 13 s/tile |
| ---: | ---: | ---: | ---: |
| 6 h | 1,992 | 1,630 | 1,379 |
| 7 h | 2,324 | **1,901** | 1,609 |
| 8 h | 2,656 | 2,173 | 1,839 |

**Time would allow 1,900. Geometry does not. The set is 1,400.**

With 14 MASH slides supplying 80% of the draw, 1,900 cannot be drawn while
keeping two tile-grid steps between sampled tiles. The draw falls back to one
step, which permits *adjacent* tiles — 253.6 µm apart, sharing a border, mostly
the same lobular region. Measured on the real frame (10,070 tiles, 32 slides):

| n | separation achieved | touching pairs | unique tiles involved |
| ---: | ---: | ---: | ---: |
| 1,900 | 1 | 878 | 1,097 of 1,710 (**64%**) |
| 1,600 | 1 | 602 | 823 of 1,440 (57%) |
| **1,400** | **2** | **0** | **0** |

So 1,900 is worse than 1,400 on its own terms. Counting a touching pair as
roughly one field of information rather than two puts 1,900 at ~1,270 effective
independent fields — *below* the 1,260 that 1,400 delivers outright — while
spending 500 more of her decisions to get there. Volume loses to diversity here
on arithmetic, not on principle.

**1,400 presentations** = 1,260 unique tiles + 140 repeats (10%).
1,400 × 11 s ≈ 4.3 h effective ≈ 5.1 h gross ≈ **3 sessions**.

**Does a CNN get enough?** The hash split put 10 of 32 slides in test:
**912 train / 488 test** presentations.

| positive rate | train pos/neg | test pos/neg | test 95% CI, sens @0.80 | spec @0.85 |
| ---: | --- | --- | --- | --- |
| 15% | 137 / 775 | 73 / 415 | ±0.092 | ±0.034 |
| 25% | 228 / 684 | 122 / 366 | ±0.071 | ±0.037 |
| 30% | 274 / 638 | 146 / 342 | ±0.065 | ±0.038 |

Enough to fine-tune an ImageNet-pretrained backbone with heavy augmentation —
not enough to train from scratch, and not enough to claim a sensitivity better
than about ±7 points. Both worth stating in advance rather than discovering at
the figure stage. The test share came out 35%, not the nominal 25%: the split
hashes slide names so that it is stable as slides are added, and at 32 slides
the hash lands where it lands. Left alone deliberately — re-seeding until the
split looks tidy is choosing an evaluation set by its appearance.

**To get back above 1,400, add SLIDES.** A bigger window per slide does not
help: separation binds *within* a window, so more tiles in the same window are
tiles that cannot be taken. The 44 already-scored cluster slides are what
unlocks it (§6).

**Repeats.** 140 pairs gives intra-rater κ to about **±0.13** (95%, at 85%
agreement against 55% chance). That is the ceiling any model should be compared
against, and for ballooning it cannot be assumed high.

---

## 5. The draw

Per-tile score = **max cell score among the veto-passing hepatocytes in the
tile**. Max, because the label is "contains at least one" — a mean would let
one striking cell be diluted by a hundred ordinary neighbours. Tiles with no
veto-passing cell score −∞ and land in the bottom band; that is the detector
saying "nothing here", which is a real and different state.

Bands are cut at **within-slide** percentiles: a slide-level staining offset
moves every cell on that slide together, so a cohort-wide percentile would sort
slides rather than tiles.

| band | percentile | share | why |
| --- | --- | ---: | --- |
| `high` | ≥90 | 25% | where it is most confident — and most testable |
| `upper` | 70–90 | 20% | |
| `mid` | 30–70 | 15% | |
| `low` | <30 | 15% | tiles it scored low. Her "yes" here is the most informative single answer in the set |
| `anchor` | *none* | 25% | uniform draw, no score conditioning |

**The anchor stratum is not padding.** It is the only stratum that can estimate
the true prevalence, and the only thing that makes the enriched strata
reweightable back to it. Without it the set can report the prevalence of its
own enrichment and nothing else. It is drawn first, so a shortfall elsewhere
cannot eat it.

### What the score actually does — measured, not assumed

**Recomputed 2026-08-18 under one config.** The first version of this table
mixed two: the cluster run predates the 2026-08-16 retune
(`min_area_ratio` 1.3 → 2.0, `max` 6.0 → 4.0), so it compared thresholds as
much as biology. `outputs/ballooning_cohort_scores_tuned.csv` re-derives all
70 slides from the stored chunks under the tuned vetoes — no slide is opened,
so the 38 that live only on the cluster are included too.

Per-tile score = max cell score among veto-passing hepatocytes; median over
slides:

| cohort | slides | median tile score | range | p90 | tiles with ≥1 proposal | veto pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chow | 3 | **3.09** | 1.81–3.36 | 5.26 | 0.94 | 3.6% |
| NASH (R22-354) | 2 | 2.90 | 2.67–3.12 | 5.89 | 0.86 | 3.0% |
| **CCl4** | 18 | **2.89** | 1.94–3.74 | 5.01 | 0.96 | 3.4% |
| other (6 accessions) | 30 | 2.65 | 1.12–3.83 | 5.43 | 0.90 | 2.7% |
| **MASH** | 17 | **1.91** | −0.09–2.90 | 4.41 | 0.86 | 2.4% |

**The ordering is inverted, and correcting the config made it starker rather
than softer.** MASH — the cohort that should carry ballooning — scores lowest
of all five, below chow. Quantified against CCl4:

- only **1 of 17** MASH slides reaches the CCl4 median
- Mann–Whitney **p < 0.0001**, **AUC = 0.085**

An AUC of 0.085 is not "no signal". It is a *strong* signal pointing the wrong
way: flip the sign and the same score tells MASH from CCl4 at AUC 0.92. So the
detector is measuring something real and reproducible that is **anti-correlated
with the cohort we want**. That is consistent with it measuring hepatocyte
swelling and pallor generically — CCl4's hydropic degeneration is the strongest
such signal in the collection and the classic ballooning mimic — while
fat-laden MASH hepatocytes read as *less* pale-and-swollen once droplets are
excluded as void.

**Two caveats on that number, both real:**

1. **Cohort is confounded with staining batch here.** MASH is one batch, CCl4
   another. Some of the AUC could be stain rather than biology. It cannot be
   separated with the slides on this laptop — which is a further argument for
   fetching the other seven batches (§6).

   **Answered 2026-08-29, once those batches were here:
   [BALLOONING_SCORE_BATCH.md](BALLOONING_SCORE_BATCH.md).** Short version: not
   stain, but not biology either — a gap this size between two staining runs is
   ordinary in this collection (rank 8 of 28 pairs), the one within-batch diet
   contrast reads a coin, and what does track the score, within one batch, is
   how much fat the slide has. Also: the numbers in the table above are computed
   from a score column that mixes two ranking configs, and three of the five
   cohorts shift when that is fixed. The re-derived table is the one to use.
2. **Slide-level medians are not tile-level labels.** A cohort can score low on
   average and still contain the tiles we want. This is evidence about
   *enrichment quality*, not proof that MASH lacks ballooning.

**What this changes, and what it does not.** It does not change the labelling
scheme — that is the point of asking a pathologist rather than the detector.
It does change how much the score is trusted to enrich, and two safeguards
already in the draw carry the weight:

1. **Bands are cut within slide**, which removes exactly this slide- and
   cohort-level offset. A cohort-wide percentile would have filled the top band
   with CCl4 and chow and called it enrichment.
2. `n_scored_cells` is recorded per tile in `sampling_frame.csv`, so once the
   labels exist we can test *count* against *max* as the tile-level predictor
   without collecting anything more.

**Expected balance — stated as a range, because the score's calibration is
exactly what is in doubt, and the table above is not encouraging:**

- if the ordering carries signal: high band ~45–60% positive, low ~5–10%,
  anchor ~10–15% → **overall ~28–33%**
- if it carries none: every band ≈ anchor → **overall ~12%**

If the pilot comes back flat across bands, that *is* the finding — the score
has no tile-level information — and the full set should be re-enriched on
something else rather than on it.

Given the inversion above — MASH lowest of five cohorts, AUC 0.085 against
CCl4 — the honest prior is the low end. Plan for ~12% and be pleased to be
wrong; a set that comes back flat across bands is a real result about the
score, not a failed session.

**Diversity guards**

- no slide above 6% of the set (automatically relaxed to an even split when
  there are too few slides for 6% to be reachable)
- **CCl4 capped at 20%.** Its hydropic degeneration makes its tiles the most
  informative hard negatives available — and dangerous in bulk: if most
  negatives came from CCl4, the cheapest thing a CNN could learn is "this
  staining batch = negative", which would score well and mean nothing
- ≥2 tile-grid steps between two sampled tiles on one slide, so twenty tiles
  are not twenty views of one lobule
- repeats placed ≥120 presentations after their original, under a **different
  tile id** so a deliberate repeat is never confused with her pressing back to
  correct herself

**Split by slide, decided by hashing the slide name.** Not by shuffling a list:
a shuffle depends on which slides were in the draw, so a slide that was train
in a 14-slide pilot could come out test in a 45-slide set, and its pilot labels
would be training data for a model evaluated on that same slide. Hashing the
name means every set built at this seed agrees for all time.

---

## 6. What was built, and what it is missing

**Built: `review_sets/ballooning_tiles_v1/`** — 1,400 presentations, 1,260
unique tiles + 140 repeats, from **32 slides**. Verified after generation:

| property | asked for | got |
| --- | --- | --- |
| band shares | 25/20/15/15/25 | exact, every band filled |
| CCl4 share | ≤20% | 20.1% |
| max slide share | ≤6% | 5.9% |
| tile separation | ≥2 grid steps | 2 — **zero touching pairs** |
| repeat gap | ≥120 presentations | min 122, median 353 |
| split | by slide, hashed | 10 test / 22 train slides |
| reviewer sees score/band/split | never | absent from manifest *and* wire |

A pilot of 300 (`ballooning_tiles_pilot/`) remains, drawn the same way. Run the
pilot first — it is the only thing that will tell you whether 11 s/field is
right, and everything below rests on that number.

**The two packages are disjoint.** Built from one frame at one seed they
overlapped by 68 tiles, a quarter of the pilot; since a pilot label is ordinary
training data, that was 12 minutes of her time spent re-judging fields we
already had. `--exclude review_sets/ballooning_tiles_pilot` removes it, and
pilot + v1 are now a clean union of **1,530 unique tiles**. No slide changes
train/test side between the two — that is what hashing the split from the slide
name buys, and it is checked, not assumed.

### What this set cannot do

**It is 2 staining batches of 9, and it has no chow control.** 80% MASH,
20% CCl4, and every MASH tile is from one batch cut on one day. Two specific
claims are therefore out of reach no matter how good her labels are:

- **"the model reads normal liver as negative."** There is no untreated liver
  in the set. CCl4 is the only negative, and its hydropic degeneration is the
  ballooning *mimic* — a hard negative, not a normal one.
- **"the model generalises across staining."** A held-out slide here is a new
  animal, not a new batch. Batch effects will look like generalisation.

Both are gaps in the *evaluation*, not in the labels. Her verdicts define
ballooning at field level and stay valid whatever else is added later; the
train/test split is hashed from slide names precisely so that a later, wider
set can absorb this one without a slide changing sides.

### What closes it

Fetching the 44 already-scored cluster slides — 5 per batch, all 9 batches,
3 of the 4 chow slides, **already scored** so no new StarDist run. ~13 GB.

```bash
rsync -av --files-from=<(ls bal_chunks | sed 's|$|.svs|') \
      fir:~/projects/def-<pi>/slides/ data/fir_subset/
```

(The cluster stores them under batch subfolders; `cluster/.dripped` has the
full paths.) Then rebuild with `--slide-dir data/fir_subset` added. That does
two things at once: it adds the batches and the chow, and it lifts the
1,400 ceiling — the ceiling is set by having only 14 MASH slides to space tiles
across, so more slides is the *only* thing that raises it.

**The larger gap is metadata, not bytes.** 182 of the 260 slides have no diet
recorded anywhere in the filenames or `cluster/.dripped`. Whether any of
R22-063 / R25-079 / R25-151 / R25-223 / R25-345 / R26-041 contain untreated
controls is a question for the lab, and it is worth more than any amount of
compute here: if even a handful are chow, the specificity claim becomes
available and no new labelling is needed to get it.

---

## 7. Running it

```bash
# build a set (this is the command that produced v1)
.venv/bin/python -m mashpath.review.tileset \
    --slide-dir data/R25-264_MASH_HE --slide-dir data/2026-04-20_CCl4_HE \
    --score-dir bal_chunks --score-dir bal_local \
    --config configs/core.yaml --plan configs/ballooning_tiles.yaml \
    --scored-only --out review_sets/ballooning_tiles_v1

# score a slide's frame first, if it has none. Spaced windows, not one block:
.venv/bin/python -m mashpath.features.ballooning.cli chunk \
    --slide <path.svs> --output-dir bal_local --start 296 --stop 396 \
    --config configs/core.yaml --config configs/ballooning.yaml
# ...repeat at spaced offsets, then merge them:
.venv/bin/python -m mashpath.features.ballooning.cli finalize \
    --slide <path.svs> --output-dir bal_local --allow-disjoint \
    --config configs/core.yaml --config configs/ballooning.yaml

# label (pilot first -- it is what validates the 11 s/field estimate)
.venv/bin/python -m mashpath.cli tile-app review_sets/ballooning_tiles_pilot
# open the printed localhost URL; her brief is docs/BALLOONING_TILE_BRIEF.md

# what came back
.venv/bin/python -c "from mashpath.review.tiles_app import agreement_report; \
    print(agreement_report('review_sets/ballooning_tiles_pilot'))"
```

Each package holds:

| file | |
| --- | --- |
| `manifest.csv` | presentation order, ids, image paths. **No score, band or split** |
| `sampling_frame.csv` | the same rows *with* score/band/split/cohort — analysis only |
| `images/`, `context/` | 512 px core tile, 1024 px context view |
| `verdicts.csv` | append-only, one row per judgment, fsynced |
| `sampling_report.md` | what was drawn, by band / cohort / batch / slide |

The split is not cosmetic: the app is only ever given `manifest.csv`, so the
browser cannot leak what it was never sent. The slide name is not shown either
— three tiles into a section a reviewer knows which section she is in, and from
then on is partly labelling the slide.

## 8. After the pilot — read these before committing four sessions

1. **Median seconds per tile.** The whole budget rests on 11 s. Rescale
   `n_presentations` from what she actually does.
2. **Positive rate by band.** Flat ⇒ the score has no tile-level information ⇒
   re-enrich the full set on something else.
3. **Unsure rate.** High ⇒ 253.6 µm at this magnification is not a judgeable
   field, and the tile size or the context view needs changing before, not
   after, the full set is drawn.
4. **Her answer to "is this the right question?"** The candidate round failed
   on exactly that and only found out at the end.
