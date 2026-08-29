# Is the ballooning score reading biology, or reading the staining run?

Measured 2026-08-29, while the annotation study is out. No labels are involved
and none are needed: every number here comes from detector output already on
disk, and the question — does a score that separates cohorts separate them
because of the tissue or because of the stain — is the one `BATCH_EFFECTS.md`
says has to be settled *before* labels arrive, not after.

Reproduce with:

```bash
PYTHONPATH=. .venv/bin/python -m mashpath.train.audit \
    --frame review_sets/study_frame.csv.gz \
    --score-dir bal_chunks --score-dir bal_local
```

Writes `outputs/ballooning_score_{per_batch,slides,batch_pairs}.csv`.

**The finding, in one paragraph.** `BALLOONING_TILESET.md` §5 records that the
detector's per-tile score puts MASH *below* CCl4 at AUC 0.085, and lists as its
first caveat that cohort is confounded with staining batch and that this "cannot
be separated with the slides on this laptop". With eight batches now in the
frame it can be. The score turns out **not** to be a stain reader in the simple
sense — it barely identifies which batch a tile came from (mean |AUC−0.5| =
0.055) — but the cohort gap it produces is **not distinguishable from an
ordinary difference between two staining runs**: it ranks 8th of 28 batch pairs,
against a median pair of 0.303. Meanwhile the one contrast where stain is held
fixed by construction shows nothing — 4 chow against 4 NASH livers cut and
stained together read 0.489 at tile level, a coin — and a different explanation
does survive testing: within the MASH batch, **the score falls as steatosis
rises** (Spearman −0.50 over 14 slides). The inverted ordering is best read as the
detector being *suppressed by fat*, which is a worse problem than a stain
artefact, because it is strongest exactly on the slides where ballooning matters
most.

---

## 0. A confound in the input, found first

`study_frame.csv.gz` carries the `score` column each run wrote, and **the runs
do not agree on what a score is.** 43 of the 53 slides scored at the time ran under the
ranking vetoes the pathologist gave us on 2026-08-16 (`min_area_ratio` 2.0,
`max_area_ratio` 4.0); 10 ran under the ones those replaced (1.3 / 6.0), which
admit roughly four times as many cells and admit them lower down.

The 10 are not scattered. They are **every slide of three batches**:

| batch | slides on the stale vetoes |
| --- | ---: |
| 2025-11-14HE | 5 of 5 |
| 2025-07-09_Evelyn_HE | 3 of 3 |
| P273-2026-02-13_Christina HE | 2 of 2 |

This is not a claim that `BALLOONING_TILESET.md` §1 was wrong. It says the 32
slides of the earlier frame were re-finalized under the current config before
the draw, and they were — every one of them is in `bal_local/`, and every one of
them still reproduces exactly. The drift entered when the frame grew: the wave
fetch added six batches whose `bal_chunks/` output was scored on the way through
and never re-finalized, and `build_frame` reads the stored column, so it
inherited them.

So the raw frame shows those three batches as the highest-scoring in the
collection, and they are the highest-scoring because of when they were run:

| batch | stored column | one config | change |
| --- | ---: | ---: | ---: |
| 2025-08-25HE (MASH) | 2.06 | 2.06 | — |
| 2025-11-14HE | 3.95 | **2.12** | −1.83 |
| 2025-07-09_Evelyn_HE | 5.14 | **2.53** | −2.61 |
| P273-2026-02-13_Christina HE | 4.12 | **2.59** | −1.53 |
| 2025-03-28_Celina 656D H&E | 2.77 | 2.77 | — |
| 2025-03-07_Jaya HE | 2.84 | 2.84 | — |
| 2026-04-20_CCl4_HE | 2.86 | 2.86 | — |
| 2025-09-24HE | 3.54 | 3.54 | — |

**A 2.5× spread across staining runs collapses to 1.7× once every slide is
scored the same way.** Read off the stored column, "the score varies enormously
by batch" is most of a finding and all of an artefact.

`review/rescore.py` fixes it at the point of use: `raw_score` is the weighted
z-sum before any veto and does not depend on the ranking config, and every
input `apply_vetoes` needs is a column of `cells.csv` — so the vetoes are simply
applied again, to every slide, under one config, by calling the same
`apply_vetoes` the pipeline calls rather than restating the rules. Everything
below is computed on the re-derived column.

`tests/test_rescore.py` asserts the property that makes this trustworthy:
re-applying a run's **own** config reproduces the score column that run wrote,
cell for cell, across all 71 stored runs. One cell in 2.8 million differs, and
it is a cell whose area ratio lands exactly on the threshold after a round trip
through CSV, where the veto's strict `<` is decided by the last binary digit.

**What this did and did not affect.** It did not affect the package the
pathologist has: the draw is 100% uniform and the score steers nothing, which is
why `fullset.py` removed the enriched stratum rather than setting it to zero.
It did affect every band in `study_frame.csv.gz`, and it would affect any future
round that tried to enrich. The one-line fix — `build_frame` calling
`rescore.tile_scores` with the config it already has, instead of reading the
stored column — is deliberately **not** applied here: it changes what package
building does, and this is a bad week to change that. It should be done before
the next package is built.

---

## 1. Can the score tell you which staining run a tile came from?

Mostly not.

| batch | slides | tiles | median score | separability | tiles with a proposal |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2025-08-25HE (MASH) | 14 | 4289 | 2.06 | 0.403 | 78.3% |
| 2025-11-14HE | 5 | 1230 | 2.12 | 0.432 | 84.1% |
| P273-2026-02-13_Christina HE | 2 | 537 | 2.59 | 0.492 | 90.3% |
| 2025-07-09_Evelyn_HE | 3 | 745 | 2.53 | 0.505 | 88.5% |
| 2025-03-28_Celina 656D H&E | 8 | 2059 | 2.74 | 0.530 | 88.3% |
| 2025-03-07_Jaya HE | 3 | 833 | 2.84 | 0.546 | 94.1% |
| 2026-04-20_CCl4_HE (CCl4) | 18 | 4418 | 2.86 | 0.564 | 96.1% |
| 2025-09-24HE | 3 | 729 | 3.54 | 0.625 | 82.7% |

Separability is `batch_separability` from `evaluate.py`: the AUC of "is this
tile from this batch?" given the score alone. **Mean |AUC−0.5| = 0.055, worst
0.125.** This is the first time that function has been pointed at real output
rather than synthetic data, and the answer is a genuine negative: at tile level
the score carries little batch information. The simple version of the worry —
"it is reading the stain" — is not supported.

The proposal rate is a second channel and moves further: the MASH batch produces
a veto-passing cell on 78% of its tiles against 96% for CCl4. That is not a
score difference, it is a *coverage* difference, and it points the same way as
§4.

---

## 2. Slide-to-slide spread: 45% of it lies between batches

Over the 56 slide medians: between-batch sum of squares 14.16, within-batch
17.03 — **between/total = 0.454**.

This is not by itself evidence of a stain effect, and must not be quoted as
one. Batch is one-to-one with experiment here, so between-batch variance holds
every real cohort difference as well as every staining difference. What it gives
is a ceiling: at most 45% of the spread in slide medians could be biology, and
at most 45% could be stain. It is the reason the next section is necessary.

---

## 3. Is the cohort gap bigger than an ordinary difference between two batches?

No.

Ranking all 28 batch pairs by how far apart their slide medians sit
(|AUC−0.5| on slide medians, both sides ≥2 slides):

| rank | pair | n | AUC | separation |
| ---: | --- | ---: | ---: | ---: |
| 1–5 | five pairs involving 2025-09-24HE, 2025-08-25HE, Jaya, Evelyn, Christina, 2025-11-14 | 2–14 | 0.000 / 1.000 | 0.500 |
| 6 | Celina vs 2025-09-24HE | 8 v 3 | 0.042 | 0.458 |
| 7 | 2025-09-24HE vs CCl4 | 3 v 18 | 0.926 | 0.426 |
| **8** | **2025-08-25HE (MASH) vs CCl4** | **14 v 18** | **0.103** | **0.397** |
| 9 | 2025-11-14HE vs CCl4 | 5 v 18 | 0.111 | 0.389 |
| 10 | Jaya vs 2025-11-14HE | 3 v 5 | 0.867 | 0.367 |
| … | … | | | |
| 28 | Jaya vs CCl4 | 3 v 18 | 0.500 | 0.000 |

Full table in `outputs/ballooning_score_batch_pairs.csv`.

**The MASH↔CCl4 separation is 0.397, rank 8 of 28, against a median pair of
0.303.** Seven pairs of staining runs are further apart than the cohort contrast
the AUC-0.085 result rests on — and rank 9 is `2025-11-14HE vs CCl4` at 0.389,
a pair with no cohort story attached to it at all.

Read the top of that table with the sample sizes in view: five pairs reach a
perfect 0.500 with three slides a side, where perfect separation costs nothing.
The point is not that those pairs are impressive. It is that with this many
slides per batch, a separation of 0.4 is **what two arbitrary staining runs
routinely produce**, so a separation of 0.4 between two cohorts is not evidence
about cohorts.

This is exactly the comparison `BATCH_EFFECTS.md` §6 said needed more batches to
become possible, and it is the first thing the nine-batch fetch has paid for.

---

## 4. What does move the score, measured with stain held fixed: fat

Within the MASH batch — 14 slides, one staining run, one day, one reagent lot,
macrovesicular fat spanning 3.7% to 14.9% — the ballooning score falls as fat
rises:

| batch | n | Spearman(score, macro fat) | p |
| --- | ---: | ---: | ---: |
| 2025-08-25HE (MASH) | 14 | **−0.495** | 0.072 |
| 2026-04-20_CCl4_HE | 18 | +0.036 | 0.887 |

CCl4's fat range is 0.1–0.4%, i.e. no range at all, and it shows no relationship
— which is what a fat-driven effect should look like. The two lowest-scoring
slides in the entire collection are `R25-264-1` (−0.10, 14.3% fat) and
`R25-264-12` (0.58, 14.9% fat), the two fattiest MASH slides.

At n=14 and p=0.072 this is suggestive, not established, and it is stated that
way. But it is a within-batch measurement, so **stain cannot explain it**, and
it is the mechanism `BALLOONING_TILESET.md` §5 hypothesised — "fat-laden MASH
hepatocytes read as less pale-and-swollen once droplets are excluded as void" —
now measured rather than assumed. The falling proposal rate in §1 says the same
thing from the other side: on fatty slides fewer cells survive the vetoes at
all.

If it holds, it is a worse problem than a batch artefact. A stain effect can be
augmented or normalized away. A detector whose sensitivity drops as steatosis
rises is losing signal precisely on the slides where NASH-CRN ballooning is
being graded, and no amount of colour jitter touches it.

---

## 5. The one contrast where stain cannot be the answer

R22-354 is the only batch in 260 slides where treatment varies with staining
held fixed: eight livers, four chow and four NASH, cut and stained in one run on
2025-03-28. Three of the eight had never been scored — the detector had only
ever seen 3 chow and 2 NASH — so they were scored for this, under the same rule
as the other five (a centred 300-tile window, `start = (N-300)//2`, which
reproduces all five existing windows exactly). The four-a-side comparison is
also balanced on genotype, 2 WT and 2 ACLY656 each way, which is worth as much
as the diet balance and was not designed for.

| diet | genotype | slide | median score |
| --- | --- | --- | ---: |
| chow | ACLY656 | R22-354_16_ACLY656 CHOW1 | 1.799 |
| chow | WT | R22-354_3_WT-CHOW3 | 2.842 |
| chow | WT | R22-354_1_WT-CHOW1 | 3.099 |
| chow | ACLY656 | R22-354_19_ACLY656 CHOW4 | 3.342 |
| nash | WT | R22-354_7_WT-NASH1 | 2.382 |
| nash | WT | R22-354_8_WT-NASH2 | 2.707 |
| nash | ACLY656 | R22-354_25_ACLY656 NASH 6 | 2.711 |
| nash | ACLY656 | R22-354_23_ACLY656 NASH4 | 3.124 |

**Tile-level AUC 0.489 over 965 NASH and 1094 chow tiles. Slide-level AUC 0.375,
Mann–Whitney p = 0.69.** The medians interleave completely: the lowest-scoring
and the highest-scoring slide of the eight are both chow.

Read this for what it is. Four animals a side cannot exclude a modest effect,
and the direction of the slide-level number is NASH slightly *below* chow, which
is where §4 would put it. What it does exclude is the reading that makes the
cross-batch result interesting: if the score responded to NASH pathology at all
strongly, the one place we can look without stain in the way is where it would
show, and it does not. A detector that reads a difference of 0.397 between two
staining runs and 0.011 between diseased and normal liver from the same run is
telling us which of those two it is sensitive to.

**This is what the chow slides were fetched for.** `BALLOONING_STUDY.md` §5
lists "the model reads normal liver as negative" as newly available; this is the
first claim actually made with them, and it is made about the detector rather
than about a model. Four is few — if any of R22-063 / R25-079 / R25-151 /
R25-223 / R25-345 / R26-041 turn out to hold untreated controls, this is the
test that gets stronger with no new labelling, which is the second time that
question has paid off in one document.

---

## 6. Controls

**Sampling geometry is not doing it.** Slides scored as one contiguous 300-tile
strip and slides scored as several spaced windows are mixed across batches, so
it is worth ruling out. Inside the CCl4 batch, which contains both:

| how the slide was sampled | slides | median | range |
| --- | ---: | ---: | --- |
| one strip | 4 | 2.70 | 2.64–3.03 |
| spaced windows | 14 | 2.92 | 1.85–3.73 |

A 0.2 difference against a within-batch range of 1.9. Not the effect.

**Not every slide is scored.** 56 of the 146 sections in the frame have detector
output, 8 of the 9 batches (2025-04-29HE has none), and the per-batch slide
counts are as low as 2. Every number here is bounded by that, and the pairwise
table carries its sizes for exactly that reason.

---

## 7. What this licenses, and what it does not

**Licensed:**

- Do not quote AUC 0.085 (or its uniform-config value, slide-level **0.103**)
  as evidence that the detector measures something real and inverted. A
  difference that size between two batches is ordinary in this collection.
- Do not quote per-batch score levels off `study_frame.csv.gz`'s stored column
  at all. Re-derive first.
- Treat "the score is depressed by steatosis" as the leading hypothesis for the
  inverted ordering, and test it directly when labels land: her circled cells on
  MASH sections against the detector's proposals on the same fields.

**Not licensed:**

- This says nothing about whether the score finds ballooned cells. It is a test
  of what the score's *slide-level* ordering means, and the label set exists
  because tile-level truth does not.
- It does not clear the score of batch effects. Tile-level separability near 0.5
  is consistent with a per-slide offset that happens to average out across a
  batch, and 45% of the slide-median spread sits between batches.
- It does not rescue the enrichment idea. §3 removes a reason to trust the
  score's cohort ordering; it adds none to trust its within-slide ordering,
  which is what enrichment would need.

**When her file arrives, the first thing to compute** is the tile-level AUC of
this score against her field verdicts, held-out-batch, using
`held_out_batch_eval`. That is the measurement this whole document is a
stand-in for.
