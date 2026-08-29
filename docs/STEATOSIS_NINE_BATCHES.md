# Steatosis: the detector outside the two batches it was tuned on

Measured 2026-08-29. The pseudo-labelling pipeline was tuned on 51 slides from
2 staining batches (`LIMITATIONS.md` §4, §11) and had never been run on the
other seven. All 260 slides turned out to be on this laptop — the lab's OneDrive
folder holds every batch, materialized, at the same 0.4953 µm/px — so it can be.

**259 slides, 9 batches, 60 seeded random tiles each, `configs/tuned.yaml`
unchanged.** One failure: `R26-122-14_HE.svs`, the known truncated file.

```bash
OD="/Users/davidgeorge/Library/CloudStorage/OneDrive-McMasterUniversity/Dongdong Wang's files - Training"
.venv/bin/python -m mashpath.cli survey \
    --group "2025-03-07_Jaya=$OD/2025-03-07_Jaya HE" \
    --group "2025-03-28_Celina=$OD/2025-03-28_Celina 656D H&E" \
    --group "2025-04-29=$OD/2025-04-29HE" \
    --group "2025-07-09_Evelyn=$OD/2025-07-09_Evelyn_HE" \
    --group "2025-08-25=$OD/2025-08-25HE" \
    --group "2025-09-24=$OD/2025-09-24HE" \
    --group "2025-11-14=$OD/2025-11-14HE" \
    --group "2026-04-20_CCl4=$OD/2026-04-20_CCl4_HE" \
    --group "P273-2026-02-13_Christina=$OD/P273-2026-02-13_Christina HE" \
    --config configs/tuned.yaml --tiles 60 --workers 8 \
    --out outputs/survey_9batches.csv
```

Nothing was retuned for this. The point of running a detector on seven staining
runs it has never seen is spoiled the moment a parameter is allowed to move.

---

## 1. The headline: the detector separates known diet, with stain held fixed

`R22-354` is the only accession in 260 slides whose filenames record diet, and
all eight of its slides were cut and stained in one run on 2025-03-28. It is
therefore the only place in the collection where **diet varies and stain does
not** — and the detector had never seen any of it.

| diet | genotype | slide | macro fat | Otsu cutoff |
| --- | --- | --- | ---: | ---: |
| chow | WT | R22-354_3_WT-CHOW3 | 0.07% | 63 |
| chow | WT | R22-354_1_WT-CHOW1 | 0.09% | 64 |
| chow | ACLY656 | R22-354_19_ACLY656 CHOW4 | 0.15% | 62 |
| chow | ACLY656 | R22-354_16_ACLY656 CHOW1 | 0.20% | 51 |
| nash | WT | R22-354_7_WT-NASH1 | 7.58% | 59 |
| nash | ACLY656 | R22-354_23_ACLY656 NASH4 | 9.64% | 60 |
| nash | ACLY656 | R22-354_25_ACLY656 NASH 6 | 9.80% | 53 |
| nash | WT | R22-354_8_WT-NASH2 | 10.46% | 61 |

**AUC 1.000. Mann–Whitney p = 0.0286. The lowest NASH slide is 38× the highest
chow slide.** Genotype is balanced, 2 WT and 2 ACLY656 a side, so the contrast
is diet and not strain.

State the power honestly: with 4 animals a side, p = 0.0286 is the *smallest
p this design can produce* — perfect separation happens by chance with
probability 2/70. It is 8 animals. What it is not is ambiguous: there is no
overlap and the gap is a factor of 38, on labels that came from the lab's
filenames rather than from anything this pipeline computed.

**Two consequences beyond the p-value.**

*Normal liver reads at the false-positive floor.* Chow comes in at 0.07–0.20%,
inside the CCl4 range of 0.04–0.34%. Two negatives of completely different kinds
— untreated liver and a fibrosis model — land in the same place. `LIMITATIONS.md`
§3 says the negative control's validity is assumed rather than verified; this is
the first independent check of it, and it passes.

*It is not the tissue mask.* The chow slide with the **most permissive** mask in
the batch (Otsu 51, the lowest of the eight) still reads 0.20%. Otsu cutoffs
interleave across diet. See §3 for the same question asked of all 259 slides.

---

## 2. Nine batches

Per-slide macro fat, tissue-area unweighted, from the 60-tile sample:

| batch | n | macro mean | range | <1% | >5% |
| --- | ---: | ---: | --- | ---: | ---: |
| 2026-04-20_CCl4 (negative control) | 37 | **0.17%** | 0.04–0.34 | 37 | 0 |
| 2025-07-09_Evelyn (R25-223) | 23 | 1.30% | 0.44–2.36 | 7 | 0 |
| 2025-04-29 (R25-151) | 43 | 2.18% | 0.02–14.38 | 23 | 6 |
| 2025-03-07_Jaya (R25-079) | 29 | 3.71% | 0.11–8.87 | 5 | 9 |
| 2025-03-28_Celina (R22-354) | 8 | 4.75% | 0.07–10.46 | 4 | 4 |
| 2025-09-24 (R22-063) | 25 | 6.69% | 1.37–13.42 | 0 | 19 |
| 2025-08-25 (R25-264, MASH) | 40 | 8.02% | 0.20–14.82 | 3 | 27 |
| P273-2026-02-13_Christina (R26-041) | 30 | 8.07% | 0.20–12.45 | 1 | 23 |
| 2025-11-14 (R25-345) | 24 | 8.64% | 0.47–16.01 | 1 | 21 |

**The negative control reproduces.** CCl4 comes back at 0.165% mean / 0.342%
worst on a 60-tile sample, against the full-run 0.173% / 0.44% in the README.
That number was the basis for "treat any slide-level call below ~0.5% as
indistinguishable from zero", and it survives being re-measured 23 days later
against a cohort seven times larger.

**No batch degenerates.** `white_threshold` is an *absolute* grey level of 210,
tuned against two staining runs, and the obvious failure mode was that a
darker or lighter run would push every slide to zero or to saturation. None of
the seven unseen batches does either. Every one lands inside the 0.04–16% range
the tuned cohorts occupy, and the ordering across batches is a plausible
biological ordering rather than a stain ordering — see §3.

**R25-079 was called steatotic by eye and reads steatotic.**
`BALLOONING_STUDY.md` §5 records "R25-079 was a candidate control batch and is
**not** one — its sections are visibly steatotic". That was a human looking at
slides. The detector, which was not told, puts that batch at 3.71% mean with 9
of 29 slides above 5%. An independent confirmation that cost nothing.

### What this changes for the training set

Severity used to be perfectly confounded with batch: the MASH batch was all fat,
the CCl4 batch none, so "predict fat" and "recognise the staining run" were the
same function and no split could tell them apart. That is over.

**Six of the nine batches now contain both a slide under 1% and a slide over
5%.** 2025-04-29 alone spans 0.02% to 14.38% across 43 slides, with 23 under 1%
and 6 over 5%.

| | before | after |
| --- | ---: | ---: |
| slides above 5% macro | 14 | **~109** |
| slides below 1% macro | 37 | **~81** |
| staining batches | 2 | **9** |
| batches spanning <1% and >5% | 0 | **6** |

That makes `splits.py`'s leave-one-batch-out a real evaluation rather than
"train on the fat cohort, validate on the fat-free one", and it is the
precondition for anything trained on these labels meaning what it claims.

---

## 3. The control that had to be run: is this the tissue mask?

Tissue detection is Otsu on HSV saturation, chosen per slide. Across all 259
slides the macro fat estimate is **negatively correlated with the Otsu cutoff**
(pooled Spearman −0.364; within-batch as strong as −0.72 on 2025-04-29 and −0.68
on Jaya). Two readings, with opposite consequences:

- **Biology.** Fat voids carry no stain, so a steatotic section is less saturated
  overall and Otsu picks a lower cutoff *because* of the fat.
- **Artefact.** A lower cutoff admits more pale near-background into the tissue
  mask, and the fat detector then finds white things in it. The fat number would
  be partly a mask artefact, and every cross-batch comparison above would be
  measuring how Otsu happened to land.

Settled by re-running all 259 slides with the adaptive threshold switched off:
`--tissue-method fixed --saturation-threshold 55`, one cutoff for every slide in
every batch, mid-range of the observed 43–78.

| | adaptive Otsu | fixed at 55 |
| --- | ---: | ---: |
| mean macro fat, 259 slides | 4.68% | 4.68% |
| mean tissue area | 163 mm² | 164 mm² |
| CCl4 mean / worst | 0.165% / 0.342% | 0.167% / 0.355% |
| Celina chow vs NASH | AUC 1.000, p 0.0286 | AUC 1.000, p 0.0286 |

**Per-slide agreement: Spearman 0.9966, Pearson 0.9987, median absolute relative
change 0.6%.** No batch mean moves by more than 0.14 points.

The fat measurement does not depend on where the tissue threshold lands. The
correlation runs the other way: steatosis lowers the section's saturation and
therefore its Otsu cutoff, and moving the cutoff back does not move the fat.

`outputs/survey_9batches_fixedtissue.csv`.

---

## 4. What this does and does not remove from LIMITATIONS.md

**Weakened:**

- **§3, negative control validity assumed.** Two independent negatives now agree:
  CCl4 fibrosis at 0.17% and untreated chow liver at 0.13%. The floor is not an
  artefact of one cohort's biology.
- **§4, small number of biological replicates.** 51 slides → 259, 14 steatotic →
  ~109.
- **§11, single stain, scanner and site.** Nine staining runs spanning
  2025-03-07 to 2026-04-20, unretuned, no degeneration. Still one scanner and
  one site.
- **§10 and the tissue-mask worry generally.** §3 above shows the measurement is
  insensitive to the mask threshold across 259 slides.

**Untouched, and still the gate:**

- **§1, no annotation-based accuracy metric.** Nothing here is a Dice score.
  Perfect separation of chow from NASH says the slide-level number tracks
  disease; it says nothing about whether the *pixels* are right, and the 360-cell
  sweep showing every setting separates the cohorts while reporting 6.82–10.91%
  mean fat is unaffected. **Absolute fat fractions remain uncalibrated.**
  `outputs/annotation_set_v1/` — 40 tiles, prefilled masks, written criteria —
  is still the single highest-value missing input, and is still unsent.
- **§2, sensitivity unmeasured.** A systematically missed droplet class is
  invisible to every check here, including this one.
- **§7, droplet merging.** Untouched.

**New, and worth stating before someone else does.** The seven new batches have
no diet or cohort metadata, so their fat readings are a detector's opinion with
no external check — except in R22-354, which is why §1 carries the weight it
does. `2025-04-29` having 23 of 43 slides under 1% is either a large untreated
control group or a detector failure on that stain, and nothing on this laptop
distinguishes those two. **That is a question for the lab and it is now worth
more than it was this morning:** if those 23 are controls, the diet validation in
§1 goes from 4 animals to nearly 30.
