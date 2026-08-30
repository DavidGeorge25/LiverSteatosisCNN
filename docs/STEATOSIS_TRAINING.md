# Training the U-Net: what the numbers will and will not mean

Design record for `mashpath/train/unet.py`, `outputs/dataset_v2` and
`cluster/{dataset,unet}.sbatch`, written 2026-08-29 **before any model has been
trained**. Same reason as `BATCH_EFFECTS.md`: the parts of a training pipeline
that decide whether a result means anything are the hardest to change once
numbers exist, and the easiest to get wrong under deadline.

**Updated 2026-08-29 with the first trained fold** — see §7. Still CPU, still
one fold of eight; the leave-one-batch-out sweep is a cluster job.

---

## 1. `dataset_v2`: 37,650 tiles, 8 batches, no split on disk

```
outputs/dataset_v2/            251 slides, 8 staining batches, 37,650 tiles, ~25 GB
  images/       512 px level-0 RGB
  labels/       9-member ensemble consensus
  confidence/   per-pixel vote fraction, 0-255
  tissue/       the tissue mask -- see §3
  manifest.csv  one row per tile, carrying slide, batch, diet, cohort
outputs/dataset_v2_reserved/   8 slides, 1,200 tiles -- R22-354, never trained on
```

Built by `build_flat_dataset`, 13.9 min at `workers=8`. One slide fails, the
known truncated `R26-122-14_HE`.

**No train/val/test directories, and that is the design.** `dataset_v1` has
them, planned by slide and stratified by severity, and that layout is the
mistake `BATCH_EFFECTS.md` §2 is about: it makes a slide-level split look like
the model's split. Two slides from one batch were cut on one day by one person
from one reagent lot; holding one out tests generalisation to a new **animal**,
not to a new **stain**. With `batch` on every row, `mashpath.train.splits` folds
at training time, where it can refuse a tile-level split outright and enforce
the reserved batches. A directory named `train/` can do neither.

**Cohort is derived, not declared.** Six of the nine batches have no metadata,
so a slide whose exported tiles average under 1% fat is labelled `negative`.
Recorded per slide in the manifest, so a later metadata key from the lab can
correct it without re-exporting 25 GB.

| batch | slides | tiles | mean fat |
| --- | ---: | ---: | ---: |
| 2026-04-20_CCl4_HE | 37 | 5,550 | 0.17% |
| 2025-07-09_Evelyn_HE | 23 | 3,450 | 1.22% |
| 2025-04-29HE | 43 | 6,450 | 2.13% |
| 2025-03-07_Jaya HE | 29 | 4,350 | 3.50% |
| 2025-09-24HE | 25 | 3,750 | 6.52% |
| 2025-08-25HE | 40 | 6,000 | 7.84% |
| P273-2026-02-13_Christina HE | 30 | 4,500 | 7.90% |
| 2025-11-14HE | 24 | 3,600 | 8.68% |

173 slides read positive, 78 negative.

---

## 2. The reserved batch has its own directory

`build_flat_dataset` **refuses to export a reserved batch**. R22-354 is written
separately, by a call that passes `reserved=()` explicitly, into
`dataset_v2_reserved/`.

Two locks on the same door, deliberately. `splits.py` already prevents a
reserved batch reaching a training fold; leaving it out of the training set as
well means a script that never imports `splits.py` still cannot touch it. It is
the only place in 260 slides where diet is known and stain is held fixed — the
one measurement in this project that is not the detector grading its own
homework (`STEATOSIS_NINE_BATCHES.md` §1) — and evaluating on it should take a
path someone has to name.

The exported labels preserve the contrast the survey found: chow 0.06–0.18%,
NASH 7.9–11.2%.

---

## 3. Predictions are confined to tissue

Found by the smoke test, and worth stating because it would not have shown up
as an error. The detector does `white &= tissue_mask` before it measures
anything, so every teacher number is fat area over **tissue** area. A student
scored on the whole tile is scored on a larger canvas — and glass is white, so
the difference is systematic rather than noise. **5.8% of tiles are under 90%
tissue and 972 are under 70%.** Before the fix, an untrained model reported fat
fractions above 1.0, which is the visible tip of it; a trained model would have
reported something plausible and slightly wrong.

So `tissue/` is exported per tile and `predict_frame` intersects with it. When
the directory is absent it says so loudly rather than silently measuring
something else. `tests/test_unet.py` pins both behaviours, and pins the
order-stability of the evaluation pipeline — the tissue mask is joined to its
prediction **by row order**, so a shuffled evaluation set would mask every tile
with a different tile's tissue and still produce plausible numbers.

---

## 4. What is measured

**`dice_vs_teacher`, and never "Dice".** Agreement with a programmatic label is
agreement with the teacher. A model that matched it perfectly would have learned
its false positives too — including the pale-cytoplasm one that `LIMITATIONS.md`
§6 says is systematic rather than random, and therefore learnable. It is
reported per held-out batch, and it answers "did the model learn the rule, and
does the rule survive a new stain", not "is it right".

**The external test.** Chow vs NASH on R22-354, at slide level, reported
alongside **the teacher's own AUC on the same slides**. The two have to be read
together: a model scoring 1.000 where the teacher scores 1.000 has inherited a
working measurement; a model scoring below it has lost something the
pseudo-labels contained. Four animals a side, so this can confirm a failure
much more strongly than it can confirm a success.

**The floor test.** Predicted fat on the CCl4 cohort, against the teacher's
0.17% mean / 0.44% worst. This is the one place the student can be shown to
*beat* the teacher rather than copy it: a model that generalises the shape rule
instead of memorising the threshold should reject pale cytoplasm the teacher
accepts. Above the floor it is hallucinating; below it, and still separating
chow from NASH, is the result worth having.

**What none of them are.** None is an accuracy measurement, because there is
still no reference segmentation — `LIMITATIONS.md` §1, unchanged. The 40-tile
package at `outputs/annotation_set_v1/` remains the gate.

---

## 5. The ablations, and which defaults are claims

| switch | default | why that default |
| --- | --- | --- |
| `confidence_weighting` | **on** | 26% of a positive tile's label area is contested; unweighted BCE spends equal gradient on a pixel nine members agreed about and one five argued over |
| `augment` | **strong** | removes the stain shortcut whether or not a residue was left |
| `stain_normalize` | **none** | `BATCH_EFFECTS.md` §4: normalization is an intervention on the exact axis under test, so the first number must be the un-normalized baseline it has to beat |

Each is one flag on `python -m mashpath.train.unet`. The point of writing them
as switches rather than as a chosen configuration is that "we used confidence
weighting" is a claim, and a claim needs the run without it.

---

## 6. Running it

```bash
cd $MASHPATH_ROOT && git pull

# One slide per array task, capped at 15 concurrent. Not one job over all 260:
# a slide read off Lustre costs ~82x a local read, so a single job would spend
# its wall clock on per-operation latency. Staged to node-local NVMe it is the
# same ~30 s a slide it is on a laptop.
awk -F/ '!seen[$NF]++' cluster/.dripped \
  | grep -v '2025-03-28_Celina' > /scratch/$USER/train_slides.txt   # 252 lines
sbatch --array=1-252%15 cluster/dataset_array.sbatch /scratch/$USER/train_slides.txt

# Waits for the whole array via --dependency=singleton (shared job name), then
# assembles the manifest, writes batches.txt, and exports the reserved batch
# into its own directory.
sbatch --dependency=singleton cluster/dataset_finalize.sbatch

# Leave-one-batch-out, one fold per task, driven by the batches.txt the
# finalize step wrote from the manifest itself.
DS=$MASHPATH_OUT/dataset_v2
sbatch --array=1-$(wc -l < $DS/batches.txt) cluster/unet.sbatch $DS/batches.txt
```

**The array cap is not decoration.** An uncapped submit last time had 219 tasks
copy a ~500 MB slide off Lustre simultaneously; every one timed out having
barely started, and a TIMEOUT bills the full wall clock -- ~9,750 core-hours of
the group's fairshare. `%15` is the cap that worked.

Eight folds, each training on seven staining runs and validating on the eighth,
each then scored on the reserved batch and the CCl4 floor. `report()` in
`evaluate.py` prints the worst batch before the mean, which is the number that
predicts what happens when the lab stains a new run next month.

**Read the spread, not the mean.** A model at 0.90 on seven batches and 0.55 on
the eighth has a mean of 0.85, reads as a good model, and will fail on the next
staining run. The 0.55 is the finding.

---

## 7. The first fold: CCl4 held out

Trained on this laptop, CPU only. 1.9M-parameter U-Net (`base_filters` 16),
192 px random crops, batch 8, confidence-weighted BCE + soft Dice, strong colour
and stain jitter, no stain normalization. **Fit on 188 slides across 7 staining
batches; 26 dev slides chose the checkpoint; the 37 CCl4 slides were scored
once, at the end.** Nine epochs at ~28 min each; dev loss flat from epoch 4
(0.0085) to epoch 9 (0.0084), and the run was cut at epoch 9 of 12 by a process
exit — the checkpoint is epoch 9 and the last three epochs would not have moved
it.

`outputs/unet/ccl4_fold/summary.json`.

### The floor: it does not invent fat on a staining run it never saw

| | model | teacher |
| --- | ---: | ---: |
| CCl4 mean, 37 slides | **0.167%** | 0.166% |
| CCl4 worst slide | **0.351%** | 0.445% |

The student matches its teacher's mean on fat-free tissue to the third decimal
and is **better on the worst slide** — 0.351% against 0.445%, both under the
0.5% below which a slide-level call is treated as zero. This is the first
evidence in the project of the student improving on the teacher rather than
copying it, and it is the direction that matters: the teacher's worst case is
where its systematic false positive on pale cytoplasm shows.

Read the honest caveat with it: the model predicts *some* fat on 50% of CCl4
tiles against the teacher's 33%, so it is more willing to mark a few pixels — it
just marks fewer of them where it does. Tile-level willingness up, slide-level
area down.

### The external test: it reproduces the diet validation, and widens it

R22-354 is held out on **both** axes — never trained on, and its staining run
(`2025-03-28_Celina`) is not in the training set at all. So this is an unseen
stain and an unseen diet.

| slide | diet | teacher | model |
| --- | --- | ---: | ---: |
| R22-354_3_WT-CHOW3 | chow | 0.06% | 0.09% |
| R22-354_19_ACLY656 CHOW4 | chow | 0.10% | 0.07% |
| R22-354_1_WT-CHOW1 | chow | 0.12% | 0.14% |
| R22-354_16_ACLY656 CHOW1 | chow | 0.18% | 0.15% |
| R22-354_7_WT-NASH1 | nash | 7.93% | 8.61% |
| R22-354_8_WT-NASH2 | nash | 10.03% | 10.75% |
| R22-354_25_ACLY656 NASH 6 | nash | 11.07% | 11.72% |
| R22-354_23_ACLY656 NASH4 | nash | 11.20% | 12.01% |

**Model AUC 1.000, matching the teacher's 1.000 — and the gap widens from 43.5×
to 57.6×**, because the model reads chow lower (0.150% vs 0.182% at the top) and
NASH higher (8.61% vs 7.93% at the bottom). Four animals a side, so this
confirms a failure far more strongly than it confirms a success; what it does
establish is that nothing was lost in the transfer from labels to model.

### Agreement with the teacher

`dice_vs_teacher` 0.656 on the held-out CCl4 batch, where 67% of tiles carry an
empty label and Dice on near-empty masks is unstable by construction. On the
reserved batch it runs 0.86–0.93 on the four NASH slides and 0.53–0.82 on the
four chow ones — the same instability, from the same cause. **This is agreement,
not accuracy**, and it is the number to quote least.

### What this fold does and does not establish

It establishes that a U-Net trained only on programmatic pseudo-labels
transfers to an unseen staining run without inventing fat, and carries the one
external validation the project has through that transfer intact.

It does not establish generalisation across batches in general — that is the
other seven folds, and `evaluate.py`'s `report()` prints the worst before the
mean for a reason. It says nothing about pixel accuracy; `LIMITATIONS.md` §1 is
untouched. And the model is 1.9M parameters at 192 px because that is what a
laptop CPU will run in an evening, not because it is the right size.
