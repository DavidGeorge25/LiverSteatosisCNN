# Batch effects: what was built, and what it can and cannot fix

Design record for `mashpath/train/`, written 2026-08-20, before any labels
exist. Nothing here trains a model. It is the part of a training pipeline that
decides whether a number means anything, built early because it is the part
that is hardest to change once numbers exist and easiest to get wrong under
deadline.

Tested by `tests/test_batch.py` (27 tests) against synthetic data.

---

## 1. The problem, stated precisely

**Staining batch is one-to-one with experiment.** Every MASH slide was cut and
stained in one run (`2025-08-25HE`), every CCl4 slide in another
(`2026-04-20_CCl4_HE`), and the same holds for all nine batches in the
260-slide collection. `cluster/.dripped` is the only record of which slide
belongs to which run; it is not recoverable from the SVS.

A model therefore has two routes to a good validation score:

1. learn what a ballooned hepatocyte looks like, or
2. learn which staining run a tile came from, and recall that run's class
   balance.

Both produce the same number in-distribution. Only the first survives contact
with a slide stained next month. **Nothing in preprocessing or training can
separate them; only the evaluation design can**, and only if it is built before
the numbers arrive.

This is the single largest threat to every claim this project will want to make,
and it is a *design* problem, not a compute problem.

---

## 2. Batch-aware splitting — `splits.py`

The important one.

**The rule.** Train and validation are separated by staining batch. Not by tile,
and not by slide.

**Why slide-level is not enough.** Two slides from one batch were stained
together, on one day, by one person, from one reagent lot. Holding one out tests
generalisation to a new **animal**, not to a new **stain**. It is the more
tempting mistake precisely because it looks careful — it is the guard this
project already had, and the design doc for the tile set describes it as such.
It is still enforced here, as a second condition, because it catches a different
failure (the same tissue on both sides). It is not sufficient.

**How the wrong thing is made hard, not just discouraged.**

- No function in the module takes `test_size`, `train_size`, `random_state`,
  `shuffle`, `n_splits`, `frac`, `indices` or `seed`. A test asserts this by
  reflection over the module's signatures, so adding one fails the suite.
- `BatchSplit` is frozen and is defined by batch **names**. There is no
  constructor that accepts row indices — a split object that could be built
  from rows could be built from a shuffle, and then nothing downstream could
  tell this apart from the mistake it exists to prevent.
- `split_by_batch` is keyword-only on its `validation` argument, so
  `split_by_batch(df, ["batch1"])` does not silently mean something.
- `BatchSplit.assign` calls `assert_no_leakage` **itself**. A leaking split
  raises where it is made, not where someone remembers to check.
- A frame with no `batch` column raises with a message naming where batch comes
  from, rather than falling back to any other column.
- A batch present in the frame but placed in no fold raises. Silently dropping
  one is how a cohort disappears from a result.
- A frame with fewer than two batches raises, and the message says to fetch more
  batches rather than suggesting an alternative split. There is no alternative.

**Demonstrated, not asserted.** `test_a_batch_reading_cheat_survives_a_tile_split_and_dies_here`
builds a frame where the label is confounded with batch — the situation on disk
— and a feature that is *nothing but the batch index*:

| evaluation | AUC of a pure batch signature |
| --- | ---: |
| random tile-level split | **0.738** |
| held-out batch (worst) | 0.393 |
| held-out batch (mean) | 0.407 |

The tile-level split reports a strong predictor. There is no predictor. If that
test ever starts passing trivially, the protection has been lost.

**`key.csv`'s `split` column is not this.** The packages carry a `split` of
train/test hashed from the slide name, which exists so that a slide never
changes sides as sets are rebuilt. It is a slide-level split and must not be
used as the model's split. `splits.py` never reads it.

---

## 3. Held-out-batch evaluation — `evaluate.py`

`held_out_batch_eval(frame, fit_predict)` runs leave-one-batch-out and returns
one row **per unseen batch**. `report()` prints the worst batch and the spread
before the mean.

**Why not a pooled number.** A pooled metric averages over exactly the thing
under test. A model at AUC 0.90 on six batches and 0.55 on the seventh has a
mean of 0.85, reads as a good model, and will fail on the next staining run. The
0.55 is the finding.

`fit_predict(train, test) -> scores` is the only interface, so the harness works
for a CNN, a logistic regression on hand features, or a stub — which is why it
is testable now, with no model.

**`batch_separability`** asks a different question: how well does the model's
own output distinguish one batch from the others, ignoring labels entirely?
0.5 means the output carries no batch information. Far from 0.5 in either
direction means it does — and it will show that even when accuracy looks fine,
because with batch confounded with cohort, reading the stain *is* a way to get
the labels right in-distribution. This is the diagnostic for the case where the
split alone was not enough.

`auc` returns **NaN**, not 0.5, when a held-out batch has only one class. A
fabricated coin-flip inside a mean over batches is worse than a gap.

---

## 4. Stain normalization — `stain.py`. Off by default

Macenko and Reinhard, both implemented, `enabled: false` in `configs/core.yaml`.

**Off by default is the deliberate half.** Normalization is usually treated as
free. Here it is an intervention on the exact axis the whole evaluation is
about, so the first numbers produced must be the un-normalized baseline that any
later claim has to beat: same labels, same architecture, normalization on and
off, compared on held-out-batch AUC. If it is on from the first run there is no
measurement of what it did.

There is also a specific reason to distrust Macenko here: it estimates stain
vectors from the image it is given, and on a heavily steatotic section a large
share of tissue area is white droplet void carrying no stain, which the
optical-density floor discards. The estimate is then made from less tissue on
exactly the slides whose biology matters. Reinhard never estimates a stain
vector and is more robust for it; when the two disagree on a steatotic slide,
that is usually why.

Both refuse rather than guess on a tile with too little stained tissue. A silent
fallback to the reference matrix would make an unnormalized tile look
normalized.

---

## 5. Colour augmentation — `augment.py`. On by default

`StainJitter` then `ColourJitter`, `strength: strong` in `configs/core.yaml`.

**On by default is the other deliberate half, and they are opposites for a
reason.** Normalization tries to make every batch look the same and then asks
the model to trust that it worked. Augmentation makes the model stop relying on
how a batch looks at all. When batch is confounded with cohort, a normalizer
that leaves any residual batch signature leaves a shortcut the model is free to
take; an augmenter that destroys colour information removes the shortcut whether
or not residue was left. They are not alternatives — normalize if it measures
better, augment either way.

**Stronger than a typical ImageNet recipe, on purpose.** ±18° hue, ±30%
saturation, ±25% brightness and contrast, plus per-stain multiplicative and
additive jitter in optical-density space. The nine staining batches differ from
each other by more than a mild jitter spans, and a jitter narrower than the real
between-batch variation teaches invariance over a range that does not include
the range the model will meet. A test checks that the jitter can actually reach
another batch's appearance.

The morphology this task depends on — a cell enlarged relative to its
neighbours, rounded, rarefied cytoplasm — is size, shape and texture, and
survives strong colour distortion. `StainJitter` is bounded so the two stains
cannot swap dominance; a nucleus that stops being the darkest thing in the field
is no longer the object the label is about.

Geometric augmentation (flips, 90° rotations) is free and correct for histology
but is not part of the batch problem and belongs with the training loop.

---

## 6. What this does not fix

**It does not fix the confound.** With two batches on disk, a batch-level split
means training on one cohort and validating on the other, and the number that
comes back will be poor and hard to interpret. That is the honest reading of
what these data support. A tidier number from a tile-level split would be an
artefact, which is the whole point.

**The fix is slides, then metadata.** Nine batches, ~13 slides each, would make
leave-one-batch-out a real evaluation with seven training batches per fold. And
182 of 260 slides have no diet recorded anywhere; if even a handful of
R22-063 / R25-079 / R25-151 / R25-223 / R25-345 / R26-041 are untreated
controls, the specificity claim becomes available with no new labelling at all.
That is a question for the lab and it is worth more than any amount of compute.
