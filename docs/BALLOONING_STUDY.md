# Ballooning: the full-frame annotation study

Design record for `review_sets/ballooning_full`, built 2026-08-20. Supersedes the tile-level binary design in
`BALLOONING_TILESET.md` §3–§5 for this round; §2 (the chow gap) and §6 (what is
missing) still stand and are the reason for most of what follows.

Built by:

```bash
.venv/bin/python -m mashpath.cli study \
    --slide-dir data/R25-264_MASH_HE --slide-dir data/2026-04-20_CCl4_HE \
    --slide-dir data/_wave \
    --out ~/Desktop/Ballooning_Study --stage assemble \
    --hours 0 --fields-per-section 16 --round ballooning_full_v3
```

Verified by `tests/verify_package.py` (40 checks).

**Sent as ONE package, not a warm-up followed by the rest.** A separate warm-up
was built and then dropped: it costs a round trip and a second setup, and the
thing it was for is already inside the main package. The app stops after the
second section and asks whether the images are judgeable and whether the tool is
in the way, and her answer rides out in the CSV. She is a collaborator on this
project rather than someone working to a fixed budget, so the seconds-per-field
number the warm-up was going to measure does not gate anything: sections are
presented in a seeded random order, so stopping at any point leaves a random --
and therefore still batch-balanced -- subset of sections. `--warmup-out` still
works if a future round wants one.

---

## 1. What she is asked

Two things, deliberately separate:

1. **Per field** — circle every ballooned hepatocyte. Centre and diameter, in
   fractions of the field and in microns, so a mark round-trips to slide
   coordinates.
2. **Per section** — a NASH-CRN ballooning grade, 0/1/2, given from a montage of
   that section's fields.

Cell-level marks were avoided in the earlier tile design on the grounds that
Brunt et al. 2022 found substantial disagreement among nine hepatopathologists
about *which* cells are ballooned. That finding has not changed and the marks
here are not treated as an arbiter of individual cells; they are treated as
localisation for the field-level call, which is what makes it possible to check
that a model is looking at the cell she meant rather than at something else in
the field that correlates with it.

---

## 2. Sampling

**100% uniform. No enrichment, and none available to ask for.**

The detector's per-tile score is *anti*-correlated with the cohort of interest —
MASH is the lowest-scoring of five cohorts, below chow, at AUC 0.085 against
CCl4 (`BALLOONING_TILESET.md` §5). Enriching on it would spend her attention
worse than chance. The `enriched` stratum has therefore been **removed from the
code**, not set to zero: there is no parameter for it, so it cannot come back in
a later build, and a `key.csv` carrying an `enriched` row is a reliable sign the
package is stale. `verify_package.py` checks for exactly that.

**Equal depth per section — 27 presentations, every section.**

The section grade is a prevalence judgement, so the chance of meeting a
ballooned cell at all rises with the number of fields behind the grade. If MASH
sections rested on 24 fields and CCl4 sections on 8, part of any grade
difference between them would be sampling depth — and since cohort is
confounded with batch, that artefact would be confounded with the thing under
test. Depth is held constant; balance is bought with slide counts instead.

The ceiling of 24 unique fields keeps a section to about five minutes and its
montage to one screen. The floor of 8 is where sampling noise starts to exceed
the distance between grade 1 and grade 2.

**Batch balance, bought by dropping slides.**

No staining batch may hold more than 60% of the fields. With equal depth, that
is a slide-count question, so it is enforced by taking fewer slides from the
dominant batch — never by giving that batch shallower sections.

| | before | after |
| --- | ---: | ---: |
| largest batch share | 72.5% | **60.0%** |
| slides | 51 | 35 |

Sixteen CCl4 slides were dropped to get there. That is a real cost in animal
diversity and it buys no *batch* diversity, because there are only two batches
either way. It is still right: it costs 1.4 h of her time as well, and a set
where one batch owns three-quarters of the labels makes "this staining run is
negative" the cheapest rule a model can learn.

**Repeats: 10% of presentations, not of unique fields.**

3 repeats on 24 unique fields = 11.1% of what she actually sits through. Taking
10% of the unique count would have given 2, i.e. 7.7%, and the intra-rater kappa
would then rest on fewer pairs than the design claims.

Repeats sit at the **end of their own section's block**. A section-grouped design
cannot move them further: a repeat placed an hour away would be sitting in a
different section's montage, where it does not belong. The gap is therefore
minutes rather than the hour the tile design achieved, so **the resulting kappa
is an upper bound** on her true intra-rater agreement. Stated here rather than
discovered later.

A repeat has its own `field_id` and points at the same image. The montage
**deduplicates by image**, so a repeated positive cannot inflate the prevalence
the grade is a judgement about — and deduplicating by image rather than by a
"this is a repeat" flag means nothing on the wire tells her which presentation
was the second look.

**Spread within a section.** `_spread` takes one tile per 2048-px spatial cell
before repeating any cell, so 27 fields are not 27 views of one lobule.

---

## 3. What was drawn

All nine staining batches, fetched from Fir 2026-08-21 in 13 waves.

| | |
| --- | ---: |
| sections (slides) | **146** |
| presentations | **2,628** |
| unique fields | 2,336 |
| repeats | 292 (11.1%) |
| fields per section | 16 + 2 repeats = 18 |
| estimated time @ 13 s/field | 11.3 h |
| estimated time @ 17 s/field | 14.2 h |
| package size | 603 MB |

**No time cap** (`--hours 0`). She is a collaborator working to the science, not
to a budget; sections are ordered so that any prefix is representative, and her
file is written after every answer. A study trimmed in advance to a guess about
how long she will sit is worse than a complete one she stops partway through.

**Per staining batch**

| batch | slides | share of fields |
| --- | ---: | ---: |
| 2026-04-20_CCl4_HE | 32 | 21.9% |
| 2025-08-25HE | 22 | 15.1% |
| 2025-03-07_Jaya HE | 14 | 9.6% |
| 2025-04-29HE | 14 | 9.6% |
| 2025-07-09_Evelyn_HE | 14 | 9.6% |
| 2025-09-24HE | 14 | 9.6% |
| 2025-11-14HE | 14 | 9.6% |
| P273-2026-02-13_Christina HE | 14 | 9.6% |
| 2025-03-28_Celina 656D H&E | 8 | 5.5% |

Five CCl4 slides were dropped by the batch cap, which now scales: no batch may
hold more than twice an even share (22.2% at nine batches), or 60%, whichever is
stricter. The old flat 60% could never bind once the collection grew past two
batches -- CCl4 came back from the fetch at 24.5% without tripping anything.

**All 8 R22-354 slides are in**, 4 chow and 4 NASH, exempt from every drop rule.
This is the only within-batch treatment contrast in the collection and the only
normal liver, so it is worth more per slide than anything else here.

**Section order is batch-interleaved, random within batch.** A plain shuffle is
unbiased but has variance, and variance is expensive when she may stop at any
point: on this draw a shuffle put only ONE of the four chow sections in the
first half. Interleaving leaves the expected composition of a prefix where the
shuffle put it and removes the unlucky draws.

| sections done | batches covered | largest batch | chow | NASH |
| --- | ---: | ---: | ---: | ---: |
| 15 of 146 | 9/9 | 13.3% | 1/4 | 1/4 |
| 36 of 146 | 9/9 | 11.1% | 3/4 | 1/4 |
| 73 of 146 | 9/9 | 12.3% | **4/4** | **4/4** |
| 146 of 146 | 9/9 | 21.9% | 4/4 | 4/4 |

It also helps the blinding: consecutive sections come from different batches, so
she never works through a run of one cohort's characteristic appearance.

---

## 4. Blinding

The browser is given `manifest`-equivalent data only: section id, field id, and
two image paths. It never receives slide name, cohort, batch, split, stratum or
detector score. Sections are presented in a seeded random order under opaque
six-hex-character ids derived from `sha256(seed:full:slide)`.

`verify_package.py` searches the built page for every slide name, cohort, batch
and split value in `key.csv` and fails if any appears. The check was confirmed
to fire by injecting a slide name into a copy of the page.

The page also carries **no network requests at all** — the Google Fonts links
were removed, so the package works offline and renders identically on a machine
with no internet.

---

## 5. What this set can now do, and what it still cannot

The cluster fetch happened on 2026-08-21: 100 slides, 52 GB, 13 waves, nothing
failed. What that changed:

- **A chow control exists.** Four untreated livers, and they look like normal
  liver -- no steatosis, regular polygonal hepatocytes. "The model reads normal
  liver as negative" is available for the first time.
- **A within-batch treatment contrast exists.** R22-354 holds chow and NASH cut
  and stained together, so for those eight slides diet varies while stain is
  held fixed. Nowhere else in 260 slides is that true.
- **Held-out-batch evaluation is now a real evaluation.** Nine batches means
  eight training batches per fold instead of one.

Still out of reach:

- **Diet is unknown for 138 of the 146 slides.** Only R22-354 encodes it in the
  filename. Whether R22-063 / R25-079 / R25-151 / R25-223 / R25-345 / R26-041
  contain untreated controls is a question for the lab, and it is still worth
  more than any amount of compute: if even a handful are chow, the specificity
  claim widens with no new labelling at all.
- **Cohort is unknown for 84 slides** for the same reason.
- R25-079 was a candidate control batch and is **not** one -- its sections are
  visibly steatotic.

---

## 6. The number everything rests on

**13 s/field is unvalidated.** It is extrapolated from an estimated 11 s for the
earlier *binary* tile call, adjusted upward for the circling. If it is really 17
s, the main study is 4.5 h; if it is 20 s, it is 5.2 h. That is why the budget
was set to 4.0 h rather than the 5.0 h ceiling — building to the ceiling at an
unvalidated rate spends someone else's evening.

The warm-up measures it. The app records per-field seconds and shows a running
median, and `S.fields[*].s` is in the CSV. **Rescale from what she actually
does** before committing to the main study.

Read these off the warm-up before anything else:

1. **Median seconds per field.** Everything above rests on it.
2. **Unsure rate.** High ⇒ 253.6 µm at this magnification is not a judgeable
   field, and the field size or context view needs changing *before* the main
   study, not after.
3. **Circles per positive field.** If she is drawing twenty, the interaction is
   wrong for the task and should become a count or a paint stroke.
4. **Her answer to "is this the right question?"** The candidate round failed on
   exactly that and only found out at the end.

---

## 7. The return path

When her file arrives:

```bash
.venv/bin/python -m mashpath.review.merge_study \
    review_sets/ballooning_full/key.csv ballooning_EK.csv outputs/labels
```

Writes three tables, all carrying slide / cohort / batch / diet / split /
sample_type so nothing downstream has to re-join:

| file | one row per |
| --- | --- |
| `fields.csv` | field judged: verdict, cell count, seconds |
| `cells.csv` | circled cell, in **level-0 slide coordinates** with a diameter in µm and the pixel radius a mask would use |
| `grades.csv` | section: the NASH-CRN grade, the validation target |

It prints prevalence **on the uniform rows only** -- a repeat is a second look
at a field already counted, so including repeats would double-count every
repeated positive -- and the intra-rater kappa from `intra_rater()`.

**Repeats are paired on (slide, x, y), not on field_id.** A repeat is issued
under its own id precisely so she cannot tell it is one; the ids do not point at
each other and were never meant to. Kappa rather than raw agreement, because
with most fields negative, agreeing by chance runs high and a raw 90% would read
as excellent while meaning very little.

Tested end to end by `tests/test_ingest.py`, which synthesises a session file in
`study.js`'s own hand-rolled CSV format -- commas joined by hand, only some
fields quoted -- against the real 146-section key, including a partly-finished
run. A test that wrote a tidier file than the app does would pass while the real
file failed.

Checked there: every circle lands inside the field it was drawn on and on the
right slide, radii come back in a plausible pixel range, a 40%-complete session
merges what exists and still spans all nine batches, and kappa brackets
correctly at 1.0 for identical answers and below zero for systematic
disagreement.

One guard worth knowing about: `merge` refuses to run if `key.csv` holds a field
size other than 512 px. `study.js` computes every circle diameter against a
hardcoded 253.6 µm field, so a different size would silently rescale every
diameter and radius in `cells.csv` with nothing downstream looking wrong.
