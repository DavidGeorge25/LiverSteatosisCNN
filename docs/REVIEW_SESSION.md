# Running the review session

For the person operating the session, not the pathologist. Their one-page brief
is `PATHOLOGIST_BRIEF.md` — send it in advance so they arrive knowing what the
task is.

## Before they sit down

```bash
.venv/bin/python -m mashpath.review.app bal_chunks
# open the printed URL, check a few crops load
```

Have the reviewer's initials ready — the app records them per verdict, and
that is what makes inter-rater kappa computable afterwards.

## The first ten minutes are the experiment

Do NOT start the clock on 450 candidates. Sit with them through the first
10–15 and ask four questions. If any answer is bad, the remaining 435 are
worthless and you want to find that out now rather than after an hour.

**1. "Can you actually judge this from this image?"**

The crop is a fixed physical width with the proposed cell outlined. If they
find themselves wanting to pan around, zoom out, or look at an adjacent field,
the crop is too small and every verdict after that is a guess. Cheap to fix
(`ballooning.crops.padding_um`), expensive to discover later.

**2. "Is the outline helping or getting in the way?"**

Every candidate has both an outlined and an unmarked version. Some pathologists
find the outline anchors them; others find it prejudges the call. Let them
switch and say which they prefer — then use that consistently, because mixing
the two mid-session makes the verdicts non-comparable.

**3. "Am I asking the right question?"**

We ask: *is this individual cell ballooned, yes/no/unsure*. That is not how
ballooning is scored clinically — NASH-CRN grades a whole section 0–2 on how
many ballooned cells are present. A pathologist may reasonably say "I don't
call individual cells, I call slides." If so, stop and rethink the task before
spending their hour.

**4. "What fraction of these look plausible to you?"**

If they say "almost none", the detector is aimed wrong and the review set will
be 450 rejections — which is a real result, but you would rather know in ten
minutes. If they say "almost all", the vetoes are too aggressive and we are
only showing them easy calls.

## During

Leave them alone. Do not hover, and do not react to individual verdicts —
"really?" from the person who built the detector is enough to shift how
someone marks the next twenty.

Two things worth mentioning once, at the start:

- **Rejecting is the point.** The set deliberately includes cells the detector
  scored low and cells it vetoed outright. A high rejection rate means the
  sampling is doing its job.
- **"Unsure" is a real answer.** A forced call on an unclear crop is worse
  than no call, because it becomes a training label either way.

## Questions to ask afterwards

**"When you rejected something, what was usually wrong with it?"**

This is the single most valuable thing you get from the session, and the labels
do not capture it. "Most of them were just normal hepatocytes" points at the
score; "most were out of focus" points at the crops; "most were at the section
edge" points at tissue detection. Write down what they say verbatim.

**"Did you see anything that was clearly ballooned that we did NOT outline?"**

Recall is invisible in this design — we only ever show them our own proposals.
If they noticed real ballooning sitting unmarked next to a proposed cell, the
detector is missing a whole class of positives and no amount of label
collection will reveal it.

**"Did the slides look like they came from different batches?"**

The set is balanced across 9 staining runs. If the stain variation is obvious
to them, it will be obvious to a model too, and that is a confound to control
for rather than discover later.

## Afterwards

```bash
# what was recorded, and how consistent
.venv/bin/python -c "
from mashpath.review import verdicts as v
print(v.agreement_report('bal_chunks/<slide>/ballooning/review/verdicts.csv'))"
```

Report the numbers you get honestly, including the unflattering ones:

- **confirm rate** — what fraction of proposals were real
- **intra-rater agreement** — the same crop shown twice to one person; this is
  the ceiling on how well anything can score, model included
- **inter-rater kappa** (if two reviewers) — the ceiling on agreement between
  *any* two judges, and the number a model should be compared against rather
  than against 100%

Published inter-observer kappa for ballooning is modest even among experienced
liver pathologists. If your kappa comes out around 0.5, the task is hard, not
broken — but you need the number before you can claim a model is good.

## What this session cannot tell you

- **Whether the detector has good recall.** Only proposals are shown.
- **Whether the score is calibrated.** It orders candidates; the confirm rate
  per band is the first evidence about whether that ordering means anything.
- **Anything about inflammation or steatosis.** Different features, different
  questions, and inflammation's candidates are not review-ready.
