# Hosted round 1 — the key

The web page shows opaque field ids (`f001`…). It does not contain the slide
name, the score, the band or the split — not hidden, *absent*, so there is
nothing to find in devtools. `key.csv` is the mapping, and it stays here.

She returns `ballooning_answers.csv` (field_id, reviewer, verdict, seconds).
Join it back with:

```bash
.venv/bin/python -m mashpath.review.merge_hosted \
    review_sets/hosted_round_1/key.csv ~/Downloads/ballooning_answers.csv
```

That writes `verdicts.csv` beside the key, in the same format the desktop app
produces, so both rounds pool into one training set.
