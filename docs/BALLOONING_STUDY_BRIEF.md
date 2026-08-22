# Ballooning — annotation study

Thank you for doing this. One page, about two minutes. Everything here is also
in `README.txt` inside the folder, so nothing is lost if this note is.

## The package

`Ballooning_Study` — 146 sections, 18 fields each, spread over as many sittings
as you like. Doing all of it is roughly 11 hours, and **there is no expectation
that you will.** There is no deadline and no need to finish.

The sections are ordered so that stopping anywhere still leaves a fair sample:
all nine staining batches appear within the first fifteen sections, and the
scarcest and most valuable tissue is front-loaded rather than left to chance.
Whatever you get through is usable as it stands.

**One ask before you commit a long stretch to it.** Do two or three sections and
then stop for a moment — after the second the tool asks whether this is working,
and emailing us the file as it stands at that point costs you fifteen minutes.
It is the cheapest chance to change anything, and everything you have done still
counts.

## What we are asking

Two separate questions.

**1. In each field, circle every ballooned hepatocyte.** Click to place a
circle, drag the corner handle to size it to the cell. Then <kbd>Y</kbd> to
record the field, <kbd>N</kbd> if there are none, <kbd>U</kbd> if you cannot
judge it.

**2. After the last field of a section, grade that section 0 / 1 / 2** on the
NASH-CRN ballooning scale — none, few, many. You are shown all of that section's
fields together to grade from.

The fields are a **uniform random sample** of each section. Nothing was
pre-selected by an algorithm, so what you see is representative of the whole
section and "few versus many" means what it usually means.

## What counts as ballooned

The usual criteria, for the avoidance of doubt rather than as instruction —
where your reading differs from this list, follow yours and please tell us:

- **enlarged** relative to the hepatocytes around it — the comparison is local,
  which is why the surrounding tissue is always on screen to the right
- **rounded**, losing the polygonal outline
- **rarefied or flocculent cytoplasm** — wispy and cleared rather than smoothly
  eosinophilic
- often a **hyperchromatic nucleus**, sometimes displaced
- frequently **near a terminal hepatic venule** (zone 3)

A hepatocyte distended by a single large fat droplet is not ballooning. Neither
is a pale cell that is not enlarged.

## Please use "unsure" freely

A forced call becomes a training label indistinguishable from a confident one,
and a model trained on guesses learns to guess. Ambiguity is information here.
If a field is genuinely borderline, <kbd>U</kbd> is the right answer and is more
useful to us than a coin flip.

## The tool

| | |
| --- | --- |
| click | place a circle |
| drag | move it; drag the corner handle to resize |
| red **×** | remove that circle |
| <kbd>Y</kbd> / <kbd>N</kbd> / <kbd>U</kbd> | yes (with your circles) / none here / cannot judge |
| <kbd>←</kbd> | back to the previous field |
| scroll | zoom; <kbd>0</kbd> resets |

Pressing <kbd>Y</kbd> with nothing circled asks once whether you meant to, then
takes it if you press <kbd>Y</kbd> again.

## Saving

On **Begin** it asks where to keep your results file — Documents is fine. Every
answer is written to that file the moment you make it; there is no save button
to forget. Close the tab whenever you like, mid-section is fine, and reopen with
the same initials to carry on where you stopped.

## If something looks wrong

Stop and email David George, davidg1091620@gmail.com. Nothing is lost by
stopping. There is also a short "is this working?" prompt after the second
section — please use it. A problem found early is much cheaper than one found
at the end, and "this is the wrong question" is a finding we need to hear.
