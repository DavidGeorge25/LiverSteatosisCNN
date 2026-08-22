# Ballooning — tile labelling

Thank you for doing this. One page, about two minutes.

## The question

You will see one field of liver at a time. For each one, a single call:

> **Does this field contain at least one ballooned hepatocyte?**

**Yes** / **No** / **Unsure**. No counting, no grading. One ballooned cell is
enough for "yes"; twenty is still "yes".

If you answer yes, please also **click the cell you mean** before moving on — a
single click, not an outline. It takes a second, and it is what lets us check
the model is looking at the same cell you were rather than at something else in
the field that happens to correlate with it. Click more than one if you like;
<kbd>Backspace</kbd> undoes the last. A plain yes with no click still counts.

We are asking at field level rather than cell level deliberately. Individual
ballooned cells are the part experienced liver pathologists disagree about
most, so a cell-by-cell ground truth would partly record who drew it. "Is there
one in this field" is closer to the judgment the grading criteria are built on,
and it is roughly ten times faster, which is the difference between this being
a few afternoons and a few weeks.

## What counts

The usual criteria, for the avoidance of doubt rather than as instruction —
where your reading differs from this list, follow yours and please tell us:

- **enlarged** relative to the hepatocytes around it — the comparison is local,
  which is why the neighbouring tissue is always on screen
- **rounded**, losing the polygonal outline
- **rarefied / flocculent cytoplasm** — wispy and cleared rather than smoothly
  eosinophilic
- often a **hyperchromatic nucleus**, sometimes displaced
- frequently **near a terminal hepatic venule** (zone 3)

A hepatocyte distended by a single large fat droplet is not ballooning. Neither
is a pale cell that is not enlarged.

## Borderline cases

**Use "unsure" rather than forcing a call.** A forced guess becomes a training
label either way and quietly teaches the model the wrong boundary; an "unsure"
is excluded from training entirely, which is the correct handling. Please also
use it when the *image* is the problem — out of focus, folded, a section
artefact, mostly background. Those tell us the pipeline needs fixing, which is
more useful to us than a call you did not really make.

There is a note box if you want to say why, but it is optional and never
required.

## The tool

Open the address we give you, type your initials, press Start.

| key | |
| --- | --- |
| <kbd>Y</kbd> | yes — at least one ballooned hepatocyte |
| <kbd>N</kbd> | no |
| <kbd>U</kbd> | unsure |
| <kbd>←</kbd> | back, to change the previous answer |
| <kbd>C</kbd> | open the surrounding-tissue view full size |
| scroll / <kbd>0</kbd> | zoom in and out / reset |

The panel on the right always shows the surrounding tissue with your field
outlined in blue, so you can judge size against the neighbours without leaving
the field. Scroll to zoom into the field itself when you need cytoplasmic
detail.

**Everything saves as you answer.** Close the window whenever you like; when
you come back, type the same initials and it resumes at the next unanswered
field. There is no submit button and nothing to lose.

## Time

There are **1,400 fields** in total, which we expect to be about **three
sessions of two hours** — roughly 550 a session, at an estimated 11 seconds
each.

That 11 seconds is our estimate and has never been measured, so the first
session is partly measuring it. Please work at whatever pace the images
actually need and let the total fall where it falls; if it is running much
slower than this, that is information we need rather than a problem with how
you are working. There is a smaller **300-field pilot** we would like you to do
first, for exactly that reason — if anything about the task or the images is
wrong, we would rather find out in one sitting than after three.

Some fields appear more than once, deliberately and far apart. That measures
how reproducible the task is, which sets the ceiling on how well any automated
method could do. Just answer each as you find it.

## Two things worth saying out loud

**Expect to answer "no" a lot.** The set deliberately mixes fields our current
software scored high with fields it scored low, because where you disagree with
it is the most useful signal we can get. A high "no" rate means the sampling is
doing its job.

**If the question feels wrong, please stop and say so.** If after twenty fields
you are thinking "this is not judgeable at this magnification", or "you are
asking the wrong thing", we would much rather find that out in ten minutes than
after four sessions.
