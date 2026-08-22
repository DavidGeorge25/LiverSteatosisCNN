# Reviewing candidate cells — a one-page brief

Thank you for doing this. This page explains what the software did, what we are
asking you to judge, and what happens to your answers. It should take about two
minutes to read, and there is nothing technical you need to know.

## What the software did

We are building a tool that measures liver histology in mouse MASH models
automatically, so that a whole study can be scored consistently instead of a
handful of slides being scored by eye.

Some things the software can already find on its own. **Fat droplets** are a
solved problem: a macrovesicular droplet is a round clear hole in the tissue,
and a computer can identify that geometrically without being taught. We are not
asking you about steatosis.

**Hepatocyte ballooning is different.** There is no shape or colour that defines
it absolutely. A hepatocyte is ballooned because it is markedly larger and paler
than the hepatocytes *around* it — a judgment that is relative, varies between
lobules, and is exactly the kind of thing a computer cannot learn without being
shown examples first.

So the software has done the part it can do: it segmented every hepatocyte on
the slide, measured each one against its immediate neighbours, and flagged the
cells that stand out. What it has produced is a list of **candidates** — cells
worth a human looking at. It has no idea which of them are actually ballooned.
That is the part we need you for.

## What we are asking

You will see one cell at a time, outlined in red, with the surrounding tissue in
frame so you can compare it against its neighbours. For each one:

- **Yes** — this is a ballooned hepatocyte
- **No** — it is not
- **Unsure** — you cannot tell from this image

That is the whole task. There is no grading, no scoring, no counting.

**Please expect to answer "no" often.** The software was deliberately set to
propose too much rather than too little, and the set you are shown deliberately
includes cells it thought were unremarkable as well as cells it flagged
strongly — otherwise we would only ever learn what a florid example looks like
and never where the line actually falls. A high rejection rate is the system
working correctly, not failing.

**"Unsure" is a real answer.** Please use it rather than guessing. A cell you
genuinely cannot call is useful information — if many candidates are unclear,
that tells us the images we are producing are not good enough to judge from, and
that is something we need to fix rather than paper over.

Some cells will appear more than once. That is intentional and lets us measure
how consistent the labelling is. Please just answer each one as you find it.

## What happens to your answers

Your verdicts become the training examples for a model that will then score
ballooning across the full study automatically. Concretely:

- Cells you confirm teach it what ballooning looks like.
- Cells you reject teach it what *nearly* looks like ballooning but is not —
  which matters just as much, and is the part most such projects get wrong.
- Cells marked unsure are excluded from training entirely, rather than being
  forced into one category.

If two people review, we also compare their answers to each other, and each
person's repeated cells to their own. That gives us an honest measure of how
reproducible the task is. This is a property of the task, not a test of the
reviewer — ballooning is known to have modest inter-observer agreement even
among experienced liver pathologists, and knowing the real figure for this
material tells us the ceiling on how well any automated method could possibly
do. If the model later agrees with you 80% of the time and two pathologists
agree with each other 75% of the time, the model is performing at the limit of
the task.

Nothing here is used to assess anyone's performance, and no result will be
published that attributes a verdict to a named reviewer.

## Practical notes

- Roughly one hour, around 400–450 cells. You can stop at any point; everything
  answered up to then is kept and remains usable.
- Answers save as you go. If the session is interrupted, nothing is lost.
- If a crop looks wrong — nothing outlined, tissue folded, section artefact,
  out of focus — mark it unsure and add a note if you have time. Those tell us
  the image pipeline has a problem, which is more valuable than a forced call.
- If you find yourself thinking "this is the wrong question" about a lot of
  them, please say so and stop. That would mean the candidate generation is
  aimed incorrectly, and spending an hour labelling it would not fix that.

## Questions worth asking us before you start

- *What magnification am I effectively looking at?* Each crop is a fixed
  physical width, so the scale is the same in every image.
- *Should I apply NASH-CRN criteria?* Judge each cell as a cell. We are not
  asking for a slide-level ballooning grade — that comes later, from counts.
- *What if a cell is at the edge of the crop?* Mark it unsure; a cell you
  cannot see whole cannot be judged, and we would rather re-crop it.
