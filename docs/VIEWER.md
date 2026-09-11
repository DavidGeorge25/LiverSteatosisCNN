# The slide viewer: an exhaustive measurement, and a way to look at it

`mashpath/viewer/` turns a trained fold into two things a PI can be shown: a
continuous fat percentage, and the spatial map that percentage came from. There
is no 0–3 grade anywhere in it, and that is a design constraint rather than an
omission — the argument for this project is that a continuous, objective
measurement carries information an ordinal grade throws away, so the interface
must not quietly reintroduce one.

Two measurements shaped the design more than any preference did. Both are
reproducible from this repo and both are load-bearing.

---

## 1. The model works at exactly one magnification

The U-Net was trained on 512 px **level-0** tiles at 0.4953 µm/px. Every one of
the 259 slides in `dataset_v2` carries that same mpp — one scanner, one
magnification — and `train/augment.py` is colour-only: the geometry is flips and
90° rotations, and there is no scale jitter anywhere in the pipeline. The model
has therefore never seen a fat droplet at any size other than its native one.

Run it on a downsampled pyramid level and it does not degrade gracefully, it
stops firing. Measured on identical windows of `R25-264-1`:

| | level 0 (native) | level 1 (4× down) | level 2 (16× down) |
| --- | ---: | ---: | ---: |
| mean fat over 6 tissue windows | **14.54%** | 0.81% | 0.06% |
| ratio to native | 1.00× | **0.06×** | 0.00× |

The obvious way to make a whole-slide viewer feel fast — a coarse overview pass
at a low pyramid level, refined on zoom — would therefore open a fatty slide
reading **0.81%** and jump to ~15% when the user zoomed in. For a demo whose
entire claim is a stable objective number, a headline that moves 17-fold with
zoom is the worst available failure.

**So inference never runs at a display scale.** It runs once, at native
resolution, over the whole slide. What the viewer downsamples for a zoomed-out
view is the *mask*, never the input. Display scale and inference scale are
decoupled, and that is why the percentage on screen does not move as the user
navigates. `dzi.py` carries this rule; `tests/test_viewer.py` pins it.

The general form matters for the first slide that arrives from another scanner:
**resample by mpp, never by pyramid level.** `precompute.py` does this
automatically and records `resampled_to_training_mpp` in the artifact.

## 2. Every number this project has published so far is a 6% sample

`dataset_v2` exports 150 tiles per slide; the survey used 60. Both are uniform
random samples of the tissue tiles, and both are small. Measured on
`R25-264-1` (2,487 tissue tiles):

| tiles | coverage | estimate | 95% CI |
| ---: | ---: | ---: | ---: |
| 25 | 1.0% | 15.74% | ±1.94 pp |
| 50 | 2.0% | 15.89% | ±1.44 pp |
| 100 | 4.0% | 15.04% | ±1.06 pp |
| 200 | 8.0% | 14.34% | ±0.74 pp |
| **all 2,487** | **100%** | **14.35%** | — |

So the figures in `STEATOSIS_TRAINING.md` — "NASH 7.93%", "CCl4 floor 0.167%",
"chow 0.06–0.18%" — each carry roughly ±0.5–1 pp of sampling noise that has
never been reported next to them. Quoting two decimals off that is the precise
thing a reviewer attacks, and it is a *weaker* position than an ordinal grade,
which at least does not claim precision it lacks.

And this is not theoretical. `R25-264-1` measured both ways, same fold, same
preprocessing:

| method | result |
| --- | ---: |
| the 150 tiles `dataset_v2` exported (6.0% of the slide) | 14.10% |
| **every one of its 2,487 tissue tiles** | **15.97%** |

**1.87 pp apart — 13% in relative terms**, on a slide where the sample was drawn
correctly and uniformly. Nothing was done wrong; that is simply what a 6% sample
of a spatially structured section costs. Reported to two decimals, the first
number is precise about the wrong thing.

**So the viewer measures every tissue tile.** The result is a census, not an
estimate: there is no sampling term left, no confidence interval is needed, and
two decimals are defensible. It is also cheap — see the timings below.

An artifact built with `--limit` is marked `coverage: partial` and the interface
says **"a sample, not a census"** in place of the census badge. That path exists
for smoke tests only.

---

## What the artifact contains

One directory per slide, written by `mashpath.viewer.precompute`:

```
<out>/<slide>/
  meta.json           the census, the droplet and zonal summaries, and the
                      full provenance of the fold that produced them
  tiles.csv           one row per tissue tile: fat px, tissue px, mean
                      probability, ambiguous px, droplet counts
  mask/l0/<x>_<y>.png full-resolution mask, on the same 512 px level-0 grid
                      the model ran on. Tiles with no fat are NOT written --
                      a missing file reads as zero, which is why a whole
                      slide's mask is a few MB rather than 180
  mask/ds4.png        one 4x-downsampled coverage map, for zoomed-out views
```

`meta.json` always carries `validated_against_reference: false`. Full coverage
makes the number precise, not correct: there is still no reference
segmentation (`LIMITATIONS.md` §1) and `outputs/annotation_set_v1/` is still the
gate. The viewer prints that caveat rather than leaving it to be remembered — a
convincing overlay is exactly the thing that tempts a reader to assume the
question is settled.

---

## Running it

### The census (GPU, one slide per array task)

```bash
source cluster/fir_env.sh
sbatch --array=1-$(wc -l < /scratch/$USER/demo_slides.txt) \
       cluster/viewer_precompute.sbatch \
       /scratch/$USER/demo_slides.txt \
       /scratch/$USER/outputs/unet/lobo/2025-08-25HE/best.keras
```

One slide per task, staged to `$SLURM_TMPDIR`. **The staging is not optional**:
a tile read straight off Lustre costs ~82× what the same read costs on
node-local NVMe (205 s against 2.5 s, measured — see `cluster/steatosis.sbatch`),
which turns a four-minute pass into a five-hour one.

The slide list is a file with one path per line, read with `sed` and quoted
everywhere, because this cohort has a batch directory called
`2025-03-28_Celina 656D H&E` and splitting on whitespace silently drops it.

Locally, without a cluster:

```bash
python -m mashpath.viewer.precompute path/to/slide.svs \
    --checkpoint outputs/unet/lobo_fir/2025-08-25HE/best.keras \
    --out outputs/viewer/<slide>
```

### Serving it

```bash
python -m mashpath.viewer.server --root outputs/viewer \
    --checkpoint outputs/unet/lobo_fir/2025-08-25HE/best.keras
```

`--checkpoint` is optional and only enables the live spot check. Without it the
server never imports TensorFlow, which is several seconds and a lot of memory
it does not otherwise need.

Stdlib `http.server`, bound to `127.0.0.1`, for the same two reasons as
`review/app.py`: a demo that fails because a dependency did not resolve on Fir
is a demo that did not happen, and these tiles are patient-adjacent material
served without authentication, so the only reachable surface should be an SSH
tunnel. OpenSeadragon is **vendored** into `static/` rather than loaded from a
CDN — compute nodes have no outbound network, and a CDN link is a blank page.

---

## The day-to-day workflow on Fir

The census needs a GPU. **Serving does not** — the server reads precomputed
masks off disk and region-reads the slide, and the only thing that would want a
GPU is the live spot check, which is one region on demand. So the demo session
is a *CPU* allocation, which matters: `gpubase_interac` has a single node, and
queuing for a scarce H100 while your PI is standing there is an avoidable risk.

```bash
# 1. once, ahead of time: the census (GPU batch job, ~1 min a slide)
sbatch --array=1-3 cluster/viewer_precompute.sbatch slides.txt <checkpoint>

# 2. on demo day: a CPU session that serves it
sbatch cluster/viewer_serve.sbatch 8000 "" "" 0.0.0.0
#   ...or interactively:
#   salloc --account=def-wangd123 --cpus-per-task=4 --mem=16G --time=3:00:00

# 3. the log prints the node name and the exact tunnel command
cat /scratch/$USER/logs/serve_<jobid>.out

# 4. from the laptop, second terminal
ssh -N -L 8000:<node>:8000 fir
# then open http://localhost:8000
```

### The tunnel gotcha, which costs an hour if you meet it cold

`ssh -L 8000:<node>:8000 fir` has the **login node** open the connection to
`<node>:8000`. If the server is bound to `127.0.0.1` on the compute node, the
login node cannot reach it and the tunnel fails with nothing useful in the log.
Two ways out, and `viewer_serve.sbatch` prints the right command for whichever
you chose:

| bind | tunnel | trade |
| --- | --- | --- |
| `0.0.0.0` | `ssh -N -L 8000:<node>:8000 fir` | works immediately; the port is reachable by other jobs on the cluster's internal network for the life of the job |
| `127.0.0.1` (default) | `ssh -J fir $USER@<node> -N -L 8000:127.0.0.1:8000` | stays private, but needs your public key in `~/.ssh/authorized_keys` on Fir — which does not exist on this account yet |

For a 30-minute demo on de-identified mouse material, `0.0.0.0` is the
pragmatic choice and is what the measured workflow above uses. If this is going
to run regularly, spend the two minutes on `authorized_keys` and keep the
loopback bind.

One more, if you use a multiplexed SSH connection (this repo's `~/.ssh/config`
does, for Duo): a *new* `ssh -L` cannot add a forward to an existing master and
will silently fall back to a fresh connection that re-prompts for Duo. Ask the
master to add it instead — `ssh -O forward -L 8000:<node>:8000 fir` — and check
first that the local port is actually free, because a stale forward holding it
reports as the same generic "Port forwarding failed".

**Why interactive rather than a batch job with a persistent endpoint.** A batch
job would survive disconnection, which sounds better, but it buys nothing here
and costs the thing that matters: with `salloc` you can restart the server in
two seconds when something needs changing, and you can see it fail. A demo is a
30–60 minute event inside a 3-hour allocation. The one real argument for a batch
job is if the GPU-backed live inference had to stay warm for hours, and it does
not.

**Why not a login node.** Alliance discourages sustained work there, and the
82× Lustre penalty applies to every region read, so panning would be visibly
slow. If you must (no allocation available, five minutes to spare), it will
work — the artifacts are small reads — but stage the slides first.

---

## Measured performance

The census, on one H100 with the slide staged to node-local NVMe
(`2025-08-25HE` fold, 7.8M parameters):

| slide | tiles | wall clock | per tile | census |
| --- | ---: | ---: | ---: | ---: |
| `R22-354_3_WT-CHOW3` | 843 | **45 s** | 53 ms | 0.07% |
| `R22-354_23_ACLY656 NASH4` | 1,485 | **61 s** | 41 ms | 12.04% |
| `R25-264-22` | 2,272 | **61 s** | 27 ms | 3.23% |

**A whole slide is about a minute.** That settles the question the progressive
design existed to answer: with a GPU there is nothing to be progressive about.
Precompute once, serve instantly, and the viewer never waits on the model at
all. The same pass on a laptop CPU is 586 s for 2,487 tiles (236 ms/tile), which
is still fine for a one-off but is why the cohort-wide run belongs on the
cluster.

At 27–53 ms/tile on an H100 the bottleneck is no longer inference — it is the
region read, the tissue mask, the connected-component pass and the PNG write.
There is room to speed this up; there is no reason to.

Laptop CPU, for the pieces that are the same everywhere:

| stage | cost |
| --- | ---: |
| open slide + tissue detection | 0.2 s |
| enumerate 2,487 tissue tiles | 0.02 s |
| level-0 tile read (512 px) | 6.8 ms |
| tissue mask per tile | 0.3 ms |
| live spot check, 2048² region, 4 CPU cores | ~2 s |

Tile serving, once the census exists (this is what the demo feels like):

| level | slide tile | overlay tile |
| --- | ---: | ---: |
| 16 (full resolution) | 7–16 ms | 2–3 ms |
| 14 | 84 ms | 12 ms |
| 12 | 149 ms | 97 ms |
| 8 (whole slide) | 114 ms | 161 ms |

The overlay and the tissue are verified to agree **pixel for pixel** at full
resolution and to have identical tile dimensions at every level including the
ragged right and bottom edges (`tests/test_viewer.py`). If those two pyramids
drifted, the overlay would slide against the tissue as you zoom, and it would
read as the model being wrong rather than as a viewer bug.

---

## What to show, and on which slides

Every fold has seen seven of the eight staining batches, so most slides are
training data for whichever checkpoint you pick and cannot serve as evidence.
Two exceptions matter:

* **R22-354** (`2025-03-28_Celina 656D H&E`) is the reserved batch — excluded
  from every fold by construction, and the only slides in the project with
  known diet.
* Each fold's own **held-out batch**.

With the `2025-08-25HE` fold, all three demo slides are unseen:

| role | slide | diet | census | why it is honest |
| --- | --- | --- | ---: | --- |
| clean | `R22-354_3_WT-CHOW3` | chow | **0.07%** | reserved batch, never trained on by any fold |
| in between | `R25-264-22` | — | **3.23%** | this fold's held-out batch |
| fatty | `R22-354_23_ACLY656 NASH4` | NASH | **12.04%** | reserved batch, never trained on by any fold |

A **172× separation** between the chow control and the NASH animal, both
measured exhaustively, both unseen by the model. The chow/NASH pair is also the
project's one external validation, so the demo and the evidence are the same
object — which is the property that makes this worth showing rather than just
worth looking at.

---

## What the interface deliberately does not do

* **No 0–3 score, no severity bins, no category labels.** `tests/test_viewer.py`
  asserts the artifact carries none, so the viewer cannot render one by
  accident.
* **No confidence from the nine-way vote.** That confidence map exists only for
  the exported training tiles — it is a property of the pseudo-labels, and a
  slide the ensemble never scored has none. What is shown is the model's own
  sigmoid: the mean probability of the pixels it called fat, and the fraction of
  tissue sitting in the ambiguous 0.35–0.65 band. Labelled as such in the panel.
* **No accuracy claim.** See `validated_against_reference` above.
