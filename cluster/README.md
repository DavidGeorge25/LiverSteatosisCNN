# Running mashpath on Fir

Measured numbers, not estimates: 52 readable slides, **84,562 tiles** total
(median 1,193/slide, max 4,554). Rates below are from this laptop; the GPU
column is what changes on an H100.

| stage | s/tile | serial total | GPU? |
|---|---|---|---|
| tissue + tile plan | 0.004 | 0.1 h | no — I/O bound |
| steatosis | 0.033 | 0.8 h | **no** — classical CV, no model |
| inflammation (stages 1–2) | 0.78 | 18.3 h | partly — 41% StarDist |
| ballooning | 2.04 | 47.9 h | partly — 41% StarDist |

## What is actually GPU-bound

Only StarDist. Profiled per tile at upscale 2:

```
slide read             0.01s    0.6%
StarDist               0.83s   40.6%   <- GPU
watershed expand       0.39s   19.0%       CPU
texture + regionprops  0.81s   39.8%       CPU
```

So an H100 shortens 41% of ballooning and nothing else. **Amdahl caps the
per-task speedup near 2x** however fast the GPU is — which is why the job
scripts ask for 8 CPUs alongside each GPU. The CPU remainder is the critical
path, and a GPU task starved of cores is slower than the laptop.

Throughput comes from the array, not from the device: 53 independent tasks at
`%8` concurrency turns 48 serial hours into roughly 3–6 wall-clock hours.

**Do not request a GPU for steatosis.** It cannot use one, and the GPU
partition queues far longer than the CPU partition.

## Order

```bash
# once, login node
bash cluster/setup_fir.sh
source ~/venvs/mashpath/bin/activate
python cluster/cache_model.py          # compute nodes have no network

# data (see the header of fetch_data.sh -- needs one browser consent first)
bash cluster/fetch_data.sh
python cluster/verify_slides.py ~/projects/def-<pi>/slides

# jobs
mkdir -p logs
export SLIDE_DIR=~/projects/def-<pi>/slides
export OUT_DIR=~/scratch/mashpath-out

sbatch cluster/steatosis.sh
sbatch cluster/ballooning.sh
sbatch cluster/inflammation.sh
```

Set `--array=0-N` to `(number of slides - 1)`. One slide in the current cohort
(`R26-122-14_HE.svs`) does not open in OpenSlide — exclude it or that task
burns a GPU allocation to crash.

## Reviewing from the cluster

The review app binds to localhost and has no authentication, deliberately.
Tunnel to it rather than exposing it:

```bash
# laptop
ssh -L 8000:localhost:8000 fir
# on fir
source ~/venvs/mashpath/bin/activate
python -m mashpath.cli review-app ~/scratch/mashpath-out --port 8000
# then open http://localhost:8000 on the laptop
```

Better for a real session: `rsync` the review packages down and run the app
locally. They are small (crops only, not slides), and a pathologist should not
be waiting on an SSH tunnel between candidates.

```bash
rsync -av --include='*/' --include='review/***' --exclude='*' \
    fir:~/scratch/mashpath-out/ ./outputs/
```

## Storage

| path | for | note |
|---|---|---|
| `~/projects/def-<pi>/slides` | the SVS files | backed up, ~1 TB default |
| `~/scratch/mashpath-out` | pipeline outputs | large, **purged ~60 days** |
| `$HOME` | code + venv | ~50 GB, do not put data here |

Copy anything you care about out of `scratch` before the purge window.
