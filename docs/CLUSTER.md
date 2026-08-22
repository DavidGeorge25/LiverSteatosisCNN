# Running on Fir (Digital Research Alliance of Canada)

Account `def-wangd123`. Everything lives in `/scratch`, not `$HOME` — see
*Known Fir problems* below before changing that.

```
/scratch/$USER/
  mashpath/      the repo (rsync'd from the laptop)
  venv-mashpath/ the Python environment
  slides/        SVS files + slide_list.txt
  outputs/       per-feature results
  logs/          SLURM stdout/stderr
  .keras/        StarDist weights (primed on a login node)
```

## First-time setup

```bash
ssh fir                                    # answer Duo once
cd /scratch/$USER/mashpath
bash cluster/setup_fir.sh                  # builds the venv from Alliance wheels
source cluster/fir_env.sh
python cluster/cache_model.py              # MUST run on a login node
```

## Running

```bash
source cluster/fir_env.sh
bash cluster/slides.sh                     # writes slides/slide_list.txt
N=$(wc -l < $MASHPATH_SLIDES/slide_list.txt)

sbatch --array=1-$N%20 cluster/steatosis.sbatch      # CPU
sbatch --array=1-$N%8  cluster/ballooning.sbatch     # GPU
sbatch --array=1-$N%8  cluster/inflammation.sbatch   # GPU
```

`%20` / `%8` cap concurrent tasks. Raise once you know the queue is not
starving other users; the GPU allocation is the scarcer resource.

## Which stages are GPU-bound

| stage | device | why |
| --- | --- | --- |
| tissue detection | CPU | Otsu on a level-2 thumbnail. Seconds. |
| tiling | CPU | Index arithmetic. |
| **steatosis** | **CPU** | Thresholding + connected components. No network involved. Requesting a GPU idles it. |
| **ballooning** | **GPU** | StarDist inference per tile, then CPU watershed + texture. Inference dominates. |
| **inflammation** | **GPU** | StarDist at `upscale: 2` — 4x the pixels of ballooning, and the most expensive stage. |
| review export | CPU | Region reads + PNG writes. I/O-bound. |

Measured on one H100 80GB HBM3: StarDist is ~57 ms per 512x512 tile. The
CPU-side work (watershed, GLCM texture, region reads) runs between inference
calls and is not overlapped, so wall clock per slide is **not** simply
tiles x 57 ms — see the timing table in the run notes.

## Known Fir problems

**`/home` truncates writes.** Extracting a 524-file zip into `/home` produced
23 files with "probably truncated" warnings; the identical extraction into
`/scratch` completed cleanly. Large single-file writes (`dd`, 20 MiB) succeed,
so quota and basic I/O look healthy and the fault is invisible until a venv or
a wheel silently half-installs. Reported to support 2026-08-11. Until it is
fixed, `fir_env.sh` redirects `XDG_DATA_HOME` and `XDG_CACHE_HOME` into
`/scratch` and nothing is installed under `$HOME`.

**`module` needs a login shell.** `ssh fir 'module load X'` appears to succeed
and silently leaves Python at the gentoo default (3.11.4). Use
`ssh fir 'bash -lc "module load X && ..."'`.

**`python -m venv` fails** with an `ensurepip` error. Alliance disables it;
use `virtualenv --no-download`. If a previous attempt failed partway, delete
`$XDG_DATA_HOME/virtualenv` first — a corrupt wheel cache reports as
`RuntimeError: no .dist-info at .../CopyPipInstall/pip-24.0-py3-none-any`.

**OpenCV must come from the module.** Alliance ships an `opencv-noinstall`
shim that fails any `pip install opencv-python` — and fails *the entire pip
command it appears in*, so batching it with other packages silently installs
none of them. `opencv/4.12.0` carries the cp312 build matching `python/3.12`.

**openslide-python is not in the wheelhouse.** Install it from PyPI (login
nodes have PyPI access) and load `openslide/4.0.0` for the native library.
`LD_LIBRARY_PATH` must include `$EBROOTOPENSLIDE/lib` or every SVS read dies
at import with "Couldn't locate OpenSlide shared library".

**StarDist 0.9.1 needs `pkg_resources`**, removed in setuptools 81+. Pin
`setuptools<81` in the venv.

**Compute nodes have no outbound network.** `StarDist2D.from_pretrained()`
will hang then fail inside a job. Run `cluster/cache_model.py` on a login node
first; `KERAS_HOME` points into `/scratch` so jobs can read the cache.

**`cv2` and `h5py` fight over libhdf5.** The `opencv` module loads an older
libhdf5 into the process; h5py (reached via csbdeep, via StarDist) then fails
with `libhdf5_hl.so.310: undefined symbol: H5T_IEEE_F16BE_g`. Whichever
imports first wins, and the pipeline always loses -- tissue detection touches
cv2 before any model loads. `fir_env.sh` LD_PRELOADs the newer libhdf5 to
settle it. Note that `import h5py` ALONE succeeds, so a smoke test that omits
cv2 passes and proves nothing; always test `import cv2, h5py` together.

**Cold-node CVMFS can serve a partial library.** Distinct from the above and
much rarer: a shared object read on a node with a cold cache can come back
incomplete, producing an `undefined symbol` that disappears on resubmit. If a
handful of array tasks fail with linker errors and the rest succeed, retry
before debugging.

**Do not pass `--partition`.** Fir's scheduler routes jobs itself and an
explicit partition is usually rejected with "The specified partition does not
exist, or the submitted job cannot fit in it".

**`/scratch` is purged.** Alliance expires scratch files (typically 60 days).
Anything that must survive belongs in project space — which this account does
not currently have. Request one before treating `/scratch/$USER/outputs` as
durable.

## Getting data in

`cluster/fetch_data.sh` pulls from OneDrive with rclone. It runs on a **login
node** because compute nodes have no network. Do not route bulk data through a
laptop: measured upstream on this one was ~12 MB/min, which puts 25 GB at
roughly 35 hours; OneDrive to Fir is a datacentre-to-datacentre transfer.


## OPEN: steatosis is ~30x slower in a batch job than interactively

Unresolved as of 2026-08-15. It does NOT block the review round; it only
limits cohort-wide steatosis, and `outputs/dataset_v1` (7,650 labelled tiles,
51 slides) is already trainable without it.

**What is established.**

* One slide (`R26-122-35_HE_94`, 1324 tiles) profiles at 45-66 ms/tile when
  the loop body is driven directly on a compute node, but exceeds 1.36 s/tile
  inside `mashpath.cli steatosis` in a batch job. 218 of 219 slides timed out.
* The cost is kernel time, not computation:  `real 10m / user 1m29 / sys 8m22`.
* It is NOT the slide (both fast and slow slides are `JPEG/RGB Q=70`, 256x256
  tiles, same scanner), NOT Lustre (staged to node-local disk in 0-2 s; reads
  measure 4-5 ms), NOT the tile list (identical coordinates on both machines),
  NOT thread oversubscription (`cv2.getNumThreads()` correctly returns the
  cgroup allocation), NOT a submission stampede (staging is 0-2 s), and NOT
  syscalls (`strace -c` totals 0.25 s, mostly Python imports at startup).
* `Tile.read` is a thin wrapper over `slide.read_region`, so the profiled path
  and the real path are the same code.

**Most likely remaining explanation.** System time that is not syscalls is
page-fault handling. `detect_fat` allocates several ~1 MB arrays per tile;
glibc's default `mmap` threshold is 128 KB, so each one is mmap'd, freed, and
re-faulted on next touch. The standard mitigation is to stop glibc returning
those arenas:

    export MALLOC_MMAP_THRESHOLD_=1073741824
    export MALLOC_TRIM_THRESHOLD_=1073741824

That is UNTESTED -- the comparison run errored before producing numbers.

**How to test it properly.** Run the real CLI twice on one slide with
`--limit 100`, once with those variables and once without, capturing
`/usr/bin/time -f 'real=%e user=%U sys=%S minflt=%R'` and NOT redirecting
stderr, so a failure is visible rather than silent. If `minflt` differs by
orders of magnitude, that is the answer.

**Method note.** Five hypotheses were tested and rejected before this one, each
because the reproduction was simpler than the real job -- profiling the loop
body instead of running the CLI. The `strace`/`time` measurement that actually
narrowed it should have come first.
