#!/bin/bash
# Source this before anything. Every Fir job and every interactive session.
#
#   source cluster/fir_env.sh
#
# Four things here are NOT optional and each cost real debugging to find:
#
# 1. Everything lives in /scratch, not $HOME. Fir's home filesystem truncates
#    files partway through a many-small-file write -- a wheel that unzips 23 of
#    524 files and reports success. `dd` of one big file passes, so it looks
#    healthy. Until Alliance fixes it, $HOME is unsafe for a venv, a pip cache,
#    or model weights. (Reported; see docs/CLUSTER.md.)
#
# 2. XDG_DATA_HOME/XDG_CACHE_HOME are redirected for the same reason -- pip and
#    virtualenv cache under ~/.local and ~/.cache by default and corrupt there.
#
# 3. LD_LIBRARY_PATH must carry libopenslide. openslide-python finds its shared
#    library through the loader, not through Python, so without this every SVS
#    read dies at import with "Couldn't locate OpenSlide shared library".
#
# 4. opencv comes from the MODULE, not from pip. Alliance ships an
#    `opencv-noinstall` shim that fails any pip install of opencv-python, and
#    it fails the whole pip command it appears in. opencv/4.12.0 carries the
#    cp312 build that matches python/3.12.
set -euo pipefail

module load StdEnv/2023 python/3.12 opencv/4.12.0 openslide/4.0.0

export MASHPATH_ROOT="${MASHPATH_ROOT:-/scratch/$USER/mashpath}"
export MASHPATH_VENV="${MASHPATH_VENV:-/scratch/$USER/venv-mashpath}"
export MASHPATH_SLIDES="${MASHPATH_SLIDES:-/scratch/$USER/slides}"
export MASHPATH_OUT="${MASHPATH_OUT:-/scratch/$USER/outputs}"

export XDG_DATA_HOME="/scratch/$USER/.local/share"
export XDG_CACHE_HOME="/scratch/$USER/.cache"
export LD_LIBRARY_PATH="$EBROOTOPENSLIDE/lib:${LD_LIBRARY_PATH:-}"

# 5. Force the newer libhdf5 to load FIRST.
#
#    Alliance's opencv module pulls an older libhdf5 into the process. h5py --
#    which StarDist reaches through csbdeep -- then loads libhdf5_hl against
#    the already-resident older library and dies:
#
#      ImportError: libhdf5_hl.so.310: undefined symbol: H5T_IEEE_F16BE_g
#
#    Import order decides it, and the pipeline loses: tissue detection touches
#    cv2 long before any model is loaded, so cv2 always wins the race. The
#    symptom is a ballooning/inflammation job that dies ~8 s in, AFTER printing
#    slide metadata and the tile count, which makes it read like a pipeline bug
#    rather than a linker one.
#
#    `import h5py` on its own succeeds, so a smoke test that does not import
#    cv2 will pass and prove nothing. Test with `import cv2, h5py` together.
HDF5_LIB=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcc12/hdf5/1.14.6/lib
if [ -f "$HDF5_LIB/libhdf5.so.310" ]; then
    export LD_PRELOAD="$HDF5_LIB/libhdf5.so.310:$HDF5_LIB/libhdf5_hl.so.310${LD_PRELOAD:+:$LD_PRELOAD}"
fi

# StarDist downloads pretrained weights on first use. Compute nodes have no
# outbound network, so the cache must be primed on a login node and must live
# somewhere the job can read -- see cluster/cache_model.py.
export STARDIST_CACHE="/scratch/$USER/.keras"
export KERAS_HOME="$STARDIST_CACHE"

# 6. Thread counts must match the allocation.
#
#    Two separate reasons, and the second one aborts the process:
#
#    (a) OpenMP defaults to one thread per CORE ON THE NODE, not per allocated
#        cpu. On a 64-core GPU node with --cpus-per-task=8 that is 8x
#        oversubscription -- every thread fighting for an eighth of a core.
#
#    (b) Alliance's BLIS checks that the thread count it got matches the one it
#        asked for, and calls abort() when they differ. TensorFlow adjusts the
#        OpenMP pool as it initialises, so the two disagree and the job dies:
#          libblis: A different number of threads was created than was requested.
#          libblis: Aborting.
#        It lands ~48 s in, after the GPU is acquired and the slide is open,
#        which makes it look like a pipeline fault rather than a BLAS one.
# 7. Swap BLIS out for OpenBLAS.
#
#    Setting the thread counts below is necessary but NOT sufficient. BLIS
#    still aborts non-deterministically once TensorFlow is in the same process
#    -- observed at 5m43 and again at 4m12 on identical input -- because TF
#    resizes its OpenMP pool while BLIS is mid-call, and BLIS treats any
#    mismatch as fatal rather than adapting:
#
#      libblis: A different number of threads was created than was requested.
#      libblis: Aborting.
#
#    FlexiBLAS makes the backend swappable at runtime, so this sidesteps the
#    whole argument. OpenBLAS handles a changing thread pool without aborting.
#    Note that abort() does not flush stdio, so the job log loses every line
#    the pipeline had printed -- which is why these failures look like a job
#    that did nothing for four minutes. Hence PYTHONUNBUFFERED below.
export FLEXIBLAS=openblas

# Unbuffered, always. A job killed by abort() or by the scheduler loses
# everything still sitting in the stdout buffer, and a log that stops at the
# banner is indistinguishable from a job that hung at startup.
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export BLIS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# BLAS stays SINGLE-threaded. Ballooning's CPU cost is watershed, GLCM and
# local entropy -- all single-threaded skimage/scipy that barely touch BLAS --
# so multi-threaded BLAS buys nothing here and costs a hang: with TensorFlow in
# the same process, an OpenBLAS worker busy-waits forever and the job sits at
# 95% of ONE core with the GPU at 0%, having stopped mid-tile. Observed on tile
# 675/1187 after 47 minutes of steady 4s/tile progress; the same tile completes
# in 2.5s on a laptop, so it is the thread interaction, not the data.
export OPENBLAS_NUM_THREADS=1
# Yield instead of spinning when a parallel region ends. Without this an idle
# OpenMP team burns a core waiting for work that never comes.
export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# TensorFlow is loud and none of it is actionable inside a batch job.
export TF_CPP_MIN_LOG_LEVEL=3
export KERAS_BACKEND=tensorflow

source "$MASHPATH_VENV/bin/activate"
export PYTHONPATH="$MASHPATH_ROOT:${PYTHONPATH:-}"
