#!/bin/bash
#SBATCH --job-name=mash-viewer
#SBATCH --account=def-wangd123
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/%u/logs/viewer_%A_%a.out
#SBATCH --error=/scratch/%u/logs/viewer_%A_%a.out
#
# One slide per array task: the exhaustive, native-resolution pass that
# produces the viewer's artifact and the number of record.
#
#   source cluster/fir_env.sh
#   sbatch --array=1-$(wc -l < demo_slides.txt) cluster/viewer_precompute.sbatch \
#          demo_slides.txt /scratch/$USER/outputs/unet/lobo/2025-08-25HE/best.keras
#
# One slide per task rather than one job over all of them, for the same reason
# `dataset_array.sbatch` does it: a slide read straight off Lustre costs ~82x
# what the same read costs on node-local disk (205 s against 2.5 s for one
# tile), so a single long job spends its wall clock on per-operation latency
# instead of on inference.
#
# The slide list is a file, one path per line, so a task that dies can be
# resubmitted on its own without renumbering the others. Paths may contain
# spaces -- this cohort has a batch called "2025-03-28_Celina 656D H&E" -- so
# every expansion below is quoted and the list is read with `sed`, never split
# on whitespace.
set -euo pipefail
source "${MASHPATH_ROOT:-/scratch/$USER/mashpath}/cluster/fir_env.sh"
mkdir -p "/scratch/$USER/logs"

LIST="${1:?usage: sbatch --array=1-N cluster/viewer_precompute.sbatch <slide-list> <checkpoint>}"
CKPT="${2:?second argument must be the checkpoint (best.keras)}"
OUT_ROOT="${3:-$MASHPATH_OUT/viewer}"

SLIDE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
[ -n "$SLIDE" ] || { echo "no slide at index $SLURM_ARRAY_TASK_ID"; exit 1; }
[ -f "$SLIDE" ] || { echo "not a file: $SLIDE"; exit 1; }
[ -f "$CKPT" ]  || { echo "no checkpoint: $CKPT"; exit 1; }

STEM=$(basename "$SLIDE"); STEM="${STEM%.*}"
# The durable location, kept before staging overwrites $SLIDE. This is what
# goes into meta.json: $SLURM_TMPDIR is deleted when the job ends, so an
# artifact that recorded it would point at nothing the moment it was finished.
ORIGINAL="$SLIDE"
echo "slide:      $SLIDE"
echo "checkpoint: $CKPT"
echo "out:        $OUT_ROOT/$STEM"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Stage to node-local NVMe. Not optional -- see the 82x above. $SLURM_TMPDIR is
# cleaned up by the scheduler when the job ends, so there is nothing to remove.
if [ -n "${SLURM_TMPDIR:-}" ]; then
    echo "staging $(basename "$SLIDE") -> $SLURM_TMPDIR"
    cp "$SLIDE" "$SLURM_TMPDIR/" && SLIDE="$SLURM_TMPDIR/$(basename "$SLIDE")"
fi

cd "$MASHPATH_ROOT"
# Fail loudly if the GPU is not visible to TensorFlow. No GPU allocated, a GPU
# TF cannot see, and a GPU it silently falls off all look identical from the
# loop -- they all look like "it is slow", and on a 2,500-tile pass that is
# hours rather than minutes.
#
# NOT `cluster/gpu_check.py`: that script also loads StarDist, because the
# features it was written for need it. This pass never touches StarDist, so
# using it here would add a pretrained-weights download to a node with no
# outbound network -- a minutes-long stall, or a hang, in a job that had no
# reason to care.
python - <<'PY'
import sys
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs visible to TensorFlow: {len(gpus)}", flush=True)
if not gpus:
    print("NO GPU -- the pass would run on CPU without complaining")
    sys.exit(1)
import time
with tf.device("/GPU:0"):
    a = tf.random.normal([4096, 4096]); b = tf.random.normal([4096, 4096])
    tf.matmul(a, b).numpy()
    t = time.time()
    for _ in range(5):
        c = tf.matmul(a, b)
    c.numpy()
    dt = (time.time() - t) / 5
print(f"  4096^3 matmul {dt*1000:.1f} ms -> {2*4096**3/dt/1e12:.1f} TFLOP/s", flush=True)
PY

time python -m mashpath.viewer.precompute "$SLIDE" \
    --checkpoint "$CKPT" \
    --out "$OUT_ROOT/$STEM" \
    --record-path "$ORIGINAL" \
    --batch-size 16

echo "=== meta.json ==="
cat "$OUT_ROOT/$STEM/meta.json" | head -30
