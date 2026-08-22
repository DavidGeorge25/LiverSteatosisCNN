#!/bin/bash
#SBATCH --job-name=mash-inflammation
#SBATCH --account=def-wangd123
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/%u/logs/inflammation_%A_%a.out
#SBATCH --error=/scratch/%u/logs/inflammation_%A_%a.out
#
# Inflammation: GPU, and the most expensive of the three -- nuclei are
# segmented at `upscale: 2`, which is 4x the pixels of ballooning's pass.
# Hence the longer wall clock than ballooning on the same tile count.
#
# NOTE: this runs stages 1-2 only (segment + classify). Candidate generation
# is not implemented; see mashpath/features/inflammation/detect.py.
#
#   sbatch --array=1-N%8 cluster/inflammation.sbatch
set -euo pipefail
source "${MASHPATH_ROOT:-/scratch/$USER/mashpath}/cluster/fir_env.sh"
mkdir -p "/scratch/$USER/logs"

SLIDE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MASHPATH_SLIDES/slide_list.txt")
[ -n "$SLIDE" ] || { echo "no slide at index $SLURM_ARRAY_TASK_ID"; exit 1; }
echo "task $SLURM_ARRAY_TASK_ID -> $SLIDE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Stage the slide on NODE-LOCAL disk before reading it.
#
# The slides live on Lustre (/scratch). openslide services each region read by
# pulling many small compressed tiles, and Lustre's per-operation latency turns
# that into the dominant cost: measured on Fir, one tile of R26-122-1_HE_70
# took 205 s against 2.5 s for the identical tile on a laptop SSD -- 82x, with
# no difference in cell count. Whole 300-tile chunks then blow their wall clock
# and look like a hang.
#
# $SLURM_TMPDIR is node-local NVMe and is cleaned up automatically when the job
# ends. A 1 GB slide copies in seconds and every read after that is local.
if [ -n "${SLURM_TMPDIR:-}" ]; then
    echo "staging $(basename "$SLIDE") -> $SLURM_TMPDIR"
    cp "$SLIDE" "$SLURM_TMPDIR/" && SLIDE="$SLURM_TMPDIR/$(basename "$SLIDE")"
    echo "  staged: $(du -h "$SLIDE" | cut -f1)"
fi

cd "$MASHPATH_ROOT"
time python -m mashpath.features.inflammation.cli nuclei \
    --slide "$SLIDE" \
    --config configs/core.yaml --config configs/inflammation.yaml --config configs/production.yaml \
    --output-dir "$MASHPATH_OUT/inflammation"
