#!/bin/bash
#SBATCH --job-name=mash-bal-range
#SBATCH --account=def-wangd123
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/%u/logs/balrange_%j.out
#SBATCH --error=/scratch/%u/logs/balrange_%j.out
#
# Pass 1 over an EXPLICIT tile range, for re-running or bisecting one chunk.
# `ballooning_chunked.sbatch` derives its range from the array index, which is
# right for a whole slide and useless when you need [675:676).
#
#   sbatch --export=ALL,SLIDE=$SL,START=600,STOP=675 cluster/ballooning_range.sbatch
set -euo pipefail
source "${MASHPATH_ROOT:-/scratch/$USER/mashpath}/cluster/fir_env.sh"
mkdir -p "/scratch/$USER/logs"
: "${SLIDE:?set SLIDE=...}"; : "${START:?set START=...}"; : "${STOP:?set STOP=...}"

echo "slide=$SLIDE  tiles [$START:$STOP)"
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
time python -m mashpath.features.ballooning.cli chunk \
    --slide "$SLIDE" \
    --config configs/core.yaml --config configs/ballooning.yaml --config configs/production.yaml \
    --start "$START" --stop "$STOP" \
    --output-dir "$MASHPATH_OUT/ballooning"
