#!/bin/bash
#SBATCH --job-name=mash-bal-chunk
#SBATCH --account=def-wangd123
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/%u/logs/balchunk_%A_%a.out
#SBATCH --error=/scratch/%u/logs/balchunk_%A_%a.out
#
# Pass 1 for ONE slide, split across array tasks by tile range.
#
# Sizing note, because this looked like a hang for a long time and was not.
# Per-tile cost on Fir is NOT uniform: most of a slide runs ~4.3 s/tile, but
# some regions run ~17-21 s/tile (measured: tiles 676-685 of R26-122-1_HE_70,
# 173 s for 10 tiles, unchanged by giving the job 4x the memory). The same
# tiles take 2.9 s on a laptop, so it is specific to this hardware and not to
# the data.
#
# A 300-tile chunk therefore ranges from 22 min to 105 min depending on where
# it lands, and every early failure here was a job hitting a wall clock set
# for the optimistic case. Progress prints every 25 tiles -- 7+ minutes apart
# in a slow region -- which is why a merely-slow job read as frozen.
#
# 150 tiles at the measured worst case is ~53 min, well inside 2 h.
#
#   source cluster/fir_env.sh
#   SLIDE=/scratch/$USER/slides/X.svs
#   N=$(python -m mashpath.features.ballooning.cli count --slide $SLIDE \
#         --config configs/core.yaml --config configs/ballooning.yaml --config configs/production.yaml)
#   CHUNKS=$(( (N + 299) / 300 ))
#   sbatch --array=0-$((CHUNKS-1)) --export=ALL,SLIDE=$SLIDE,CHUNK=300 \
#          cluster/ballooning_chunked.sbatch
#   # then, once every task has finished:
#   sbatch --export=ALL,SLIDE=$SLIDE cluster/ballooning_finalize.sbatch
set -euo pipefail
source "${MASHPATH_ROOT:-/scratch/$USER/mashpath}/cluster/fir_env.sh"
mkdir -p "/scratch/$USER/logs"

: "${SLIDE:?set SLIDE=/path/to/slide.svs via --export}"
CHUNK="${CHUNK:-150}"
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
STOP=$(( START + CHUNK ))

echo "slide=$SLIDE  tiles [$START:$STOP)"
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
time python -m mashpath.features.ballooning.cli chunk \
    --slide "$SLIDE" \
    --config configs/core.yaml --config configs/ballooning.yaml --config configs/production.yaml \
    --start "$START" --stop "$STOP" \
    --output-dir "$MASHPATH_OUT/ballooning"
