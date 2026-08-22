#!/bin/bash
#SBATCH --job-name=mash-steat1
#SBATCH --account=def-wangd123
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=1:30:00
#SBATCH --output=/scratch/%u/logs/steat1_%j.out
#SBATCH --error=/scratch/%u/logs/steat1_%j.out
#
# Steatosis for ONE named slide, for the drip-feed. The array version indexes
# into slide_list.txt, which is wrong while that list is still growing -- a
# task submitted now would resolve its index against a different list later.
set -euo pipefail
source "${MASHPATH_ROOT:-/scratch/$USER/mashpath}/cluster/fir_env.sh"
mkdir -p "/scratch/$USER/logs"
: "${SLIDE:?set SLIDE=...}"
echo "slide: $SLIDE"

# Node-local staging: see the note in ballooning_chunked.sbatch. Reading a WSI
# repeatedly over Lustre is what made one tile take 205 s instead of 2.5 s.
if [ -n "${SLURM_TMPDIR:-}" ]; then
    cp "$SLIDE" "$SLURM_TMPDIR/" && SLIDE="$SLURM_TMPDIR/$(basename "$SLIDE")"
    echo "  staged to node-local disk"
fi

cd "$MASHPATH_ROOT"
time python -m mashpath.cli steatosis \
    --slide "$SLIDE" \
    --config configs/core.yaml --config configs/steatosis.yaml --config configs/production.yaml \
    --output-dir "$MASHPATH_OUT/steatosis"
