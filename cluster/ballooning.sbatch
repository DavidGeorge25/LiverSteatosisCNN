#!/bin/bash
#SBATCH --job-name=mash-ballooning
#SBATCH --account=def-wangd123
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/%u/logs/ballooning_%A_%a.out
#SBATCH --error=/scratch/%u/logs/ballooning_%A_%a.out
#
# Ballooning: GPU. StarDist inference over every tissue tile is the cost.
#
# One GPU, not four. StarDist runs tile-by-tile and the pipeline is a serial
# stream over tiles -- a second GPU would sit idle unless the tile loop were
# rewritten to batch across devices. More useful to run more slides at once.
#
#   sbatch --array=1-N%8 cluster/ballooning.sbatch
set -euo pipefail
source "${MASHPATH_ROOT:-/scratch/$USER/mashpath}/cluster/fir_env.sh"
mkdir -p "/scratch/$USER/logs"

SLIDE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MASHPATH_SLIDES/slide_list.txt")
[ -n "$SLIDE" ] || { echo "no slide at index $SLURM_ARRAY_TASK_ID"; exit 1; }
echo "task $SLURM_ARRAY_TASK_ID -> $SLIDE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$MASHPATH_ROOT"
time python -m mashpath.features.ballooning.cli run \
    --slide "$SLIDE" \
    --config configs/core.yaml --config configs/ballooning.yaml \
    --output-dir "$MASHPATH_OUT/ballooning"
