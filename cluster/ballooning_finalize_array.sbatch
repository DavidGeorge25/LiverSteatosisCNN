#!/bin/bash
#SBATCH --job-name=mash-bal-fin
#SBATCH --account=def-wangd123
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000M
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/%u/logs/balfin_%A_%a.out
#SBATCH --error=/scratch/%u/logs/balfin_%A_%a.out
#
# Pass 2 for many slides. NO GPU: StarDist is done by this point; what remains
# is a cKDTree neighbourhood query, the ranking, and writing review crops.
# Measured at 3.8 s for a 40-tile slide, so 30 min is generous for 300 tiles.
#
# Kept small (2 cpus) on purpose -- the group's CPU fairshare is currently
# depressed, and a cheap job schedules sooner than a fat one.
set -euo pipefail
source "${MASHPATH_ROOT:-/scratch/$USER/mashpath}/cluster/fir_env.sh"
mkdir -p "/scratch/$USER/logs"
LIST="${1:?usage: sbatch ... <slide-list>}"
SLIDE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
[ -n "$SLIDE" ] || exit 1
echo "finalize: $SLIDE"
cd "$MASHPATH_ROOT"
python -m mashpath.features.ballooning.cli finalize \
    --slide "$SLIDE" \
    --config configs/core.yaml --config configs/ballooning.yaml \
    --config configs/production.yaml \
    --output-dir "$MASHPATH_OUT/ballooning"
