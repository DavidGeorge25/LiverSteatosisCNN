#!/bin/bash
#SBATCH --job-name=mash-steat
#SBATCH --account=def-wangd123
#SBATCH --cpus-per-task=2
#SBATCH --mem=12000M
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/%u/logs/steatA_%A_%a.out
#SBATCH --error=/scratch/%u/logs/steatA_%A_%a.out
#
# Array over a FIXED slide list, with a concurrency cap (%N on submit).
#
# The cap is the point. Submitting 219 independent jobs at once made every one
# of them `cp` a ~500 MB slide off Lustre in the same instant -- ~110 GB of
# concurrent reads. The jobs then timed out at 4 h having barely started, while
# the identical slide profiles at 0.8 min/slide on an idle node. Nothing was
# slow except the stampede they created for each other.
#
# That mistake cost ~9,750 core-hours of the group's fairshare, because a
# TIMEOUT bills the full wall clock. Hence the deliberately small request
# below: steatosis measures 0.8 min/slide and is largely single-threaded, so
# 8 CPUs and a 4 h limit bought nothing and charged 8x. 2 CPUs / 30 min is
# ~13x cheaper per slide, schedules sooner under a damaged fairshare, and is
# still 20x the measured runtime.
#
#   sbatch --array=1-$(wc -l < LIST)%15 cluster/steatosis_array.sbatch LIST
set -euo pipefail
source "${MASHPATH_ROOT:-/scratch/$USER/mashpath}/cluster/fir_env.sh"
mkdir -p "/scratch/$USER/logs"
LIST="${1:?usage: sbatch ... steatosis_array.sbatch <slide-list-file>}"

SLIDE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
[ -n "$SLIDE" ] || { echo "no slide at index $SLURM_ARRAY_TASK_ID"; exit 1; }
echo "slide: $SLIDE"

if [ -n "${SLURM_TMPDIR:-}" ]; then
    t0=$SECONDS
    cp "$SLIDE" "$SLURM_TMPDIR/" && SLIDE="$SLURM_TMPDIR/$(basename "$SLIDE")"
    echo "  staged in $((SECONDS-t0))s"   # if this is minutes, the cap is too high
fi

cd "$MASHPATH_ROOT"
time python -m mashpath.cli steatosis \
    --slide "$SLIDE" \
    --config configs/core.yaml --config configs/steatosis.yaml \
    --config configs/production.yaml \
    --output-dir "$MASHPATH_OUT/steatosis"
