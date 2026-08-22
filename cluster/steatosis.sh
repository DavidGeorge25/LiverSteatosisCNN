#!/bin/bash
#SBATCH --job-name=mash-steatosis
#SBATCH --array=0-52%13
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/steatosis_%A_%a.out
#
# Steatosis over the cohort: one slide per array task.
#
# NO GPU. Steatosis is classical CV -- threshold, morphology, regionprops --
# and there is no model in it. Asking for a GPU here would queue behind the
# GPU partition for hours to run a job that cannot use one. At ~0.033 s/tile
# and a median 1,193 tiles, a slide is under a minute; the 45 min ceiling is
# for the 4,554-tile outlier plus slide-read overhead.
set -euo pipefail
module --force purge && module load StdEnv/2023 gcc opencv/4.11.0 python/3.11 openslide/4.0.0
source "${VENV:-$HOME/venvs/mashpath}/bin/activate"

SLIDES=("$SLIDE_DIR"/*.svs)
SLIDE="${SLIDES[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID -> $SLIDE"

python -m mashpath.cli steatosis \
    --slide "$SLIDE" \
    --config configs/core.yaml --config configs/steatosis.yaml \
    --output-dir "$OUT_DIR"
