#!/bin/bash
#SBATCH --job-name=mash-inflammation
#SBATCH --array=0-52%8
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/inflammation_%A_%a.out
#
# Lobular inflammation, stages 1-2 only (nucleus segmentation + the
# hepatocyte/immune split). Stages 3-5 -- portal exclusion, focus clustering,
# candidate export -- are NOT implemented; this job produces nuclei.csv and QC,
# not a review package. See mashpath/features/inflammation/detect.py.
#
# ~0.78 s/tile at upscale 2, same 41/59 GPU/CPU split as ballooning but
# without the watershed and texture passes, so it is ~2.6x cheaper per tile.
set -euo pipefail
module --force purge && module load StdEnv/2023 gcc opencv/4.11.0 python/3.11 openslide/4.0.0 cuda/12.2
source "${VENV:-$HOME/venvs/mashpath}/bin/activate"

export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3

SLIDES=("$SLIDE_DIR"/*.svs)
SLIDE="${SLIDES[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID -> $SLIDE"

python -m mashpath.features.inflammation.cli nuclei \
    --slide "$SLIDE" \
    --config configs/core.yaml --config configs/inflammation.yaml \
    --config cluster/offline_model.yaml \
    --output-dir "$OUT_DIR"
