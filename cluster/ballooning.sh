#!/bin/bash
#SBATCH --job-name=mash-ballooning
#SBATCH --array=0-52%8
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/ballooning_%A_%a.out
#
# Ballooning candidate generation: one slide per array task, one H100 each.
#
# SIZING. Measured on a laptop at upscale 2: 2.04 s/tile, of which ~41% is
# StarDist (GPU) and ~59% is watershed + GLCM texture + regionprops (CPU, and
# staying there). So the H100 shortens 41% of the work and nothing else --
# Amdahl caps the speedup near 2x per task no matter how fast the GPU is.
#
# That is why --cpus-per-task=8 matters more than it looks: the CPU remainder
# is the critical path. A median 1,193-tile slide is ~40 min at laptop rates;
# 3 h covers the 4,554-tile outlier with headroom for a cold page cache.
#
# %8 throttles concurrent GPU tasks. Raise it if your allocation allows -- the
# tasks are independent and the array is embarrassingly parallel.
set -euo pipefail
module --force purge && module load StdEnv/2023 gcc opencv/4.11.0 python/3.11 openslide/4.0.0 cuda/12.2
source "${VENV:-$HOME/venvs/mashpath}/bin/activate"

# Keep TF off the GPU's whole VRAM: one model, one tile at a time.
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3

SLIDES=("$SLIDE_DIR"/*.svs)
SLIDE="${SLIDES[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID -> $SLIDE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m mashpath.cli ballooning \
    --slide "$SLIDE" \
    --config configs/core.yaml --config configs/ballooning.yaml \
    --config cluster/offline_model.yaml \
    --output-dir "$OUT_DIR"
