#!/bin/bash
#SBATCH --job-name=mash-unet
#SBATCH --account=def-wangd123
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/%u/logs/unet_%A_%a.out
#SBATCH --error=/scratch/%u/logs/unet_%A_%a.out
#
# Leave-one-batch-out, one fold per array task. The list of held-out batches is
# a file, one batch name per line, so a fold that dies can be resubmitted on its
# own without renumbering the others:
#
#   python -c "import pandas as pd; \
#     print('\n'.join(sorted(pd.read_csv('$DS/manifest.csv').batch.unique())))" > batches.txt
#   sbatch --array=1-$(wc -l < batches.txt) cluster/unet.sbatch batches.txt
#
# 8 cpus, not more: the input pipeline decodes three PNGs per tile and runs the
# colour augmentation through tf.numpy_function, so it is CPU-bound before it is
# GPU-bound -- but past ~8 workers the GIL in that numpy_function stops it
# scaling and the extra cores are billed for nothing.
#
# 6 h covers 15 epochs over ~33,000 tiles at 256 px with early stopping at
# patience 4; most folds should stop well before that. If a fold hits the wall
# clock, the checkpoint at best.keras is still there -- but the summary.json is
# not, so treat a missing summary as a failed fold rather than a slow one.
set -euo pipefail
source "${MASHPATH_ROOT:-/scratch/$USER/mashpath}/cluster/fir_env.sh"
mkdir -p "/scratch/$USER/logs"
LIST="${1:?usage: sbatch --array=1-N cluster/unet.sbatch <batch-list-file>}"
HELD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
[ -n "$HELD" ] || { echo "no batch at index $SLURM_ARRAY_TASK_ID"; exit 1; }

DS="$MASHPATH_OUT/dataset_v2"
OUT="$MASHPATH_OUT/unet/lobo/$HELD"
echo "held-out batch: $HELD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$MASHPATH_ROOT"
# Fail loudly if the GPU is not actually visible to TensorFlow. Three different
# faults -- no GPU allocated, a GPU TF cannot see, a GPU it silently falls off
# -- all look identical from the training loop, and all look like "it is slow".
python cluster/gpu_check.py

time python -m mashpath.train.unet \
    --manifest "$DS/manifest.csv" --root "$DS" \
    --held-out "$HELD" --out "$OUT" \
    --reserved-root "${DS}_reserved"
