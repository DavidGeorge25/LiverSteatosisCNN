#!/bin/bash
#SBATCH --job-name=mash-ds
#SBATCH --account=def-wangd123
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/%u/logs/ds_%A_%a.out
#SBATCH --error=/scratch/%u/logs/ds_%A_%a.out
#
# Export one slide's pseudo-label tiles. One slide per array task, because a
# slide read straight off Lustre costs ~82x what the same read costs on local
# disk -- measured on this cluster, see ballooning_range.sbatch. The whole
# collection in one job would spend its wall clock on per-operation latency,
# not on segmentation. Staged to node-local NVMe it is ~30 s a slide.
#
# CAP THE ARRAY. Submitting every task at once made 219 jobs `cp` a ~500 MB
# slide off Lustre in the same instant last time; they all timed out having
# barely started and billed the group ~9,750 core-hours, because a TIMEOUT
# charges the full wall clock. %15 is the cap that worked.
#
#   cd $MASHPATH_ROOT
#   awk -F/ '!seen[$NF]++' cluster/.dripped \
#     | grep -v '2025-03-28_Celina' > /scratch/$USER/train_slides.txt
#
# `awk -F/ '!seen[$NF]++'` dedupes on the FILENAME, which is what matters:
# R26-122-1_HE_70.svs is listed both at the top of slides/ and inside its batch
# folder, and exporting it twice would write the same tile ids twice and weight
# that animal double. (`sort -u -t/ -k6` looks like it would do this and does
# not -- the top-level path has one fewer field, so the two lines never
# compare equal. It returns 261.) Expect 252 lines: 260 slides minus the 8
# reserved.
#   sbatch --array=1-$(wc -l < /scratch/$USER/train_slides.txt)%15 \
#          cluster/dataset_array.sbatch /scratch/$USER/train_slides.txt
#   sbatch --dependency=singleton cluster/dataset_finalize.sbatch
#
# The reserved batch is filtered out of the list AND refused by export_one, so
# a slide of it reaching this script fails the task rather than the project.
set -euo pipefail
source "${MASHPATH_ROOT:-/scratch/$USER/mashpath}/cluster/fir_env.sh"
mkdir -p "/scratch/$USER/logs"
LIST="${1:?usage: sbatch --array=1-N%15 cluster/dataset_array.sbatch <slide-list>}"
TILES="${2:-150}"
SLIDE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
[ -n "$SLIDE" ] || { echo "no slide at index $SLURM_ARRAY_TASK_ID"; exit 1; }
echo "slide: $SLIDE"

if [ -n "${SLURM_TMPDIR:-}" ]; then
    t0=$SECONDS
    cp "$SLIDE" "$SLURM_TMPDIR/" && SLIDE="$SLURM_TMPDIR/$(basename "$SLIDE")"
    echo "  staged in $((SECONDS-t0))s"   # minutes here means the cap is too high
fi

cd "$MASHPATH_ROOT"
time python -c "
import sys
from mashpath.config import MashConfig as Config
from mashpath.features.steatosis.dataset import export_one
cfg = Config.from_yaml(['configs/tuned.yaml'])
df = export_one(sys.argv[1], cfg, sys.argv[2], max_tiles_per_slide=int(sys.argv[3]), seed=0)
print(f'{len(df)} tiles, mean fat {df.fat_fraction.mean()*100:.2f}%')
" "$SLIDE" "$MASHPATH_OUT/dataset_v2" "$TILES"
