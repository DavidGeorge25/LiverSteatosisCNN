#!/bin/bash
#SBATCH --job-name=mash-dataset
#SBATCH --account=def-wangd123
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000M
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/%u/logs/dataset_%j.out
#SBATCH --error=/scratch/%u/logs/dataset_%j.out
#
# Build the flat pseudo-label dataset from every slide on /scratch, so the
# training jobs have something to read without a 22 GB transfer from a laptop.
#
# NO GPU. This is classical CV -- threshold, morphology, connected components,
# nine times per tile -- and asking for an H100 to run it would charge the
# group for an idle accelerator. Measured at 30 s per slide single-threaded on
# an M4, so 16 cpus and 2 h is roughly 8x the expected 15 min.
#
# TWO CALLS, and the second one matters. The reserved batch (R22-354, the only
# slides with known diet) is refused by build_flat_dataset, so it is exported
# separately with reserved=() into its own directory. That is deliberately
# awkward: evaluating on it takes a path someone has to name, and no glob over
# the training set can reach it.
#
#   sbatch cluster/dataset.sbatch [tiles-per-slide]
set -euo pipefail
source "${MASHPATH_ROOT:-/scratch/$USER/mashpath}/cluster/fir_env.sh"
mkdir -p "/scratch/$USER/logs"
TILES="${1:-150}"
DS="$MASHPATH_OUT/dataset_v2"

cd "$MASHPATH_ROOT"
python - "$MASHPATH_SLIDES" "$DS" "$TILES" <<'PY'
import glob, sys, time
from mashpath.config import MashConfig as Config
from mashpath.features.steatosis.dataset import build_flat_dataset

slides_dir, out, tiles = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = Config.from_yaml(["configs/tuned.yaml"])
# Both layouts: slides in per-batch folders, and any sitting at the top level.
paths = sorted(glob.glob(f"{slides_dir}/*/*.svs") + glob.glob(f"{slides_dir}/*.svs"))
print(f"{len(paths)} slides under {slides_dir}", flush=True)

t = time.time()
df = build_flat_dataset(paths, cfg, out, max_tiles_per_slide=tiles, seed=0,
                        workers=int(__import__("os").environ.get("SLURM_CPUS_PER_TASK", 8)))
print(f"training set: {len(df)} tiles in {(time.time()-t)/60:.1f} min", flush=True)

# The reserved batch, into its own directory. reserved=() is the only way to
# export it and has to be typed, which is the point.
from mashpath.train.splits import RESERVED_BATCHES
from mashpath.features.steatosis.dataset import resolve_batches
batch_of, _ = resolve_batches(paths)
held = [p for p in paths if batch_of[__import__("pathlib").Path(p).stem] in RESERVED_BATCHES]
if held:
    r = build_flat_dataset(held, cfg, out + "_reserved", max_tiles_per_slide=tiles,
                           seed=0, workers=4, reserved=())
    print(f"reserved held-out set: {len(r)} tiles from {len(held)} slides", flush=True)
else:
    print("WARNING: no slides matched RESERVED_BATCHES -- the external test "
          "will have nothing to run on", flush=True)
PY
