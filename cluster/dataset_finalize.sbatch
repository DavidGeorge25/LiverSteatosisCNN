#!/bin/bash
#SBATCH --job-name=mash-ds
#SBATCH --account=def-wangd123
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --time=0:20:00
#SBATCH --output=/scratch/%u/logs/dsfin_%j.out
#SBATCH --error=/scratch/%u/logs/dsfin_%j.out
#
# Concatenate the array's manifest fragments, derive cohort, write manifest.csv
# and batches.txt. Opens no slide, so it is cheap and safe to rerun after a
# failed task is resubmitted -- it does not re-export the slides that worked.
#
# Shares --job-name with dataset_array.sbatch so `--dependency=singleton` makes
# it wait for the whole array without anyone pasting a job id.
#
# It also exports the RESERVED batch, into its own directory, with reserved=()
# typed. Eight slides straight off Lustre is a few minutes and does not justify
# an array; and keeping it here means the training set and the held-out set are
# never produced by the same loop.
set -euo pipefail
source "${MASHPATH_ROOT:-/scratch/$USER/mashpath}/cluster/fir_env.sh"
mkdir -p "/scratch/$USER/logs"
TILES="${1:-150}"
cd "$MASHPATH_ROOT"
python - "$MASHPATH_OUT/dataset_v2" "$MASHPATH_SLIDES" "$TILES" <<'PY'
import glob, sys
from pathlib import Path
from mashpath.config import MashConfig as Config
from mashpath.features.steatosis.dataset import (build_flat_dataset,
                                                 finalize_flat_dataset,
                                                 resolve_batches)
from mashpath.train.splits import RESERVED_BATCHES

out, slides_dir, tiles = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = Config.from_yaml(["configs/tuned.yaml"])
df = finalize_flat_dataset(out, cfg)
print("batches.txt:\n  " + "\n  ".join(sorted(df.batch.unique())), flush=True)

paths = sorted(set(glob.glob(f"{slides_dir}/*/*.svs")) | set(glob.glob(f"{slides_dir}/*.svs")))
batch_of, _ = resolve_batches(paths)
held = [p for p in paths if batch_of[Path(p).stem] in RESERVED_BATCHES]
if not held:
    raise SystemExit("no slides matched RESERVED_BATCHES -- the external test "
                     "would have nothing to run on; fix before training")
r = build_flat_dataset(held, cfg, out + "_reserved", max_tiles_per_slide=tiles,
                       seed=0, workers=2, reserved=())
print(f"reserved: {len(r)} tiles from {r.slide.nunique()} slides", flush=True)
PY
