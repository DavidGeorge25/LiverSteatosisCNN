#!/bin/bash
#SBATCH --job-name=mash-site
#SBATCH --account=def-wangd123
#SBATCH --cpus-per-task=8
#SBATCH --mem=24000M
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/%u/logs/site_%j.out
#SBATCH --error=/scratch/%u/logs/site_%j.out
#
# Render precomputed slides into a static site that any host will serve.
#
#   sbatch cluster/viewer_export_site.sbatch /scratch/$USER/site 4 68 \
#          "$MASHPATH_OUT/viewer/R22-354_3_WT-CHOW3" ...
#
# No GPU: the census already ran. This is JPEG encoding and region reads, and
# it is the region reads that decide the wall clock -- so the slides are staged
# to node-local NVMe first, exactly as the census does, for the same 82x.
set -euo pipefail
source "${MASHPATH_ROOT:-/scratch/$USER/mashpath}/cluster/fir_env.sh"
mkdir -p "/scratch/$USER/logs"

OUT="${1:?usage: viewer_export_site.sbatch <out-dir> <min-downsample> <quality> <artifact>...}"
MIND="${2:-4}"
QUAL="${3:-68}"
shift 3
[ "$#" -gt 0 ] || { echo "no artifacts given"; exit 1; }

echo "out:            $OUT"
echo "min-downsample: $MIND   (deepest zoom, as a level-0 downsample)"
echo "jpeg quality:   $QUAL"
echo "artifacts:      $#"

# Stage each artifact's slide to node-local disk and point a COPY of the
# artifact at it. The copy lives in $SLURM_TMPDIR so the real artifact on
# /scratch keeps its durable slide_path -- rewriting that in place would bake
# a dead node path into a file that outlives the job.
STAGED=()
if [ -n "${SLURM_TMPDIR:-}" ]; then
    for a in "$@"; do
        stem=$(basename "$a")
        work="$SLURM_TMPDIR/art/$stem"
        mkdir -p "$work"
        cp -r "$a/." "$work/"
        python - "$work" "$SLURM_TMPDIR" <<'PY'
import json, os, shutil, sys
work, tmp = sys.argv[1], sys.argv[2]
mp = os.path.join(work, "meta.json")
m = json.load(open(mp))
src = m["slide_path"]
dst = os.path.join(tmp, os.path.basename(src))
if not os.path.exists(dst):
    print(f"  staging {os.path.basename(src)}", flush=True)
    shutil.copy2(src, dst)
m["slide_path"] = dst
json.dump(m, open(mp, "w"), indent=2)
PY
        STAGED+=("$work")
    done
else
    STAGED=("$@")
fi

cd "$MASHPATH_ROOT"
time python -m mashpath.viewer.export_static "${STAGED[@]}" \
    --out "$OUT" --min-downsample "$MIND" --quality "$QUAL"

echo
echo "=== site size ==="
du -sh "$OUT"
du -sh "$OUT"/*/ 2>/dev/null
find "$OUT" -type f | wc -l | xargs echo "files:"
