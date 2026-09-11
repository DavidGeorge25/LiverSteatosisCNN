#!/bin/bash
#SBATCH --job-name=mash-serve
#SBATCH --account=def-wangd123
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
# 8 h, so one submission covers a working day and a demo cannot die because
# the meeting ran late. `cpubase_bynode_b2` takes up to 12 h; override with
# `sbatch --time=...` if you want longer. The job is 4 cores and no GPU, so
# the cost of leaving it up is close to nothing.
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/%u/logs/serve_%j.out
#SBATCH --error=/scratch/%u/logs/serve_%j.out
#
# Serve the precomputed viewer from a compute node, for the length of a demo.
#
#   sbatch cluster/viewer_serve.sbatch            # then read the log for the node
#   ssh -N -L 8000:<node>:8000 fir                # from the laptop
#   open http://localhost:8000
#
# NO GPU, deliberately. The census is already computed; this process reads
# mask PNGs off disk and region-reads the slide, and the only thing that would
# want a GPU is the live spot check -- one region, on demand, which CPU serves
# in a couple of seconds. `gpubase_interac` is a single node on this cluster,
# so queueing for a scarce H100 to serve static tiles is the kind of risk that
# only shows up when your PI is already in the room.
#
# `salloc` is the better default for day-to-day work (see docs/VIEWER.md) --
# you can restart the server in two seconds and you can see it fail. This
# script is for when the session has to survive a disconnection.
set -euo pipefail
source "${MASHPATH_ROOT:-/scratch/$USER/mashpath}/cluster/fir_env.sh"
mkdir -p "/scratch/$USER/logs"

PORT="${1:-8000}"
ROOT="${2:-$MASHPATH_OUT/viewer}"
CKPT="${3:-$MASHPATH_OUT/unet/lobo/2025-08-25HE/best.keras}"
# Bind address. Loopback is the safe default and is what `review/app.py` uses,
# but it only works if you can SSH INTO the compute node -- a one-hop
# `ssh -L $PORT:<node>:$PORT fir` has the LOGIN node open the connection, and
# the login node cannot reach another node's loopback.
#
#   loopback (default)  needs `ssh -J fir $USER@<node> -N -L $PORT:127.0.0.1:$PORT`,
#                       which in turn needs your public key in ~/.ssh/authorized_keys
#                       on the cluster. Most private; set it up once.
#   0.0.0.0             works with the plain one-hop tunnel, at the cost of
#                       exposing the port to other jobs on the cluster's
#                       internal network for the life of this job. Acceptable
#                       for a 30-minute demo on de-identified mouse material;
#                       not something to leave running for days.
HOST="${4:-127.0.0.1}"

if [ "$HOST" = "127.0.0.1" ]; then
    TUNNEL="ssh -J fir \$USER@$(hostname) -N -L $PORT:127.0.0.1:$PORT"
else
    TUNNEL="ssh -N -L $PORT:$(hostname):$PORT fir"
fi

cat <<EOF
=======================================================================
 viewer node : $(hostname)
 port        : $PORT
 bind        : $HOST

 From the laptop, in a second terminal:

     $TUNNEL

 Then open  http://localhost:$PORT
=======================================================================
EOF

# Stage the demo slides to node-local NVMe. The artifacts are small reads and
# can stay on Lustre, but the SLIDE is region-read on every pan and zoom, and
# off Lustre that costs ~82x what it costs on local disk -- the difference
# between a viewer that feels instant and one that stutters.
STAGE_ARG=()
if [ -n "${SLURM_TMPDIR:-}" ]; then
    python - "$ROOT" "$SLURM_TMPDIR" <<'PY'
import json, os, shutil, sys, glob
root, tmp = sys.argv[1], sys.argv[2]
for meta in sorted(glob.glob(os.path.join(root, "*", "meta.json"))):
    src = json.load(open(meta)).get("slide_path", "")
    if not src or not os.path.exists(src):
        print(f"  slide missing, will fall back to the recorded path: {src}",
              flush=True)
        continue
    dst = os.path.join(tmp, os.path.basename(src))
    if not os.path.exists(dst):
        print(f"  staging {os.path.basename(src)} -> {tmp}", flush=True)
        shutil.copy2(src, dst)
PY
    # The artifact is NOT rewritten -- the staged location is passed to the
    # server as a runtime override. Baking $SLURM_TMPDIR into meta.json would
    # outlive the job and break the next session on a different node.
    STAGE_ARG=(--slide-dir "$SLURM_TMPDIR")
fi

cd "$MASHPATH_ROOT"
exec python -m mashpath.viewer.server --root "$ROOT" --checkpoint "$CKPT" \
     --port "$PORT" --host "$HOST" "${STAGE_ARG[@]}"
