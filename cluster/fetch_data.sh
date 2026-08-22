#!/bin/bash
# Pull the slide set from OneDrive onto Fir. Run on a LOGIN node.
#
#   1. On your laptop (needs a browser, so it cannot be automated):
#          rclone authorize "onedrive"
#      Copy the JSON blob it prints.
#
#   2. On Fir, once:
#          rclone config
#      n) new remote -> name: onedrive -> storage: onedrive
#      -> "Use auto config?" answer **n** (no browser on a login node)
#      -> paste the JSON from step 1
#      -> choose the McMaster drive when it lists them
#
#   3. bash cluster/fetch_data.sh [remote_path]
#
# Runs on a login node deliberately: compute nodes have no outbound network,
# so this cannot be a batch job. It is I/O, not compute, so it does not abuse
# the login node -- but keep it in tmux, because 25 GB outlives an ssh session.
#
# --checksum not --size-only: OneDrive and Lustre disagree about mtime, so a
# size-only comparison silently accepts a truncated file. A truncated SVS opens
# fine and reads garbage in the missing region, which would surface as
# mysterious tissue-detection failures on a handful of slides.
set -euo pipefail

REMOTE="${1:-onedrive:Training}"
DEST="${MASHPATH_SLIDES:-/scratch/$USER/slides}"
mkdir -p "$DEST"

echo "=== source listing ==="
rclone lsd "$REMOTE" 2>/dev/null || true
rclone size "$REMOTE"

echo
echo "=== copying $REMOTE -> $DEST ==="
rclone copy "$REMOTE" "$DEST" \
    --transfers 8 \
    --checkers 16 \
    --checksum \
    --progress \
    --stats 30s \
    --log-file "/scratch/$USER/logs/rclone_$(date +%Y%m%d_%H%M%S).log" \
    --log-level INFO

echo
echo "=== verifying ==="
rclone check "$REMOTE" "$DEST" --checksum --one-way 2>&1 | tail -5

echo
du -sh "$DEST"
find "$DEST" -name '*.svs' | wc -l | xargs echo "SVS files:"
