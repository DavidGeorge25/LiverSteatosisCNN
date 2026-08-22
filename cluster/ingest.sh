#!/bin/bash
# Unattended: wait for a slide zip, unpack it, verify it, ship it to Fir,
# and launch all three pipelines. Written to be started and left alone.
#
#   bash cluster/ingest.sh [/path/to/watch]
#
# Every step logs to cluster/ingest.log and every step is idempotent, because
# the failure that actually happens on a 25 GB transfer is "the laptop slept
# for an hour" -- rerunning must resume, not restart.
set -uo pipefail

WATCH="${1:-$(pwd)}"
LOG="$(cd "$(dirname "$0")/.." && pwd)/cluster/ingest.log"
STAGE="$(cd "$(dirname "$0")/.." && pwd)/incoming"
REMOTE_SLIDES="/scratch/davidg25/slides"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== ingest started; watching $WATCH for *.zip ==="

# ---- 1. wait for a zip that has stopped growing -------------------------
# Size-stable for two consecutive checks, otherwise we unzip a partial file
# and get a confusing corruption error instead of "still downloading".
ZIP=""
while :; do
    CAND=$(find "$WATCH" -maxdepth 1 -name '*.zip' -size +100M 2>/dev/null | head -1)
    if [ -n "$CAND" ]; then
        S1=$(stat -f %z "$CAND" 2>/dev/null || echo 0)
        sleep 30
        S2=$(stat -f %z "$CAND" 2>/dev/null || echo 0)
        if [ "$S1" = "$S2" ] && [ "$S1" != "0" ]; then
            ZIP="$CAND"
            log "found $(basename "$ZIP") ($(echo "$S1" | awk '{printf "%.1f GB", $1/1e9}')), size stable"
            break
        fi
        log "waiting: $(basename "$CAND") still growing ($(echo "$S2" | awk '{printf "%.1f GB", $1/1e9}'))"
    fi
    sleep 30
done

# ---- 2. unpack ----------------------------------------------------------
mkdir -p "$STAGE"
if [ -z "$(find "$STAGE" -name '*.svs' 2>/dev/null | head -1)" ]; then
    log "unzipping -> $STAGE"
    # Python, not unzip(1). macOS ships Info-ZIP 6.00 which cannot read ZIP64,
    # and a >4 GB archive from OneDrive is always ZIP64 -- unzip reports
    # "start of central directory not found; zipfile corrupt" on a perfectly
    # good file, which reads exactly like a failed download.
    .venv/bin/python -c "
import zipfile, sys
z = zipfile.ZipFile(sys.argv[1])
members = [n for n in z.namelist() if n.lower().endswith(('.svs','.tif','.ndpi'))]
print(f'extracting {len(members)} slide file(s)')
z.extractall(sys.argv[2], members=members)
" "$ZIP" "$STAGE" 2>&1 | tail -5 | tee -a "$LOG"
else
    log "slides already unpacked in $STAGE, skipping unzip"
fi

N=$(find "$STAGE" -name '*.svs' -o -name '*.tif' -o -name '*.ndpi' | wc -l | tr -d ' ')
log "unpacked: $N slide file(s), $(du -sh "$STAGE" | cut -f1)"
[ "$N" -gt 0 ] || { log "FATAL: no slides found in the zip"; exit 1; }

# ---- 3. verify locally before spending bandwidth ------------------------
# A slide that will not open, or carries no scale, fails the same way after a
# 7-hour upload as it does now. Checking first is free.
log "verifying slides locally..."
.venv/bin/python cluster/verify_slides.py "$STAGE" 2>&1 | tee -a "$LOG" | tail -25
VERIFY=${PIPESTATUS[0]}
[ "$VERIFY" -eq 0 ] || log "WARNING: some slides failed verification (see above) -- continuing with the rest"

# ---- 4. ship to Fir -----------------------------------------------------
if ! ssh -O check fir >/dev/null 2>&1; then
    log "FATAL: no SSH master socket to Fir. Run 'ssh fir' (answer Duo), then rerun."
    exit 2
fi

log "transferring to Fir (resumable; safe to rerun)"
rsync -a --partial --append-verify --info=progress2 \
      --include='*/' --include='*.svs' --include='*.tif' --include='*.ndpi' --exclude='*' \
      -e ssh "$STAGE/" "fir:$REMOTE_SLIDES/" 2>&1 | tail -3 | tee -a "$LOG"
RC=$?
[ "$RC" -eq 0 ] || { log "FATAL: rsync exited $RC -- rerun to resume"; exit 3; }
log "transfer complete"

# ---- 5. verify on the far side -----------------------------------------
log "verifying on Fir..."
ssh fir "bash -lc 'cd /scratch/\$USER/mashpath && source cluster/fir_env.sh && \
    bash cluster/slides.sh && python cluster/verify_slides.py'" 2>&1 | tail -20 | tee -a "$LOG"

# ---- 6. launch the pipelines -------------------------------------------
# Steatosis is CPU and cheap, so it goes wide. The GPU features go narrower to
# avoid monopolising the allocation. Ballooning uses the chunked path: one
# stuck tile then costs one chunk instead of the whole slide.
log "submitting jobs"
ssh fir "bash -lc 'cd /scratch/\$USER/mashpath && source cluster/fir_env.sh && \
    N=\$(wc -l < \$MASHPATH_SLIDES/slide_list.txt) && \
    echo \"slides: \$N\" && \
    sbatch --array=1-\$N%20 cluster/steatosis.sbatch && \
    sbatch --array=1-\$N%6 cluster/inflammation.sbatch'" 2>&1 | tail -5 | tee -a "$LOG"

log "=== ingest done. Ballooning is per-slide chunked -- launch with:"
log "    bash cluster/submit_ballooning.sh   (after the above finish)"
