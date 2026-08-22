#!/bin/bash
# Submit jobs for slides as they finish transferring, instead of waiting for
# the whole upload.
#
#   nohup bash cluster/drip_feed.sh > cluster/drip.log 2>&1 &
#
# The upload runs at ~0.7 MB/s (a home uplink, measured -- two parallel streams
# gained 14%, so it is saturated, not stream-limited). 134 GB is therefore
# ~54 hours. Waiting for all of it before starting any compute wastes two days
# of cluster time for no reason: each slide is independent, and steatosis on
# one slide costs 4 minutes.
#
# Only steatosis is dripped. It is CPU-only, cheap, and has no cross-slide
# dependency. Ballooning and inflammation are GPU and want a considered batch
# submission once the cohort is known -- dripping them would scatter 260
# single-slide GPU jobs across the queue and annoy everyone.
#
# Safe to run twice: submitted slides are recorded and skipped.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEEN="$ROOT/cluster/.dripped"
LOG="$ROOT/cluster/drip.log"
INTERVAL="${INTERVAL:-600}"
MAX_QUEUED="${MAX_QUEUED:-25}"

touch "$SEEN"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "drip-feed started (every ${INTERVAL}s, max ${MAX_QUEUED} queued)"

while :; do
    if ! ssh -O check fir >/dev/null 2>&1; then
        log "no SSH master socket -- pausing (run 'ssh fir' to restore)"
        sleep "$INTERVAL"; continue
    fi

    # A slide still uploading appears as a dotfile with an rsync suffix; only
    # files that rsync has renamed into place are complete.
    # bash 3.2 (macOS default) has no mapfile
    READY=()
    while IFS= read -r L; do [ -n "$L" ] && READY+=("$L"); done < <(
        ssh fir 'find /scratch/davidg25/slides -name "*.svs" -not -name ".*" 2>/dev/null | sort')
    QUEUED=$(ssh fir 'squeue -u davidg25 -h -o "%i" 2>/dev/null | wc -l' | tr -d ' ')

    NEW=0
    for S in "${READY[@]}"; do
        [ -n "$S" ] || continue
        grep -Fxq "$S" "$SEEN" 2>/dev/null && continue
        if [ "$QUEUED" -ge "$MAX_QUEUED" ]; then
            log "queue at $QUEUED, holding off"
            break
        fi
        OUT=$(ssh fir "bash -lc 'cd /scratch/\$USER/mashpath && source cluster/fir_env.sh && \
              sbatch --export=ALL,SLIDE=\"$S\" cluster/steatosis_one.sbatch'" 2>&1 | tail -1)
        if echo "$OUT" | grep -q "Submitted batch job"; then
            echo "$S" >> "$SEEN"
            QUEUED=$((QUEUED + 1)); NEW=$((NEW + 1))
            log "submitted $(basename "$S")  -> ${OUT##* }"
        else
            log "submit FAILED for $(basename "$S"): $OUT"
        fi
    done

    DONE=$(wc -l < "$SEEN" | tr -d ' ')
    [ "$NEW" -gt 0 ] && log "  ${#READY[@]} slides on Fir, $DONE submitted so far"
    sleep "$INTERVAL"
done
