#!/bin/bash
# One-screen status. Run it whenever: `./status.sh`   (or `./status.sh -w` to watch)
#
# Everything comes from one SSH round trip plus a couple of local stats, so it
# is fast enough to sit in a watch loop without being annoying.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ONEDRIVE="/Users/davidgeorge/Library/CloudStorage/OneDrive-McMasterUniversity/Dongdong Wang's files - Training"
TOTAL_SLIDES=260

b()  { printf "\033[1m%s\033[0m" "$1"; }
dim(){ printf "\033[2m%s\033[0m" "$1"; }
grn(){ printf "\033[32m%s\033[0m" "$1"; }
ylw(){ printf "\033[33m%s\033[0m" "$1"; }
red(){ printf "\033[31m%s\033[0m" "$1"; }

bar() { # bar <done> <total> <width>
    local d=$1 t=$2 w=${3:-28} f
    [ "$t" -gt 0 ] || t=1
    f=$(( d * w / t )); [ "$f" -gt "$w" ] && f=$w
    printf "["
    for ((i=0;i<w;i++)); do [ $i -lt $f ] && printf "#" || printf "."; done
    printf "] %d/%d (%d%%)" "$d" "$t" $(( d * 100 / t ))
}

render() {
    clear 2>/dev/null
    echo
    b "  MASHPATH  ";  dim "$(date '+%a %H:%M:%S')"; echo; echo

    # ---- one SSH round trip for everything remote ----
    local REMOTE
    if ssh -O check fir >/dev/null 2>&1; then
        REMOTE=$(ssh fir 'echo "LANDED=$(find /scratch/davidg25/slides -name "*.svs" -not -name ".*" 2>/dev/null | wc -l | tr -d " ")"
            echo "FLIGHT=$(find /scratch/davidg25/slides -name ".*.svs.*" 2>/dev/null | wc -l | tr -d " ")"
            echo "GB=$(du -sm /scratch/davidg25/slides 2>/dev/null | cut -f1)"
            echo "RUN=$(squeue -u $USER -h -t RUNNING 2>/dev/null | wc -l | tr -d " ")"
            echo "PEND=$(squeue -u $USER -h -t PENDING 2>/dev/null | wc -l | tr -d " ")"
            echo "STEAT=$(ls /scratch/davidg25/outputs/steatosis/*/steatosis/tiles.csv 2>/dev/null | wc -l | tr -d " ")"
            echo "INFL=$(ls /scratch/davidg25/outputs/inflammation/*/inflammation/nuclei_summary.json 2>/dev/null | wc -l | tr -d " ")"
            echo "BAL=$(ls /scratch/davidg25/outputs/ballooning/*/ballooning/cells.csv 2>/dev/null | wc -l | tr -d " ")"' 2>/dev/null)
        eval "$REMOTE" 2>/dev/null
    else
        echo "  $(red "SSH socket down") — run: $(b "ssh fir")  (answer Duo, then close the window)"
        echo
        LANDED=0; FLIGHT=0; GB=0; RUN=0; PEND=0; STEAT=0; INFL=0; BAL=0
    fi
    LANDED=${LANDED:-0}; FLIGHT=${FLIGHT:-0}; GB=${GB:-0}
    RUN=${RUN:-0}; PEND=${PEND:-0}; STEAT=${STEAT:-0}; INFL=${INFL:-0}; BAL=${BAL:-0}

    # ---- upload ----
    b "  UPLOAD"; echo
    printf "    slides   "; bar "$LANDED" "$TOTAL_SLIDES" 28; echo
    printf "    data     %s GB on Fir" "$(( GB / 1024 ))"
    [ "$FLIGHT" -gt 0 ] && printf "   %s in flight" "$(ylw "$FLIGHT")"
    echo
    local nrs; nrs=$(pgrep -f "[r]sync.*fir:" | wc -l | tr -d ' ')
    if [ "$nrs" -gt 0 ]; then
        local left=$(( (TOTAL_SLIDES - LANDED) * 480 / 1024 ))   # ~480 MB/slide avg
        printf "    %s  %s stream(s), ~%s GB to go" "$(grn "running")" "$nrs" "$left"
        [ "$left" -gt 0 ] && printf "  (~%dh at 0.7 MB/s)" $(( left * 1024 / 2520 ))
        echo
    else
        printf "    %s — no rsync process\n" "$(red "STOPPED")"
    fi
    echo

    # ---- compute ----
    b "  COMPUTE"; echo
    printf "    steatosis     "; bar "$STEAT" "$LANDED" 20; echo
    printf "    inflammation  "; bar "$INFL"  "$LANDED" 20; echo
    printf "    ballooning    "; bar "$BAL"   "$LANDED" 20; echo
    printf "    jobs: %s running, %s queued\n" "$(grn "$RUN")" "$(ylw "$PEND")"
    echo

    # ---- local ----
    b "  LOCAL"; echo
    local free; free=$(df -g / | awk 'NR==2{print $4}')
    printf "    disk free     %s GB" "$free"
    [ "${free:-99}" -lt 20 ] && printf "  %s" "$(red "<< LOW")"
    echo
    local drip; drip=$(pgrep -f "[d]rip_feed" | wc -l | tr -d ' ')
    [ "$drip" -gt 0 ] && printf "    drip-feed     %s\n" "$(grn "running")" \
                      || printf "    drip-feed     %s\n" "$(red "stopped")"
    echo

    # ---- the thing that actually gates the project ----
    b "  BLOCKER"; echo
    local verdicts=0
    [ -f "$ROOT/cluster/verdicts.csv" ] && verdicts=$(( $(wc -l < "$ROOT/cluster/verdicts.csv") - 1 ))
    if [ "$verdicts" -le 0 ]; then
        printf "    pathologist labels: %s — nothing downstream can be trained or\n" "$(red "0")"
        printf "    validated until a review session happens.\n"
    else
        printf "    pathologist labels: %s\n" "$(grn "$verdicts")"
    fi
    echo
    dim "  ./status.sh -w  to auto-refresh"; echo; echo
}

if [ "${1:-}" = "-w" ]; then
    while :; do render; sleep "${2:-60}"; done
else
    render
fi
