#!/bin/bash
# Launch chunked ballooning for every slide, then the finalize step.
#
# Ballooning is the only feature that cannot run as one task per slide: a
# single process reproducibly stalls partway through a slide on Fir (tile 675
# on R26-122-1_HE_70, three separate runs, GPU idle and one core spinning; the
# same tile completes in 2.5 s on a laptop). Splitting a slide into 300-tile
# chunks means a stall costs one chunk, not the run -- measured: 3 of 4 chunks
# completed and recovered 887 of 1187 tiles where the single-process job had
# produced nothing at all.
#
#   source cluster/fir_env.sh && bash cluster/submit_ballooning.sh
set -euo pipefail
: "${MASHPATH_SLIDES:?source cluster/fir_env.sh first}"
CHUNK="${CHUNK:-300}"
LIST="$MASHPATH_SLIDES/slide_list.txt"

while read -r SLIDE; do
    [ -n "$SLIDE" ] || continue
    N=$(python -m mashpath.features.ballooning.cli count --slide "$SLIDE" \
          --config configs/core.yaml --config configs/ballooning.yaml 2>/dev/null | tail -1)
    [ -n "$N" ] || { echo "SKIP $SLIDE (tile count failed)"; continue; }
    NC=$(( (N + CHUNK - 1) / CHUNK ))
    echo "$(basename "$SLIDE"): $N tiles -> $NC chunk(s)"
    sbatch --array=0-$((NC-1))%4 \
           --export=ALL,SLIDE="$SLIDE",CHUNK="$CHUNK" \
           cluster/ballooning_chunked.sbatch
done < "$LIST"

echo
echo "When every chunk array has finished, finalize each slide:"
echo "  while read -r S; do sbatch --export=ALL,SLIDE=\$S cluster/ballooning_finalize.sbatch; done < $LIST"
