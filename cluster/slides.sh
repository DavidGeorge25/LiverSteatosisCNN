#!/bin/bash
# Build the slide list every array job indexes into.
#
# Written to a FILE rather than globbed inside each task on purpose: a glob
# re-evaluated per task can return a different order if the directory changes
# mid-run, and array task 7 would then process a different slide on a resubmit
# than it did the first time. The list is the record of what a run covered.
set -euo pipefail
SLIDES="${MASHPATH_SLIDES:-/scratch/$USER/slides}"
OUT="${1:-$SLIDES/slide_list.txt}"
find "$SLIDES" -name '*.svs' -o -name '*.tif' -o -name '*.ndpi' \
  | sort > "$OUT"
echo "$(wc -l < "$OUT") slides -> $OUT"
