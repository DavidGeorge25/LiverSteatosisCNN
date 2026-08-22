#!/bin/bash
# Fetch the study slides from Fir in WAVES, rendering fields and deleting each
# slide before the next wave. ~65 GB of slides pass through ~8 GB of disk.
#
#   bash cluster/fetch_study_slides.sh [remote] [wave_size]
#
# REQUIRES AN INTERACTIVE TERMINAL. Alliance enforces MFA on every new SSH
# connection, so the first rsync will prompt for a second factor. Open the
# multiplexed control connection once first and the rest of the run is silent:
#
#   mkdir -p ~/.ssh/cm && ssh -MNf fir-dtn     # answer the MFA prompt once
#
# Only ~2.5 MB of JPEG per slide is kept, so the full 120-slide study lands in
# well under a gigabyte. Re-running skips slides already rendered, so an
# interrupted transfer resumes rather than restarting.
#
# TWO STAGES, AND THE SPLIT MATTERS.
#
#   --stage render    once per wave, while the slides are on disk. Accumulates
#                     study_frame.csv and renders the CEILING of fields per
#                     slide (fullset.MAX_PER_SECTION), because the final
#                     allocation is not knowable yet.
#   --stage assemble  once, at the end. Decides the batch-share cap, the time
#                     budget and which slides are dropped over EVERY slide, and
#                     writes the packages from the rendered images. Opens no
#                     slide, so it runs after the pixels are gone.
#
# An earlier version of this script ran the whole build per wave. That decided
# the batch balance and the time budget on eight slides at a time, and each
# wave overwrote study_frame.csv with only its own slides.
set -euo pipefail

# The paths in the CSV are ABSOLUTE paths on the cluster
# (/scratch/davidg25/slides/<batch>/<slide>.svs), so the rsync source is the
# remote root and --no-relative flattens each file into the staging directory.
# Batch folder names contain spaces and an ampersand, which is exactly why this
# goes through --files-from rather than an argument list.
REMOTE="${1:-fir:/}"
WAVE="${2:-8}"
LIST="review_sets/proposed_slide_set.csv"
STAGE="data/_wave"
WORK="review_sets/ballooning_full"
OUT="$HOME/Desktop/Ballooning_Study"

[ -f "$LIST" ] || { echo "missing $LIST -- run the inclusion planner first"; exit 1; }
mkdir -p "$STAGE" "$WORK"

# Column 5 is the slide's path on the cluster; skip the header, anything already
# local, and anything a previous run already rendered.
#
# The resume key is key.csv, which is written only after a wave's rendering
# finishes. An interrupted wave therefore re-fetches that whole wave -- correct,
# and nearly free, because build() reuses any image already on disk. Keying on
# study_frame.csv instead would skip slides that were framed but never rendered,
# and the finalize pass would then want pixels that had already been deleted.
DONE=/tmp/rendered_slides.txt
: > "$DONE"
[ -f "$WORK/key.csv" ] && tail -n +2 "$WORK/key.csv" | cut -d, -f5 | sort -u > "$DONE"

# `mapfile` is a bash 4 builtin and macOS ships bash 3.2, so this reads into
# the array the portable way. The previous version failed at this line with
# "mapfile: command not found" before transferring a single byte.
NEED=()
while IFS= read -r line; do
  NEED+=("$line")
done < <(awk -F, 'NR>1 && $3=="no" {print $5}' "$LIST" |
  while IFS= read -r p; do
    name=$(basename "$p" .svs)
    grep -qxF "$name" "$DONE" || echo "$p"
  done)
echo "${#NEED[@]} slide(s) to fetch, $WAVE at a time"
echo "($(wc -l < "$DONE" | tr -d ' ') already rendered, skipping those)"

i=0
while [ $i -lt ${#NEED[@]} ]; do
  batch=("${NEED[@]:$i:$WAVE}")
  echo
  echo "=== wave $((i/WAVE + 1)): ${#batch[@]} slide(s) ==="
  printf '%s\n' "${batch[@]}" > /tmp/wave_files.txt
  rsync -a --no-relative --files-from=/tmp/wave_files.txt \
        "$REMOTE" "$STAGE/"

  echo "--- rendering fields ---"
  .venv/bin/python -m mashpath.cli study \
      --slide-dir "$STAGE" --work-dir "$WORK" --stage render \
      || { echo "rendering failed; slides kept in $STAGE for inspection"; exit 1; }

  echo "--- clearing staged slides ---"
  rm -f "$STAGE"/*.svs
  i=$((i + WAVE))
  df -h . | tail -1
done

echo
echo "=== all waves fetched; assembling over every slide ==="
.venv/bin/python -m mashpath.cli study \
    --slide-dir data/R25-264_MASH_HE --slide-dir data/2026-04-20_CCl4_HE \
    --slide-dir "$STAGE" \
    --out "$OUT" --work-dir "$WORK" --stage assemble \
    --hours 0 --fields-per-section 16 --round ballooning_full_v3

echo
echo "=== verifying before anything is sent ==="
.venv/bin/python tests/verify_package.py "$OUT" "$WORK"
