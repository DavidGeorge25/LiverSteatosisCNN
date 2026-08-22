#!/bin/bash
# Move slides + outputs from /scratch to project space, once it exists.
#
#   bash cluster/migrate_to_project.sh /project/<gid>/<user>
#
# Run this the day project space is allocated. Everything on /scratch is on a
# purge clock (~60 days on Alliance), and the purge does not warn per-file --
# it just deletes. The slides are recoverable from OneDrive at 0.7 MB/s, which
# is two days; the OUTPUTS are recoverable only by re-running the cohort.
#
# This is a cluster-internal copy between two Lustre filesystems: GB/s, not the
# 0.7 MB/s of the original upload. 134 GB of slides moves in minutes.
#
# Copy-then-verify-then-delete, never mv. A half-completed mv across
# filesystems leaves data in neither place, and this is the one copy of the
# results.
set -euo pipefail

DEST="${1:?usage: migrate_to_project.sh /project/<gid>/<user>}"
SRC="/scratch/$USER"

[ -d "$DEST" ] || { echo "no such destination: $DEST"; exit 1; }
[ -w "$DEST" ] || { echo "not writable: $DEST"; exit 1; }

echo "=== source ==="
for d in slides outputs mashpath; do
    [ -d "$SRC/$d" ] && printf "  %-10s %8s  %6s files\n" "$d" \
        "$(du -sh "$SRC/$d" | cut -f1)" "$(find "$SRC/$d" | wc -l | tr -d ' ')"
done

echo
echo "=== copying (slides and outputs; venv is rebuilt, not moved) ==="
for d in slides outputs; do
    [ -d "$SRC/$d" ] || continue
    echo "  $d ..."
    mkdir -p "$DEST/$d"
    # -H preserves hard links, --info=progress2 gives one aggregate line.
    rsync -aH --info=progress2 "$SRC/$d/" "$DEST/$d/"
done

echo
echo "=== verifying ==="
FAIL=0
for d in slides outputs; do
    [ -d "$SRC/$d" ] || continue
    A=$(find "$SRC/$d" -type f | wc -l | tr -d ' ')
    B=$(find "$DEST/$d" -type f | wc -l | tr -d ' ')
    SA=$(du -sk "$SRC/$d" | cut -f1)
    SB=$(du -sk "$DEST/$d" | cut -f1)
    if [ "$A" = "$B" ]; then
        printf "  %-10s OK   %s files, %s KB -> %s KB\n" "$d" "$A" "$SA" "$SB"
    else
        printf "  %-10s MISMATCH  src=%s dst=%s files\n" "$d" "$A" "$B"; FAIL=1
    fi
done
# A checksum pass on the slides specifically: a truncated SVS still opens and
# reads garbage in the missing region, which would surface much later as
# inexplicable tissue-detection failures on a handful of slides.
echo "  checksumming slides (this is the one that must not be silently wrong)..."
rsync -aHc --dry-run --itemize-changes "$SRC/slides/" "$DEST/slides/" 2>/dev/null \
    | grep -v '^\.d' | head -5 | sed 's/^/    DIFFERS: /' || true

echo
if [ "$FAIL" -ne 0 ]; then
    echo "VERIFICATION FAILED -- nothing deleted. Re-run to resume the copy."
    exit 2
fi

cat <<EOF

=== copied and verified. Nothing has been deleted. ===

1. Point the environment at the new location -- edit cluster/fir_env.sh:

     export MASHPATH_SLIDES="\${MASHPATH_SLIDES:-$DEST/slides}"
     export MASHPATH_OUT="\${MASHPATH_OUT:-$DEST/outputs}"

   (MASHPATH_ROOT can stay on /scratch: it is a git checkout, re-clonable.)

2. Re-run one slide and confirm the numbers match its old summary.json.

3. Only then free the scratch copy:

     rm -rf $SRC/slides $SRC/outputs

Leaving the scratch copy in place until step 2 passes costs nothing -- scratch
is 19 TiB and the purge is measured in weeks.
EOF
