"""Join a hosted round's answers back onto the tiles they were drawn from.

The page is deliberately ignorant: it knows `f014`, not
`R25-264-33_x021504_y007680`. That is what makes it safe to hand to a reviewer
we are asking to judge blind -- the slide, score and band are not withheld by
the interface, they are not in the file. The cost is this step.

Output matches the desktop app's verdicts.csv column for column, so a hosted
round and a laptop round pool into one training set instead of two formats.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from .names import VERDICTS_NAME

FEATURE = "ballooning_tile"
TILE_PX = 512          # tiling.tile_size at level 0; a mark is a fraction of it


def _parse_marks(raw: str) -> list[tuple[float, float]]:
    """`"0.41 0.62; 0.77 0.30"` -> [(0.41, 0.62), (0.77, 0.30)].

    Tolerant on purpose: this string may have been through a spreadsheet and an
    email client before it gets here, and a mangled point should cost that point
    rather than the whole round.
    """
    out: list[tuple[float, float]] = []
    for part in (raw or "").split(";"):
        bits = part.replace(",", " ").split()
        if len(bits) != 2:
            continue
        try:
            x, y = float(bits[0]), float(bits[1])
        except ValueError:
            continue
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            out.append((x, y))
    return out
VERDICT_CODE = {"yes": "y", "no": "n", "unsure": "u",
                "y": "y", "n": "n", "u": "u"}


def merge(key_path: str | Path, answers_path: str | Path,
          out_path: str | Path | None = None, verbose: bool = True) -> Path:
    key_path, answers_path = Path(key_path), Path(answers_path)
    key = {r["field_id"]: r for r in csv.DictReader(open(key_path, newline=""))}
    answers = list(csv.DictReader(open(answers_path, newline="")))
    # `round` and `session` were added so two files from one person -- a
    # restart, a second sitting, a different machine -- stay distinguishable.
    # Older files lack them; absent is fine, misread is not.
    sessions = {(a.get("session") or "").strip() for a in answers} - {""}
    if verbose and len(sessions) > 1:
        print(f"  {len(sessions)} sessions in this file: {sorted(sessions)}")

    out_path = Path(out_path) if out_path else key_path.parent / VERDICTS_NAME
    points_path = out_path.parent / "points.csv"
    unknown, written = [], 0
    points: list[list] = []
    notes: list[tuple] = []
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "reviewer", "slide", "feature", "candidate_id",
                    "verdict", "notes", "seconds", "pass_index"])
        seen: dict[str, int] = {}
        for a in answers:
            fid = (a.get("field_id") or "").strip()
            # Her free-text checkpoint notes ride in the same file under a
            # reserved id. They are not verdicts and must not be counted as
            # skipped rows either.
            if fid.startswith("_note_at_"):
                notes.append((fid.replace("_note_at_", ""), a.get("reviewer", ""),
                              a.get("marks", "")))
                continue
            k = key.get(fid)
            if not k:
                unknown.append(fid)
                continue
            v = VERDICT_CODE.get((a.get("verdict") or "").strip().lower())
            if not v:
                unknown.append(fid)
                continue
            # pass_index distinguishes a deliberate repeat from a first look, the
            # same way the desktop app does -- intra-rater agreement needs the
            # two to be distinguishable, and the tile id alone cannot do it.
            tile = k["tile_id"]
            idx = seen.get(tile, 0)
            seen[tile] = idx + 1
            w.writerow([a.get("answered_utc", ""), a.get("reviewer", ""),
                        k["slide"], FEATURE, tile, v, "",
                        a.get("seconds", ""), idx])
            written += 1

            # Clicks arrive as fractions of the tile. Turned into LEVEL-0 SLIDE
            # coordinates they become comparable with anything else measured on
            # the slide -- above all the detector's own proposed cells, which is
            # the check the earlier candidate round could never make: it only
            # ever showed her its own proposals, so it could measure precision
            # and was blind to recall.
            for px_, py_ in _parse_marks(a.get("marks", "")):
                points.append([a.get("reviewer", ""), k["slide"], tile, idx,
                               round(int(k["x"]) + px_ * TILE_PX),
                               round(int(k["y"]) + py_ * TILE_PX),
                               round(px_, 4), round(py_, 4)])

    if points:
        with open(points_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["reviewer", "slide", "tile_id", "pass_index",
                        "slide_x", "slide_y", "tile_frac_x", "tile_frac_y"])
            w.writerows(points)

    if verbose:
        print(f"{written} verdict(s) -> {out_path}")
        if points:
            print(f"{len(points)} marked cell(s) -> {points_path}")
        if unknown:
            print(f"  {len(unknown)} row(s) skipped (unknown field id or verdict): "
                  f"{unknown[:6]}{' ...' if len(unknown) > 6 else ''}")
        answered = {a.get("field_id") for a in answers}
        missing = [f for f in key if f not in answered]
        if missing:
            print(f"  {len(missing)} field(s) not answered -- a partial round is "
                  f"fine, they are simply absent from the training set")
        # The one number worth reading immediately.
        rows = list(csv.DictReader(open(out_path, newline="")))
        if rows:
            from collections import Counter
            c = Counter(r["verdict"] for r in rows)
            n = len(rows)
            print(f"  yes {c['y']} ({c['y']/n:.0%})  no {c['n']} ({c['n']/n:.0%})  "
                  f"unsure {c['u']} ({c['u']/n:.0%})")
            secs = [float(r["seconds"]) for r in rows
                    if (r["seconds"] or "").replace(".", "", 1).isdigit()]
            if secs:
                secs.sort()
                print(f"  median {secs[len(secs)//2]:.1f}s per field "
                      f"(the estimate this round exists to test was 11s)")
        # Counts, not just presence. Brunt et al. 2022 measured Fleiss kappa
        # 0.197 among nine expert liver pathologists for presence/absence of
        # ballooning, rising to 0.362 at a >=5-cell threshold -- so the count is
        # the more reproducible label and the thresholds are worth seeing.
        per_tile: dict[str, int] = {}
        for pt in points:
            per_tile[pt[2]] = per_tile.get(pt[2], 0) + 1
        yes_tiles = [r["candidate_id"] for r in rows if r["verdict"] == "y"]
        if yes_tiles:
            counted = [per_tile.get(t, 0) for t in yes_tiles]
            marked = [c for c in counted if c]
            print(f"  of {len(yes_tiles)} positive field(s): "
                  f"{len(marked)} had cells marked, "
                  f"{sum(1 for c in counted if c >= 5)} had >=5")
            if marked:
                marked.sort()
                print(f"  cells per marked field: median {marked[len(marked)//2]}, "
                      f"max {marked[-1]}")
            if len(marked) < len(yes_tiles):
                print(f"  NOTE: {len(yes_tiles) - len(marked)} positive field(s) "
                      f"carry no marks -- usable as tile labels, not as counts")
        if notes:
            print(f"\n  {len(notes)} note(s) from the reviewer:")
            for at, who, txt in notes:
                print(f"    [after field {at}] {who}: {txt}")
    return out_path


def _main(argv: list[str] | None = None) -> int:
    a = argv or sys.argv[1:]
    if len(a) < 2:
        print("usage: python -m mashpath.review.merge_hosted KEY.csv ANSWERS.csv [OUT.csv]")
        return 2
    merge(a[0], a[1], a[2] if len(a) > 2 else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
