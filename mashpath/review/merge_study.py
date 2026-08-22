"""Unblind and expand one returned study file into training-ready tables.

The page knows opaque section ids and field ids and nothing else. This joins
them back to slides and writes three tables:

  fields.csv   one row per field: verdict, cell count, seconds, sample_type
  cells.csv    one row per circled cell, in LEVEL-0 SLIDE coordinates, with a
               diameter in microns -- the point-supervision target
  grades.csv   one row per section: the NASH-CRN grade, the validation target

`sample_type` matters and is carried through: a `repeat` is a second look at a
field already judged, so prevalence must be computed on the `uniform` rows only
or every repeated positive is counted twice. The repeats are not waste -- they
are what `intra_rater` turns into the agreement ceiling any model gets compared
against.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# The field width in microns, hardcoded to match `study.js`, which computes
# every circle diameter against this exact number. It is only correct while a
# field is 512 level-0 pixels at ~0.4953 um/px, so `merge` checks rather than
# assumes -- a silent mismatch would rescale every cell diameter and every
# radius in cells.csv, and nothing downstream would look wrong.
TILE_UM = 253.6
FIELD_PX = 512


def _cells(raw: str) -> list[tuple[float, float, float]]:
    """`"x y d; x y d"` -> [(x, y, d)]. Tolerant: a mangled cell costs a cell."""
    out = []
    for part in (raw or "").split(";"):
        bits = part.replace(",", " ").split()
        if len(bits) != 3:
            continue
        try:
            x, y, d = float(bits[0]), float(bits[1]), float(bits[2])
        except ValueError:
            continue
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and d > 0:
            out.append((x, y, d))
    return out


def intra_rater(fields: list[dict], reviewer: str | None = None) -> dict:
    """Agreement between a repeat and its original, paired by tile position.

    Paired on (slide, x, y) rather than on field_id, because a repeat is issued
    under its OWN id precisely so that she cannot tell it is a repeat -- the ids
    do not point at each other and were never meant to.

    Cohen's kappa, not raw agreement: with most fields negative, agreeing by
    chance alone runs high, and a raw 90% would read as excellent while meaning
    almost nothing.
    """
    rows = [f for f in fields
            if reviewer is None or f.get("reviewer") == reviewer]
    groups: dict[tuple, list[dict]] = {}
    for f in rows:
        groups.setdefault((f["slide"], str(f["x"]), str(f["y"])), []).append(f)

    pairs: list[tuple[str, str]] = []
    for members in groups.values():
        reps = [m for m in members if m.get("sample_type") == "repeat"]
        firsts = [m for m in members if m.get("sample_type") != "repeat"]
        for r in reps:
            if firsts:
                pairs.append((firsts[0].get("verdict", ""), r.get("verdict", "")))

    n = len(pairs)
    if not n:
        return {"n_pairs": 0, "n_agree": 0, "agreement": None, "kappa": None}
    agree = sum(1 for a, b in pairs if a == b)
    po = agree / n
    cats = {c for pair in pairs for c in pair}
    pe = sum((sum(1 for a, _ in pairs if a == c) / n)
             * (sum(1 for _, b in pairs if b == c) / n) for c in cats)
    kappa = 1.0 if pe >= 1.0 else (po - pe) / (1.0 - pe)
    return {"n_pairs": n, "n_agree": agree, "agreement": round(po, 4),
            "kappa": round(kappa, 4)}


def merge(key_path, answers_path, out_dir=None, verbose: bool = True) -> Path:
    key_path, answers_path = Path(key_path), Path(answers_path)
    out_dir = Path(out_dir) if out_dir else key_path.parent
    key = {r["field_id"]: r for r in csv.DictReader(open(key_path, newline=""))}
    sect = {}
    for r in key.values():
        sect.setdefault(r["blind_id"], r)

    sizes = {int(r["size"]) for r in key.values()}
    if sizes != {FIELD_PX}:
        raise ValueError(
            f"key.csv has field size(s) {sorted(sizes)}, but study.js computes "
            f"circle diameters against a {FIELD_PX} px field ({TILE_UM} um). "
            f"Every diameter and radius here would be silently rescaled.")

    rows = list(csv.DictReader(open(answers_path, newline="")))
    fields, cells, grades, notes, unknown = [], [], [], [], []

    for a in rows:
        rec = (a.get("record") or "").strip()
        if rec == "note":
            notes.append((a.get("reviewer", ""), a.get("note", "")))
            continue
        if rec == "grade":
            m = sect.get((a.get("section_id") or "").strip())
            if not m:
                unknown.append(a.get("section_id"))
                continue
            grades.append(dict(reviewer=a.get("reviewer", ""), slide=m["slide"],
                               cohort=m["cohort"], batch=m["batch"],
                               diet=m.get("diet", "unknown"), split=m["split"],
                               blind_id=m["blind_id"], grade=a.get("grade", ""),
                               note=a.get("note", ""), utc=a.get("utc", "")))
            continue
        if rec != "field":
            continue
        k = key.get((a.get("field_id") or "").strip())
        if not k:
            unknown.append(a.get("field_id"))
            continue
        cc = _cells(a.get("cells", ""))
        fields.append(dict(reviewer=a.get("reviewer", ""), slide=k["slide"],
                           cohort=k["cohort"], batch=k["batch"],
                           diet=k.get("diet", "unknown"), split=k["split"],
                           sample_type=k["sample_type"], field_id=k["field_id"],
                           blind_id=k["blind_id"], x=k["x"], y=k["y"], size=k["size"],
                           verdict=a.get("verdict", ""), n_cells=len(cc),
                           seconds=a.get("seconds", ""), utc=a.get("utc", "")))
        size = int(k["size"])
        for nx, ny, d in cc:
            cells.append(dict(reviewer=a.get("reviewer", ""), slide=k["slide"],
                              cohort=k["cohort"], batch=k["batch"],
                              diet=k.get("diet", "unknown"), split=k["split"],
                              field_id=k["field_id"], sample_type=k["sample_type"],
                              slide_x=round(int(k["x"]) + nx * size),
                              slide_y=round(int(k["y"]) + ny * size),
                              diameter_um=round(d, 1),
                              # The pixel radius a crop or a mask would use.
                              radius_px=round(d / 2 / (TILE_UM / size), 1)))

    def w(name, recs):
        p = out_dir / name
        if not recs:
            return p
        with open(p, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(recs[0]))
            wr.writeheader()
            wr.writerows(recs)
        return p

    w("fields.csv", fields); w("cells.csv", cells); w("grades.csv", grades)

    if verbose:
        print(f"{len(fields)} fields, {len(cells)} cells, {len(grades)} grades "
              f"-> {out_dir}")
        if unknown:
            print(f"  {len(unknown)} unknown id(s) skipped")
        from collections import Counter
        uni = [f for f in fields if f["sample_type"] == "uniform"]
        if uni:
            pos = sum(1 for f in uni if f["verdict"] == "yes")
            print(f"  prevalence on the UNIFORM sample only: {pos}/{len(uni)} "
                  f"({pos/len(uni):.0%}) of fields positive")
        if "enriched" in {f["sample_type"] for f in fields}:
            print("  WARNING: `enriched` rows present. That stratum was removed; "
                  "this key.csv is stale.")
        ir = intra_rater(fields)
        if ir["n_pairs"]:
            print(f"  intra-rater on {ir['n_pairs']} repeated field(s): "
                  f"{ir['agreement']:.0%} agreement, kappa {ir['kappa']:.2f} "
                  f"-- the ceiling any model should be compared against")
        if cells:
            ds = sorted(c["diameter_um"] for c in cells)
            print(f"  circled cell diameter: median {ds[len(ds)//2]:.0f} µm, "
                  f"range {ds[0]:.0f}-{ds[-1]:.0f}")
        if grades:
            by = {}
            for g in grades:
                by.setdefault(g["cohort"], []).append(g["grade"])
            for coh, gs in sorted(by.items()):
                c = Counter(gs)
                print(f"  grade {coh:6s} n={len(gs):2d}  "
                      + " ".join(f"{k}:{c[k]}" for k in ("0","1","2","unsure") if c[k]))
        for who, txt in notes:
            print(f"  note from {who}: {txt}")
    return out_dir


def _main(argv=None) -> int:
    a = argv or sys.argv[1:]
    if len(a) < 2:
        print("usage: python -m mashpath.review.merge_study KEY.csv ANSWERS.csv [OUT_DIR]")
        return 2
    merge(a[0], a[1], a[2] if len(a) > 2 else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
