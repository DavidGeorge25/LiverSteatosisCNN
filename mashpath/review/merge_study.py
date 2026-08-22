"""Unblind and expand one returned study file into training-ready tables.

The page knows opaque section ids and field ids and nothing else. This joins
them back to slides and writes three tables:

  fields.csv   one row per field: verdict, cell count, seconds, sample_type
  cells.csv    one row per circled cell, in LEVEL-0 SLIDE coordinates, with a
               diameter in microns -- the point-supervision target
  grades.csv   one row per section: the NASH-CRN grade, the validation target

`sample_type` matters and is carried through: prevalence and grade analyses must
use the `uniform` fields only. The `enriched` fields exist so positives are not
scarce for training, and including them in a prevalence estimate would inflate it.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

TILE_UM = 253.6


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


def merge(key_path, answers_path, out_dir=None, verbose: bool = True) -> Path:
    key_path, answers_path = Path(key_path), Path(answers_path)
    out_dir = Path(out_dir) if out_dir else key_path.parent
    key = {r["field_id"]: r for r in csv.DictReader(open(key_path, newline=""))}
    sect = {}
    for r in key.values():
        sect.setdefault(r["blind_id"], r)

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
                               cohort=m["cohort"], batch=m["batch"], split=m["split"],
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
                           cohort=k["cohort"], batch=k["batch"], split=k["split"],
                           sample_type=k["sample_type"], field_id=k["field_id"],
                           blind_id=k["blind_id"], x=k["x"], y=k["y"], size=k["size"],
                           verdict=a.get("verdict", ""), n_cells=len(cc),
                           seconds=a.get("seconds", ""), utc=a.get("utc", "")))
        size = int(k["size"])
        for nx, ny, d in cc:
            cells.append(dict(reviewer=a.get("reviewer", ""), slide=k["slide"],
                              cohort=k["cohort"], split=k["split"],
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
        enr = [f for f in fields if f["sample_type"] == "enriched"]
        if enr:
            pos = sum(1 for f in enr if f["verdict"] == "yes")
            print(f"  the enriched fields ran {pos}/{len(enr)} ({pos/max(len(enr),1):.0%}) "
                  f"positive -- compare with the line above to see whether the "
                  f"detector's score means anything at all")
        rep = [f for f in fields if f["sample_type"] == "repeat"]
        if rep:
            print(f"  {len(rep)} repeated field(s) for intra-rater agreement")
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
