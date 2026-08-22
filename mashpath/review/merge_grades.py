"""Join slide grades back onto the sections they describe.

The grading page shows opaque section ids in a shuffled order, so it is blind
to cohort. This is the step that unblinds it, and it is also where the one
number the round exists for gets computed: whether the pathologist's grade
separates the cohorts at all.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

GRADES = {"0": 0, "1": 1, "2": 2}


def merge(key_path: str | Path, answers_path: str | Path,
          out_path: str | Path | None = None, verbose: bool = True) -> Path:
    key_path, answers_path = Path(key_path), Path(answers_path)
    rows = list(csv.DictReader(open(key_path, newline="")))
    meta = {}
    for r in rows:
        meta.setdefault(r["blind_id"], r)

    answers = list(csv.DictReader(open(answers_path, newline="")))
    out_path = Path(out_path) if out_path else key_path.parent / "grades.csv"

    written, unknown = [], []
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["reviewer", "session", "slide", "cohort", "batch", "split",
                    "blind_id", "grade", "seconds", "note", "graded_utc"])
        for a in answers:
            b = (a.get("blind_id") or "").strip()
            m = meta.get(b)
            if not m:
                unknown.append(b)
                continue
            w.writerow([a.get("reviewer", ""), a.get("session", ""), m["slide"],
                        m["cohort"], m["batch"], m["split"], b,
                        a.get("grade", ""), a.get("seconds", ""),
                        a.get("note", ""), a.get("graded_utc", "")])
            written.append({"cohort": m["cohort"], "slide": m["slide"],
                            "grade": a.get("grade", "")})

    if verbose:
        print(f"{len(written)} slide grade(s) -> {out_path}")
        if unknown:
            print(f"  {len(unknown)} unknown section id(s) skipped: {unknown[:5]}")
        from collections import Counter
        by = {}
        for r in written:
            by.setdefault(r["cohort"], []).append(r["grade"])
        for coh, gs in sorted(by.items()):
            c = Counter(gs)
            spread = " ".join(f"{k}:{c[k]}" for k in ("0", "1", "2", "unsure") if c[k])
            print(f"  {coh:6s} n={len(gs):2d}   {spread}")
        # The comparison the round exists to make. CCl4 is not a ballooning
        # model -- its hydropic degeneration is the mimic -- so a grade
        # distribution that does NOT separate it from MASH is the finding, and
        # it is the same finding the detector already gave at AUC 0.085.
        num = {coh: [GRADES[g] for g in gs if g in GRADES] for coh, gs in by.items()}
        if len(num) >= 2 and all(num.values()):
            try:
                from scipy.stats import mannwhitneyu
                keys = sorted(num)
                a_, b_ = num[keys[0]], num[keys[1]]
                u, p = mannwhitneyu(a_, b_)
                auc = u / (len(a_) * len(b_))
                print(f"\n  {keys[0]} vs {keys[1]}: median "
                      f"{sorted(a_)[len(a_)//2]} vs {sorted(b_)[len(b_)//2]}, "
                      f"AUC {auc:.2f}, p={p:.3g}")
                print("  (AUC 0.5 = the grade does not distinguish the cohorts)")
            except ImportError:
                pass
    return out_path


def _main(argv: list[str] | None = None) -> int:
    a = argv or sys.argv[1:]
    if len(a) < 2:
        print("usage: python -m mashpath.review.merge_grades KEY.csv GRADES.csv [OUT.csv]")
        return 2
    merge(a[0], a[1], a[2] if len(a) > 2 else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
