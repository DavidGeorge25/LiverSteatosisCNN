"""The return path: her CSV back into training-ready tables.

    python tests/test_ingest.py

This is the half of the study nobody exercises until it is too late to fix. She
works for hours, emails one file, and every downstream number comes from
`merge_study.merge` parsing it correctly. So the file is SYNTHESISED here --
against the real `key.csv`, in the app's own hand-rolled CSV format, including a
partly-finished session -- and the merge is run for real.

The CSV the app writes is not produced by a csv library: `study.js` joins each
row with commas and quotes only the fields it passes through `q()`. Mirroring
that here rather than using `csv.writer` is the point. A test that writes a
tidier file than the app does would pass while the real file failed.
"""

from __future__ import annotations

import csv
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mashpath.review.merge_study import intra_rater, merge  # noqa: E402

KEY = ROOT / "review_sets/ballooning_full/key.csv"

# How often a field is called positive, by cohort. Chow should be near zero and
# MASH highest; the exact numbers do not matter, only that they differ, so the
# per-cohort reporting has something to separate.
POSITIVE_RATE = {"ACLY656": 0.08, "MASH": 0.25, "CCl4": 0.12, "unknown": 0.15}


def _q(text: str) -> str:
    return '"' + str(text or "").replace('"', '""').replace("\n", " ") + '"'


def fake_answers(key_rows: list[dict], path: Path, seed: int = 7,
                 fraction: float = 1.0, agreement: float = 0.85) -> dict:
    """Write a session file exactly as `study.js` would.

    `fraction` stops partway through, which is the expected case: she is not
    required to finish. `agreement` is how often a repeat gets the same verdict
    as its original, so intra-rater kappa has something real to measure.
    """
    rng = random.Random(seed)
    by_section: dict[str, list[dict]] = {}
    for r in key_rows:
        by_section.setdefault(r["blind_id"], []).append(r)
    order = sorted(by_section, key=lambda s: int(by_section[s][0]["present_order"]))
    order = order[: max(1, round(fraction * len(order)))]

    rows = [["round", "session", "reviewer", "record", "section_id", "field_id",
             "verdict", "grade", "n_cells", "cells", "seconds", "note", "utc"]]
    utc = "2026-08-25T10:00:00.000Z"
    truth: dict[tuple, str] = {}          # (slide,x,y) -> the first verdict given
    n_fields = n_cells = n_grades = 0
    repeats_agreeing = repeats_total = 0

    for sid in order:
        panels = sorted(by_section[sid], key=lambda r: int(r["panel"]))
        for k in panels:
            where = (k["slide"], k["x"], k["y"])
            rate = POSITIVE_RATE.get(k["cohort"], 0.15)
            if where in truth:            # a repeat: agree most of the time
                repeats_total += 1
                if rng.random() < agreement:
                    verdict = truth[where]
                    repeats_agreeing += 1
                else:
                    verdict = rng.choice([v for v in ("yes", "no", "unsure")
                                          if v != truth[where]])
            else:
                verdict = ("yes" if rng.random() < rate
                           else ("unsure" if rng.random() < 0.06 else "no"))
                truth[where] = verdict
            cells = ""
            n = 0
            if verdict == "yes":
                n = rng.randint(1, 4)
                cells = "; ".join(
                    f"{rng.uniform(.05,.95):.4f} {rng.uniform(.05,.95):.4f} "
                    f"{rng.randint(18, 60)}" for _ in range(n))
            n_cells += n
            n_fields += 1
            rows.append(["ballooning_full_v3", "TESTSESSION", "EK", "field", sid,
                         k["field_id"], verdict, "", n, _q(cells),
                         f"{rng.uniform(6, 30):.1f}", "", utc])
        grade = rng.choice(["0", "1", "1", "2", "unsure"])
        n_grades += 1
        rows.append(["ballooning_full_v3", "TESTSESSION", "EK", "grade", sid, "",
                     "", grade, "", "", "", _q(""), utc])

    rows.append(["ballooning_full_v3", "TESTSESSION", "EK", "note", "", "", "",
                 "", "", "", "", _q("circling is fine, some fields very pale"), utc])
    path.write_text("\n".join(",".join(str(c) for c in r) for r in rows))
    return dict(fields=n_fields, cells=n_cells, grades=n_grades,
                sections=len(order), repeats=repeats_total,
                repeats_agreeing=repeats_agreeing)


def _load_key() -> list[dict]:
    with open(KEY, newline="") as fh:
        return list(csv.DictReader(fh))


def test_a_finished_session_round_trips():
    if not KEY.exists():
        print("  SKIP test_a_finished_session_round_trips (no key.csv)")
        return
    key_rows = _load_key()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        want = fake_answers(key_rows, td / "answers.csv")
        merge(KEY, td / "answers.csv", td, verbose=False)

        fields = list(csv.DictReader(open(td / "fields.csv", newline="")))
        cells = list(csv.DictReader(open(td / "cells.csv", newline="")))
        grades = list(csv.DictReader(open(td / "grades.csv", newline="")))

        assert len(fields) == want["fields"] == len(key_rows), (
            f"{len(fields)} fields merged, {want['fields']} written, "
            f"{len(key_rows)} in the key")
        assert len(cells) == want["cells"], f"{len(cells)} != {want['cells']}"
        assert len(grades) == want["grades"]
        assert {g["grade"] for g in grades} <= {"0", "1", "2", "unsure"}


def test_circles_land_inside_the_field_they_were_drawn_on():
    """The whole point of the cell table: a mark has to come back as a real
    place on a real slide, not a fraction of something."""
    if not KEY.exists():
        print("  SKIP test_circles_land_inside_the_field_they_were_drawn_on")
        return
    key_rows = _load_key()
    by_id = {r["field_id"]: r for r in key_rows}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake_answers(key_rows, td / "answers.csv")
        merge(KEY, td / "answers.csv", td, verbose=False)
        cells = list(csv.DictReader(open(td / "cells.csv", newline="")))
        assert cells
        for c in cells:
            k = by_id[c["field_id"]]
            x, y, size = int(k["x"]), int(k["y"]), int(k["size"])
            cx, cy = float(c["slide_x"]), float(c["slide_y"])
            assert x <= cx <= x + size, f"{cx} outside [{x}, {x + size}]"
            assert y <= cy <= y + size, f"{cy} outside [{y}, {y + size}]"
            assert c["slide"] == k["slide"]
            # 18-60 um at 0.4953 um/px is 36-121 px across, so 18-61 px radius.
            assert 15 <= float(c["radius_px"]) <= 65, c["radius_px"]


def test_an_unfinished_session_merges_what_is_there():
    """She is not required to finish, so a partial file is the expected case,
    not an error case."""
    if not KEY.exists():
        print("  SKIP test_an_unfinished_session_merges_what_is_there")
        return
    key_rows = _load_key()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        want = fake_answers(key_rows, td / "answers.csv", fraction=0.4)
        merge(KEY, td / "answers.csv", td, verbose=False)
        fields = list(csv.DictReader(open(td / "fields.csv", newline="")))
        grades = list(csv.DictReader(open(td / "grades.csv", newline="")))
        assert len(fields) == want["fields"] < len(key_rows)
        assert len(grades) == want["sections"]
        # And a partial run still spans every batch, which is what the
        # interleaved section order was for.
        assert len({f["batch"] for f in fields}) == 9


def test_repeats_pair_up_and_give_an_intra_rater_kappa():
    if not KEY.exists():
        print("  SKIP test_repeats_pair_up_and_give_an_intra_rater_kappa")
        return
    key_rows = _load_key()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        want = fake_answers(key_rows, td / "answers.csv", agreement=0.85)
        merge(KEY, td / "answers.csv", td, verbose=False)
        fields = list(csv.DictReader(open(td / "fields.csv", newline="")))
        stats = intra_rater(fields)
        assert stats["n_pairs"] == want["repeats"] > 0, stats
        assert stats["n_agree"] == want["repeats_agreeing"], stats
        assert 0.6 < stats["agreement"] < 1.0, stats
        assert -1.0 <= stats["kappa"] <= 1.0, stats


def test_perfect_and_chance_agreement_bracket_kappa():
    """Kappa is the number a model gets compared against, so its two ends have
    to be right: identical answers are 1.0, and answers that agree only as often
    as the marginals predict are ~0."""
    same = [{"field_id": f"f{i}", "slide": "s", "x": i, "y": 0,
             "sample_type": t, "verdict": v}
            for i, v in enumerate(["yes", "no", "no", "yes", "no"])
            for t in ("uniform", "repeat")]
    assert abs(intra_rater(same)["kappa"] - 1.0) < 1e-9

    flipped = []
    for i, (a, b) in enumerate([("yes", "no"), ("no", "yes"), ("yes", "no"),
                                ("no", "yes")]):
        flipped.append({"field_id": f"g{i}a", "slide": "s", "x": i, "y": 0,
                        "sample_type": "uniform", "verdict": a})
        flipped.append({"field_id": f"g{i}b", "slide": "s", "x": i, "y": 0,
                        "sample_type": "repeat", "verdict": b})
    assert intra_rater(flipped)["kappa"] < 0.0


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
