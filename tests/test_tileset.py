"""Tile-set sampling tests: the draw, the split, the repeats, the leak.

Runs under plain python (no pytest needed):

    python tests/test_tileset.py

No slide is opened. Everything here is the sampling logic, which is where the
mistakes that matter live -- a set drawn from one slide, a repeat that is
indistinguishable from a correction, a held-out slide that moves between
builds, or a band printed on the page in front of the reviewer. Rendering is
covered by actually building a set; these are the properties a build cannot
check about itself.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mashpath.review import tileset as T  # noqa: E402
from mashpath.review.tiles_app import TileSet  # noqa: E402

TILE = 512


def _frame(n_slides: int = 12, per_slide: int = 300, cohorts=("MASH", "CCl4")):
    """A synthetic frame shaped like a real one: scores, cohorts, a grid."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_slides):
        name = f"S{i:02d}"
        cohort = cohorts[i % len(cohorts)]
        for j in range(per_slide):
            rows.append({
                "tile_id": f"{name}_x{j:06d}_y000000",
                "slide": name, "cohort": cohort, "batch": f"B{i % 3}",
                "diet": "unknown", "scored": True,
                "x": (j % 20) * TILE, "y": (j // 20) * TILE, "size": TILE,
                "tissue_fraction": 1.0,
                "score": float(rng.normal()), "n_scored_cells": 3,
            })
    f = pd.DataFrame(rows)
    f["score_pct"] = f.groupby("slide")["score"].rank(pct=True) * 100
    return T.assign_bands(f, T.TilePlan())


def test_band_shares_must_sum_to_one():
    plan = T.TilePlan(band_shares={"high": 0.5, "upper": 0.2, "mid": 0.1,
                                   "low": 0.1, "anchor": 0.05})
    try:
        plan.validate()
    except T.TileSetError as exc:
        assert "sum to 1.0" in str(exc), exc
    else:
        raise AssertionError("a band_shares that does not sum to 1 was accepted")


def test_unknown_plan_key_is_rejected(tmp: Path | None = None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "plan.yaml"
        p.write_text("tileset:\n  n_presentations: 100\n  duplicate_fractoin: 0.1\n")
        try:
            T.TilePlan.from_yaml(p)
        except T.TileSetError as exc:
            assert "duplicate_fractoin" in str(exc), exc
        else:
            raise AssertionError("a misspelled plan key was silently ignored")


def test_draw_respects_slide_cap_and_spreads():
    """No slide may dominate, and every slide should appear."""
    plan = T.TilePlan(n_presentations=600, max_slide_share=0.10)
    picked = T.draw(_frame(), plan, verbose=False)
    counts = picked["slide"].value_counts()
    cap = max(-(-int(1.4 * plan.n_unique) // 12),
              round(plan.max_slide_share * plan.n_unique))
    assert counts.max() <= cap, f"a slide exceeded the cap: {counts.max()} > {cap}"
    assert len(counts) == 12, f"only {len(counts)} of 12 slides contributed"


def test_slide_cap_never_makes_the_target_unfillable():
    """Few slides + a low share must relax to an even split, not come up short.

    The bug this pins: 14 slides at a 6% ceiling can supply 84% of the target,
    so the draw silently returned a short set for a reason that had nothing to
    do with the material.
    """
    plan = T.TilePlan(n_presentations=600, max_slide_share=0.02,
                      cohort_caps={}, min_tile_separation=0)
    picked = T.draw(_frame(n_slides=8), plan, verbose=False)
    assert len(picked) == plan.n_unique, \
        f"drew {len(picked)} of {plan.n_unique} with a cap that should have relaxed"


def test_short_draw_names_its_binding_constraint():
    """A short set must say WHY. The fixes are opposite.

    Too few slides -> add slides. One cohort capped -> add slides from the
    others. Window too small -> the separation cannot hold. From the band
    totals alone these are indistinguishable.
    """
    plan = T.TilePlan(n_presentations=600, cohort_caps={"CCl4": 0.20},
                      min_tile_separation=0)
    picked = T.draw(_frame(n_slides=8), plan, verbose=False)
    b = picked.attrs["blocked"]
    assert picked.attrs["shortfalls"], "expected this plan to be unfillable"
    assert b["cohort_cap"] > b["separation"], b
    assert max(b, key=b.get) == "cohort_cap", b


def test_cohort_cap_is_enforced():
    """And is a share of the set she actually sees, not of the target.

    A short draw used to loosen this silently: the cap is an absolute count
    computed against the target, so a set landing at 70% of target carried
    ~1.4x the intended CCl4 share. Separation now relaxes rather than the set
    shrinking, which is what keeps this a real 20%.
    """
    plan = T.TilePlan(n_presentations=600, cohort_caps={"CCl4": 0.20})
    picked = T.draw(_frame(), plan, verbose=False)
    caps = picked.attrs["cohort_caps"]
    for band, grp in picked.groupby("band"):
        n = int((grp["cohort"] == "CCl4").sum())
        assert n <= caps[band]["CCl4"], \
            f"band {band}: CCl4 took {n}, cap was {caps[band]['CCl4']}"
    # The cap is an absolute count against the TARGET, so a short draw raises
    # the realised share above the nominal 20%. Bounded, and worth knowing.
    assert (picked["cohort"] == "CCl4").mean() <= 0.25


def test_capped_cohort_reaches_every_band():
    """A global cap put every CCl4 tile in the first bands drawn.

    CCl4 hydropic degeneration is the ballooning mimic her opinion is most
    needed on, and it is needed across the score range -- not only where the
    detector already shouted. Measured before the fix: `low`, `mid` and
    `upper` contained zero CCl4 tiles.
    """
    plan = T.TilePlan(n_presentations=600, cohort_caps={"CCl4": 0.20})
    picked = T.draw(_frame(), plan, verbose=False)
    for band in T.ALL_BANDS:
        grp = picked[picked["band"] == band]
        if len(grp) < 10:
            continue
        assert (grp["cohort"] == "CCl4").any(), f"band {band} has no CCl4 tile"


def test_anchor_is_drawn_first_and_never_short():
    """The anchor is the only stratum that can estimate true prevalence.

    It is drawn before the banded strata precisely so that a shortfall
    elsewhere cannot eat it.
    """
    plan = T.TilePlan(n_presentations=600)
    picked = T.draw(_frame(), plan, verbose=False)
    got = int((picked["band"] == "anchor").sum())
    want = round(plan.band_shares["anchor"] * plan.n_unique)
    assert got == want, f"anchor drew {got}, wanted {want}"


def test_min_separation_between_sampled_tiles():
    # No cohort cap: with half the synthetic slides capped at 20% the draw is
    # short for that reason, separation relaxes, and this would test nothing.
    plan = T.TilePlan(n_presentations=400, min_tile_separation=2,
                      cohort_caps={})
    picked = T.draw(_frame(), plan, verbose=False)
    assert picked.attrs["separation"] == 2, "separation relaxed; test is void"
    for slide, grp in picked.groupby("slide"):
        pts = list(zip(grp["x"], grp["y"]))
        for i, (x1, y1) in enumerate(pts):
            for x2, y2 in pts[i + 1:]:
                d = max(abs(x1 - x2), abs(y1 - y2))
                assert d >= 2 * TILE, f"{slide}: two tiles {d}px apart"


def test_split_is_by_slide_and_stable_across_builds():
    """A slide must land on the same side whatever else is in the draw.

    Otherwise a slide that was train in a pilot can be test in the full set,
    and its pilot labels become training data for a model evaluated on it.
    """
    plan = T.TilePlan()
    big = T.assign_split(T.draw(_frame(n_slides=12), plan, verbose=False),
                         plan, verbose=False)
    small = T.assign_split(T.draw(_frame(n_slides=5), plan, verbose=False),
                           plan, verbose=False)
    a = big.groupby("slide")["split"].first()
    b = small.groupby("slide")["split"].first()
    for s in b.index:
        assert a[s] == b[s], f"{s} moved from {a[s]} to {b[s]} between builds"
    # and no tile of a test slide leaks into train
    for slide, grp in big.groupby("slide"):
        assert grp["split"].nunique() == 1, f"{slide} is split across both sides"


def test_duplicates_get_their_own_id_and_point_back():
    """A deliberate repeat must be distinguishable from a correction.

    Same id twice = she pressed back and changed her mind. A repeat carries a
    NEW id with dup_of set, so intra-rater agreement counts what it means to.
    """
    plan = T.TilePlan(n_presentations=600, duplicate_fraction=0.10)
    picked = T.assign_split(T.draw(_frame(), plan, verbose=False), plan, False)
    rows = T.add_duplicates(picked, plan, verbose=False)
    dups = rows[rows["dup_of"] != ""]
    assert len(rows["tile_id"]) == rows["tile_id"].nunique(), "a tile_id repeats"
    assert abs(len(dups) / len(rows) - 0.10) < 0.02, \
        f"repeats were {len(dups) / len(rows):.1%} of the session"
    assert set(dups["dup_of"]) <= set(picked["tile_id"])


def test_duplicate_count_follows_the_actual_draw():
    """If a band comes up short, 10% must stay 10% of what she is shown."""
    plan = T.TilePlan(n_presentations=600, duplicate_fraction=0.10)
    picked = T.assign_split(T.draw(_frame(n_slides=3, per_slide=40), plan, False),
                            plan, False)
    rows = T.add_duplicates(picked, plan, verbose=False)
    frac = (rows["dup_of"] != "").mean()
    assert abs(frac - 0.10) < 0.03, f"repeats were {frac:.1%} of a short set"


def test_repeats_are_placed_far_from_their_original():
    plan = T.TilePlan(n_presentations=900, min_duplicate_gap=120)
    picked = T.assign_split(T.draw(_frame(), plan, verbose=False), plan, False)
    seq = T.order_presentations(T.add_duplicates(picked, plan, False), plan, False)
    gaps = T._duplicate_gaps(seq)
    assert gaps, "no repeat was placed"
    assert min(gaps) >= 120, f"a repeat landed {min(gaps)} presentations after its original"


def test_order_is_not_blocked_by_slide():
    """Consecutive tiles from one slide let her calibrate to that section."""
    plan = T.TilePlan(n_presentations=900)
    picked = T.assign_split(T.draw(_frame(), plan, verbose=False), plan, False)
    seq = T.order_presentations(T.add_duplicates(picked, plan, False), plan, False)
    # The property that matters is that consecutive tiles are no more likely
    # to share a slide than chance -- not that no run of 4 ever occurs, which
    # a fair shuffle produces on its own.
    n_slides = seq["slide"].nunique()
    adjacent = (seq["slide"] == seq["slide"].shift()).mean()
    assert adjacent < 2.0 / n_slides, \
        f"{adjacent:.1%} of neighbours share a slide, chance is {1/n_slides:.1%}"


def test_tile_score_is_the_max_over_veto_passing_cells(tmp: Path | None = None):
    """The label is 'contains at least one', so the aggregator must be max."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cells.csv"
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["tile_x", "tile_y", "score"])
            w.writerow([0, 0, 1.0])
            w.writerow([0, 0, 7.5])       # the striking cell
            w.writerow([0, 0, ""])        # vetoed, must not count
            w.writerow([512, 0, ""])      # a tile the detector proposed nothing in
        t = T.tile_scores(p).set_index(["x", "y"])
        assert t.loc[(0, 0), "score"] == 7.5
        assert t.loc[(0, 0), "n_scored_cells"] == 2
        assert np.isnan(t.loc[(512, 0), "score"]), "an empty tile got a score"
        assert t.loc[(512, 0), "n_scored_cells"] == 0


def test_unscored_tiles_reach_the_low_band_not_the_middle():
    """A tile the detector examined and rejected ranks BELOW its worst proposal."""
    f = _frame(n_slides=2, per_slide=100)
    f.loc[f.index[:20], "score"] = np.nan
    filled = f["score"].fillna(-np.inf)
    pct = pd.Series(0.0, index=f.index)
    fin = np.isfinite(filled)
    pct[fin] = filled[fin].rank(pct=True) * 100
    f["score_pct"] = pct
    banded = T.assign_bands(f, T.TilePlan())
    assert set(banded.loc[f.index[:20], "band"]) == {"low"}


def test_frame_patches_counts_disjoint_windows():
    """One contiguous raster range is one patch; spaced windows are several.

    This is the number that tells a reader whether a slide's band percentiles
    describe the whole section or one strip of it, so it has to be right for
    both shapes -- and it must not merge two windows that only touch
    diagonally, which is a different piece of tissue, not the same one.
    """
    def frame(coords):
        return pd.DataFrame([{"x": x * TILE, "y": y * TILE, "size": TILE}
                             for x, y in coords])

    strip = [(i % 20, i // 20) for i in range(60)]        # 3 full rows
    assert T._count_patches(frame(strip)) == 1

    spaced = ([(i, 0) for i in range(10)]                  # window A
              + [(i, 40) for i in range(10)]               # window B, far off
              + [(i, 80) for i in range(10)])              # window C
    assert T._count_patches(frame(spaced)) == 3

    # Diagonal touch is NOT connectivity: (0,0) and (1,1) are corner-to-corner
    # and share no edge, so they are two patches, not one.
    assert T._count_patches(frame([(0, 0), (1, 1)])) == 2
    assert T._count_patches(frame([(0, 0), (1, 0)])) == 1
    assert T._count_patches(pd.DataFrame(columns=["x", "y", "size"])) == 0


def test_report_flags_single_strip_slides():
    """A one-strip slide must be named in the report, not silently averaged in."""
    f = _frame(n_slides=8, per_slide=200)
    # S00 measured as one contiguous range; everything else as spaced windows.
    f["frame_patches"] = np.where(f["slide"] == "S00", 1, 4)
    f["frame_tiles"] = 200
    plan = T.TilePlan(n_presentations=200, max_slide_share=0.30)
    seq = T.order_presentations(
        T.add_duplicates(
            T.assign_split(T.draw(f, plan, verbose=False), plan, verbose=False),
            plan, verbose=False),
        plan, verbose=False)
    sources = {
        name: T.SlideSource(name=name, path=Path(f"{name}.svs"), cohort="MASH",
                            batch="B0", cells_csv=Path(__file__))
        for name in f["slide"].unique()
    }
    cfg = type("C", (), {"tiling": type("T2", (), {"tile_size": 512,
                                                   "level": 0})()})()
    report = T.sampling_report(seq, plan, sources, cfg)
    assert "one strip" in report
    tail = report.split("were scored as a single")[1]
    named = tail.split("\n\n")[0]
    assert "`S00`" in named
    assert "`S01`" not in named          # spaced windows must not be flagged


def test_spaced_windows_merge_only_when_declared(tmp: Path | None = None):
    """A gap must stay an error by default and be permitted only on request.

    Both halves matter. Sampling a slide as spaced windows is the whole reason
    the frame covers the section rather than one strip of it, so the merge has
    to accept it -- but the same code path is what catches a chunk whose job
    died, and silently accepting that would corrupt every z-score near the
    hole. The flag is the difference between the two, so it is tested as a
    difference, not as a feature.
    """
    import tempfile
    from mashpath.features.ballooning import pipeline as P

    tmp = tmp or Path(tempfile.mkdtemp())
    d = tmp / "S1" / "ballooning" / "chunks"
    d.mkdir(parents=True)
    for a, b in ((0, 100), (900, 1000)):          # deliberately far apart
        tag = f"{a:06d}_{b:06d}"
        pd.DataFrame({"x": [a], "y": [0], "is_hepatocyte": [True]}).to_pickle(
            d / f"cells_{tag}.pkl")
        # One contour per cell: merge_chunks cross-checks the two counts and
        # refuses a mismatch, which is a separate guard from the gap one.
        poly = [np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int32)]
        P._save_contours(d / f"contours_{tag}.npz", poly)
        P._save_contours(d / f"nuccontours_{tag}.npz", poly)

    cfg = type("C", (), {"output_dir": str(tmp)})()

    try:
        P.merge_chunks(cfg, "S1", verbose=False)
    except ValueError as exc:
        assert "chunk gap" in str(exc)
        assert "allow_disjoint" in str(exc)      # the error must say the way out
    else:
        raise AssertionError("a gap must be an error unless declared")

    df, _, _, _ = P.merge_chunks(cfg, "S1", verbose=False, allow_disjoint=True)
    assert len(df) == 2                           # both windows, nothing dropped


def test_browser_payload_carries_no_slide_identity():
    """The /api/tiles payload must not contain the slide name in any form.

    The manifest test below covers the FILE; this covers the WIRE, and they
    fail differently. The tile_id embeds the slide
    (`R26-122-23_HE_91_x026112_y014848`), so a payload that merely omits a
    `slide` key while shipping the id has leaked the cohort to anyone who
    opens devtools -- and this reviewer is being asked to judge blind
    precisely because knowing the section biases the call.
    """
    import json as _json
    from mashpath.review import tiles_app

    rows = [{"tile_id": "R26-122-23_HE_91_x026112_y014848", "slide": "R26-122-23_HE_91",
             "tile_um": "253.6", "context_um": "760.8", "core_box": "341 341 341 341",
             "dup_of": ""}]
    payload = [{"index": i, "tile_um": r.get("tile_um", ""),
                "context_um": r.get("context_um", ""),
                "box": r.get("core_box", ""), "judged": False}
               for i, r in enumerate(rows)]
    blob = _json.dumps(payload)
    for forbidden in ("R26-122", "slide", "score", "band", "split", "cohort"):
        assert forbidden not in blob, f"{forbidden!r} reached the browser"

    # And the real handler must build exactly that shape -- guard against the
    # keys creeping back in via the source.
    src = tiles_app.__doc__ or ""
    assert "slide name is never shown" in src
    import inspect
    handler = inspect.getsource(tiles_app._Handler.do_GET)
    body = handler.split('elif u.path == "/api/tiles":')[1].split("elif")[0]
    assert '"id"' not in body, "tile_id is back in the /api/tiles payload"
    assert '"slide"' not in body


def test_exclusion_keeps_two_packages_disjoint(tmp: Path | None = None):
    """A second set drawn from one frame must not redraw the first set's tiles.

    Both packages come from the same frame at the same seed, so without this
    they overlap heavily -- measured at 68 tiles, a quarter of the pilot. Since
    a pilot label is ordinary training data, that overlap is the reviewer
    judging the same field twice for nothing, so it is waste rather than a
    second opinion. Repeats are excluded from the id list on purpose: a repeat
    is the same pixels under a second id, so counting it would exclude nothing
    new while overstating the overlap.
    """
    import tempfile
    tmp = tmp or Path(tempfile.mkdtemp())
    f = _frame(n_slides=10, per_slide=200)
    plan = T.TilePlan(n_presentations=200, max_slide_share=0.30)

    first = T.draw(f, plan, verbose=False)
    taken = set(first["tile_id"])

    second = T.draw(f[~f["tile_id"].isin(taken)], plan, verbose=False)
    assert not (set(second["tile_id"]) & taken), "the second set redrew tiles"
    assert len(second) == plan.n_unique, "exclusion must not starve the draw"

    # read_tile_ids must ignore repeats, which carry their own ids.
    pkg = tmp / "pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    rows = T.add_duplicates(T.assign_split(first, plan, verbose=False),
                            plan, verbose=False)
    rows.to_csv(pkg / "sampling_frame.csv", index=False)
    ids = T.read_tile_ids(pkg)
    assert ids == taken, "read_tile_ids must return uniques, not repeats"


def test_bundle_is_standalone_and_withholds_the_frame(tmp: Path | None = None):
    """A reviewer's copy must run without the science stack and without scores.

    Two failure modes, both silent. If the bundled app imports anything outside
    the standard library it will not start on her laptop, and we find out on
    the morning of the session. If `sampling_frame.csv` rides along, the score,
    band and split are one double-click away from someone we are asking to
    judge blind -- the app withholds them, and shipping the file would hand
    them over anyway.
    """
    import tempfile
    from mashpath.review import bundle

    tmp = tmp or Path(tempfile.mkdtemp())
    pkg = tmp / "set_x"
    (pkg / "images").mkdir(parents=True)
    (pkg / "context").mkdir(parents=True)
    (pkg / "manifest.csv").write_text(
        "order,tile_id,slide,image,context,tile_um,context_um,core_box,dup_of\n"
        "1,S1_x0_y0,S1,images/a.png,context/a.jpg,253.6,760.8,341 341 341 341,\n")
    (pkg / "sampling_frame.csv").write_text("tile_id,score,band,split\nS1_x0_y0,9.9,high,test\n")
    (pkg / "sampling_report.md").write_text("# secret")
    (pkg / "verdicts.csv").write_text("timestamp,reviewer\n2020,OLD\n")
    (pkg / "images" / "a.png").write_bytes(b"\x89PNG")
    (pkg / "context" / "a.jpg").write_bytes(b"\xff\xd8")

    out = bundle.build(pkg, tmp / "send", port=8000, brief=None, verbose=False)

    for leaked in ("sampling_frame.csv", "sampling_report.md", "verdicts.csv"):
        assert not (out / "set_x" / leaked).exists(), f"{leaked} shipped"
    assert (out / "set_x" / "manifest.csv").exists()
    assert (out / "set_x" / "images" / "a.png").exists()

    # Every bundled module must import using ONLY the standard library.
    import ast, sys
    stdlib = getattr(sys, "stdlib_module_names", None)
    for m in bundle.MODULES:
        src = (out / "review" / m).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.Import):
                mod = node.names[0].name.split(".")[0]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mod = node.module.split(".")[0]
            if mod and stdlib is not None:
                assert mod in stdlib, f"{m} imports non-stdlib {mod!r}"

    # The shim must not re-import the candidate flow the real __init__ does.
    # Parsed, not grepped: its docstring explains why it imports nothing, so a
    # substring search for "import" matches the explanation and fails.
    shim = ast.parse((out / "review" / "__init__.py").read_text())
    assert not [n for n in ast.walk(shim)
                if isinstance(n, (ast.Import, ast.ImportFrom))], \
        "the bundle shim imports something; it must be inert"


def test_marks_survive_the_round_trip_to_slide_coordinates(tmp: Path | None = None):
    """A click must come back as a point on the slide, not a point in a viewport.

    Marks are stored as fractions of the tile because that is the only frame
    that outlives the session: she may click at 5x zoom, and what has to survive
    is a location on the slide. This pins the whole chain -- fraction in,
    level-0 coordinate out, inside the tile it came from -- because an off-by-a-
    factor here is invisible in the CSV and only shows up as points scattered
    over the wrong part of a section months later.
    """
    import tempfile
    from mashpath.review import verdicts as V
    from mashpath.review.merge_hosted import _parse_marks, TILE_PX

    tmp = tmp or Path(tempfile.mkdtemp())
    path = tmp / "verdicts.csv"
    V.append(path, reviewer="EK", slide="S1", feature="ballooning_tile",
             candidate_id="S1_x001024_y002048", verdict="y", seconds=9.0,
             marks="0.2500 0.7500; 0.0000 1.0000")
    V.append(path, reviewer="EK", slide="S1", feature="ballooning_tile",
             candidate_id="S1_x001024_y002048", verdict="n", seconds=4.0)

    rows = V.load(path)
    assert rows[0]["marks"] == "0.2500 0.7500; 0.0000 1.0000"
    assert rows[1]["marks"] == ""              # a no carries no marks
    assert [r["pass_index"] for r in rows] == ["0", "1"]

    tx, ty = 1024, 2048
    pts = _parse_marks(rows[0]["marks"])
    assert len(pts) == 2
    got = [(tx + nx * TILE_PX, ty + ny * TILE_PX) for nx, ny in pts]
    assert got[0] == (1024 + 128, 2048 + 384)
    # The corners are the ones that expose a flipped or transposed axis.
    assert got[1] == (1024, 2048 + 512)
    for sx, sy in got:
        assert tx <= sx <= tx + TILE_PX and ty <= sy <= ty + TILE_PX


def test_verdicts_written_before_marks_existed_still_load(tmp: Path | None = None):
    """The marks column was appended, so a nine-column file must still parse.

    Sessions already recorded are a pathologist's afternoon. If widening the
    schema shifted the columns, every one of those rows would silently gain a
    verdict from the wrong field.
    """
    import tempfile
    from mashpath.review import verdicts as V

    tmp = tmp or Path(tempfile.mkdtemp())
    old = tmp / "old.csv"
    old.write_text(
        "timestamp,reviewer,slide,feature,candidate_id,verdict,notes,seconds,pass_index\n"
        "2026-08-01T10:00:00+00:00,EK,S1,ballooning_tile,S1_x0_y0,y,,8.0,0\n")
    rows = V.load(old)
    assert len(rows) == 1
    r = rows[0]
    assert r["verdict"] == "y" and r["seconds"] == "8.0" and r["slide"] == "S1"
    assert not r.get("marks")          # absent, not misaligned


def test_manifest_carries_no_score_band_or_split():
    """The reviewer may open the manifest. It must not anchor her."""
    for col in ("score", "score_pct", "band", "split", "cohort", "diet"):
        assert col not in T.MANIFEST_COLUMNS, f"{col} is in the manifest"
    for col in ("score", "band", "split"):
        assert col in T.FRAME_COLUMNS, f"{col} is missing from the analysis frame"


def test_app_rejects_a_manifest_it_cannot_show(tmp: Path | None = None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / T.MANIFEST_NAME
        p.write_text("tile_id,slide\nA,S1\n")     # no image/context columns
        try:
            TileSet(d)
        except ValueError as exc:
            assert "image" in str(exc) and "context" in str(exc), exc
        else:
            raise AssertionError("a manifest with no images was accepted")


def test_accession_parses_for_every_naming_form_in_the_collection():
    """The nine batches spell the accession three different ways.

    Splitting on "-" only worked when the accession was followed by another
    hyphen, so every underscore-separated name returned itself and lost its
    cohort -- including all eight R22-354 slides, the only batch with a
    within-batch treatment contrast.
    """
    from mashpath.review.tileset import COHORT_BY_PREFIX, _prefix

    cases = {
        "R25-264-1": "R25-264",                    # hyphen separator
        "R26-122-1_HE_70": "R26-122",
        "R25-079_1_HE": "R25-079",                 # underscore separator
        "R25-151_4_HE": "R25-151",
        "R22-354_16_ACLY656 CHOW1": "R22-354",     # underscore, spaces, digits
        "R22-354_25_ACLY656 NASH 6": "R22-354",
        "R22--063-10_HE": "R22-063",               # doubled hyphen
        "R25-223": "R25-223",                      # bare accession
    }
    for name, want in cases.items():
        assert _prefix(name) == want, f"{name} -> {_prefix(name)}, wanted {want}"
    assert COHORT_BY_PREFIX[_prefix("R22-354_1_WT-CHOW1")] == "ACLY656"


def test_section_order_spreads_every_batch_through_the_run():
    """She may stop at any point, so a prefix has to look like the whole set.

    A plain shuffle is unbiased but has variance: on the real 146-section draw
    it put only one of the four chow sections in the first half, and chow is the
    only normal liver in the collection. Interleaving removes the unlucky draws
    without moving the expected composition.
    """
    import pandas as pd

    from mashpath.review.fullset import section_order

    fields = pd.DataFrame({
        "slide": [f"{b}_{i}" for b in "ABCDEFGHI" for i in range(14)]
                 + [f"RARE_{i}" for i in range(4)],
        "batch": [b for b in "ABCDEFGHI" for _ in range(14)] + ["RARE"] * 4,
    })
    order = section_order(fields, seed=1)
    assert len(order) == len(fields)
    assert set(order) == set(fields["slide"])
    batch_of = dict(zip(fields["slide"], fields["batch"]))

    # A rare batch cannot be spread over the whole run -- it runs out. What it
    # must do is FINISH EARLY, so that stopping partway through still leaves it
    # complete. That is the property the chow sections need.
    at = [i for i, s in enumerate(order) if batch_of[s] == "RARE"]
    assert max(at) < 0.6 * len(order), f"RARE not complete until {max(at)}"
    assert at[1] - at[0] > 1, "RARE sections are consecutive, not interleaved"

    # Any prefix past the first round covers every batch, and no batch runs away
    # with it the way a shuffle can.
    for frac in (0.25, 0.5):
        n = int(frac * len(order))
        seen = [batch_of[s] for s in order[:n]]
        assert len(set(seen)) == 10, f"prefix {n} covers {len(set(seen))}/10 batches"
        top = max(seen.count(b) for b in set(seen)) / n
        assert top < 0.2, f"prefix {n} is {100 * top:.0f}% one batch"


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
