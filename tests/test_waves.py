"""The wave fetch: render while the slides are here, decide once they all are.

    python tests/test_waves.py

65 GB of slides cannot sit on this laptop at once, so the study is built in
waves: fetch eight, render, delete, fetch eight more. Everything that makes the
draw defensible -- the batch-share cap, the time budget, which slides are
dropped -- is a property of the WHOLE collection, so none of it can be decided
inside a wave. This suite runs a two-wave build against real local slides and
checks that the finalize pass sees all six slides and opens none of them.

Skipped with a message if the slide data is not on this machine.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mashpath.core.slide import Slide  # noqa: E402
from mashpath.review import fullset, study  # noqa: E402

SLIDE_DIRS = [ROOT / "data/R25-264_MASH_HE", ROOT / "data/2026-04-20_CCl4_HE"]


def local_slides(n_per_dir: int = 3) -> list[Path]:
    out = []
    for d in SLIDE_DIRS:
        if d.exists():
            out += sorted(d.glob("*.svs"))[:n_per_dir]
    return out


def test_spread_is_prefix_stable():
    """The property the wave design rests on.

    A wave renders `MAX_PER_SECTION` fields per slide because the final
    allocation is not known yet. That is only sound if the first k of a
    24-field draw ARE the k-field draw at the same seed -- otherwise the
    finalize pass would ask for fields nobody rendered, and would have to fetch
    65 GB of slides a second time to get them.
    """
    import random

    frame = pd.DataFrame({
        "x": [(i % 40) * 512 for i in range(400)],
        "y": [(i // 40) * 512 for i in range(400)],
    })
    big = fullset._spread(frame, 24, random.Random("seed:slide"))
    for k in (4, 8, 12, 20):
        small = fullset._spread(frame, k, random.Random("seed:slide"))
        assert list(small.index) == list(big.index[:k]), (
            f"a {k}-field draw is not the prefix of a 24-field draw")


def test_repeats_only_ever_duplicate_a_field_that_was_drawn():
    """A repeat must reuse an already-rendered image, never introduce a new one."""
    frame = pd.DataFrame({
        "x": [(i % 20) * 512 for i in range(200)],
        "y": [(i // 20) * 512 for i in range(200)],
        "slide": ["s"] * 200, "batch": ["b"] * 200, "cohort": ["c"] * 200,
    })
    sel = fullset.plan_fields(frame, per_slide={"s": 12})
    uniq = sel[sel.sample_type == "uniform"]
    reps = sel[sel.sample_type == "repeat"]
    drawn = set(zip(uniq.x, uniq.y))
    assert len(reps) == fullset.n_repeats(12)
    assert set(zip(reps.x, reps.y)) <= drawn


def test_two_waves_then_a_finalize_that_opens_no_slide():
    slides = local_slides(3)
    if len(slides) < 6:
        print("  SKIP test_two_waves_then_a_finalize_that_opens_no_slide "
              "(needs local slide data)")
        return

    # Small numbers so the test renders in seconds rather than minutes.
    saved = (fullset.MAX_PER_SECTION, fullset.MIN_PER_SECTION,
             study.WARMUP_FIELDS_PER_SLIDE, study.WARMUP_SLIDES)
    fullset.MAX_PER_SECTION, fullset.MIN_PER_SECTION = 4, 2
    study.WARMUP_FIELDS_PER_SLIDE, study.WARMUP_SLIDES = 2, 1
    tmp = Path(tempfile.mkdtemp())
    try:
        work = tmp / "review_sets" / "wave_work"
        stage = tmp / "_wave"
        stage.mkdir(parents=True)
        (tmp / "review_sets").mkdir(exist_ok=True)

        waves = [slides[:3], slides[3:6]]
        for i, wave in enumerate(waves, 1):
            for f in stage.glob("*.svs"):
                f.unlink()                       # the previous wave is gone
            for src in wave:
                (stage / src.name).symlink_to(src)
            study.build_study([str(stage)], out_dir=None, work_dir=str(work),
                              stage="render", verbose=False)
            frame = pd.read_csv(tmp / "review_sets" / "study_frame.csv.gz")
            assert frame.slide.nunique() == 3 * i, (
                f"after wave {i} the frame holds {frame.slide.nunique()} "
                f"slides, not {3 * i} -- waves are not accumulating")

        # Every slide is now gone, exactly as after a real wave fetch.
        for f in stage.glob("*.svs"):
            f.unlink()

        opened = []
        real_init = Slide.__init__

        def spy(self, path, *a, **kw):
            opened.append(path)
            return real_init(self, path, *a, **kw)

        Slide.__init__ = spy
        try:
            study.build_study([str(stage)], out_dir=tmp / "pkg",
                              work_dir=str(work), stage="assemble",
                              warmup_dir=tmp / "warmup",
                              warmup_work_dir=str(tmp / "review_sets" / "warm"),
                              hours=0.6, verbose=False)
        finally:
            Slide.__init__ = real_init

        assert not opened, f"the finalize pass opened {len(opened)} slide(s): {opened[:3]}"
        key = pd.read_csv(work / "key.csv")
        assert key.slide.nunique() >= 4, key.slide.nunique()
        assert (tmp / "pkg" / "index.html").exists()
        assert (tmp / "warmup" / "index.html").exists()
        for rel in set(key["image"]) | set(key["context"]):
            assert (work / rel).exists(), f"{rel} was never rendered"
    finally:
        (fullset.MAX_PER_SECTION, fullset.MIN_PER_SECTION,
         study.WARMUP_FIELDS_PER_SLIDE, study.WARMUP_SLIDES) = saved
        shutil.rmtree(tmp, ignore_errors=True)


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
