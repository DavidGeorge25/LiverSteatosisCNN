"""The dataset builder's bookkeeping, tested without opening a slide.

    python tests/test_dataset.py

Every property here is one that fails silently. A slide exported twice weights
that animal double and looks like a slightly larger dataset. A reserved batch
that slips into the export destroys the project's only external validation with
nothing downstream able to notice. A `split` column left in a manifest whose
whole design is "the split does not exist yet" is an invitation.

The paths handed to `build_flat_dataset` here do not exist, so every export
fails into `failures.json` -- which is fine and is the point: the bookkeeping
under test all happens before a slide is opened, and testing it this way needs
no 250 MB fixture.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from mashpath.config import MashConfig as Config  # noqa: E402
from mashpath.features.steatosis.dataset import (build_flat_dataset,  # noqa: E402
                                                 resolve_batches)

RESERVED = "2025-03-28_Celina 656D H&E"


def _tree() -> tuple[Path, Path, list[str]]:
    """A fake slide tree plus the .dripped that records it, with one slide
    reachable by two paths -- the situation on /scratch."""
    # Named "slides", because that is the only directory name
    # `batch_index` treats as "no batch folder" -- see
    # test_a_one_slide_batch_is_flagged for why that matters.
    root = Path(tempfile.mkdtemp()) / "slides"
    root.mkdir()
    paths = []
    for batch, names in ((RESERVED, ["R22-354_1_WT-CHOW1"]),
                         ("2026-04-20_CCl4_HE", ["R26-122-1_HE_70"]),
                         ("2025-08-25HE", ["R25-264-1", "R25-264-3"])):
        (root / batch).mkdir(parents=True, exist_ok=True)
        for n in names:
            f = root / batch / f"{n}.svs"
            f.write_bytes(b"not a slide")
            paths.append(str(f))
    # the same slide, also at the top level
    dup = root / "R26-122-1_HE_70.svs"
    dup.write_bytes(b"not a slide")
    paths.insert(0, str(dup))

    dripped = root / ".dripped"
    dripped.write_text("\n".join(paths) + "\n")
    return root, dripped, paths


def _build(out_name: str, **kw):
    root, dripped, paths = _tree()
    out = root / out_name
    cfg = Config.from_yaml([str(ROOT / "configs/tuned.yaml")])
    df = build_flat_dataset(paths, cfg, out, max_tiles_per_slide=1,
                            dripped=dripped, **kw)
    prov = json.loads((out / "provenance.json").read_text())
    return df, prov, out


def test_a_slide_reached_by_two_paths_is_exported_once():
    _, prov, _ = _build("ds")
    assert len(prov["duplicate_paths_ignored"]) == 1, prov["duplicate_paths_ignored"]
    assert "R26-122-1_HE_70" in prov["duplicate_paths_ignored"][0]


def test_the_reserved_batch_is_not_exported():
    _, prov, _ = _build("ds")
    assert prov["reserved_excluded"] == [RESERVED], prov["reserved_excluded"]
    assert prov["reserved_slides_excluded"] == ["R22-354_1_WT-CHOW1"], prov
    fails = json.loads((Path(prov and "x") and _build("ds2")[2] / "failures.json")
                       .read_text())
    assert all("R22-354" not in f["slide"] for f in fails), \
        "a reserved slide reached the export and merely failed to open"


def test_exporting_a_reserved_batch_has_to_be_typed():
    """`reserved=()` is the only way, and it exists so the decision is visible
    in the call rather than in a config someone can forget."""
    _, prov, _ = _build("ds3", reserved=())
    assert prov["reserved_excluded"] == [], prov
    assert prov["reserved_slides_excluded"] == [], prov


def test_batches_come_from_dripped_not_the_folder():
    root, dripped, paths = _tree()
    batch_of, guessed = resolve_batches(paths, dripped)
    assert not guessed, f"batch was guessed for {guessed}"
    assert batch_of["R26-122-1_HE_70"] == "2026-04-20_CCl4_HE", \
        "the top-level copy did not resolve to its real staining batch"
    # And a slide absent from .dripped falls back, and says so.
    stray = root / "2025-08-25HE" / "R99-999-1.svs"
    stray.write_bytes(b"")
    b2, g2 = resolve_batches(paths + [str(stray)], dripped)
    assert g2 == ["R99-999-1"] and b2["R99-999-1"] == "2025-08-25HE"


def test_the_manifest_has_no_split_column():
    df, _, _ = _build("ds4")
    assert "split" not in df.columns, \
        "a split column in a flat manifest is an invitation to use it"


def test_a_one_slide_batch_is_flagged():
    """`batch_index` recognises "no batch folder" by the literal directory name
    `slides`. Point MASHPATH_SLIDES somewhere else and a top-level slide gets
    that directory's name as its batch -- a batch of one, which becomes a
    leave-one-batch-out fold of one slide and quietly reports a meaningless
    number. Cheap to detect, so it is detected."""
    root = Path(tempfile.mkdtemp()) / "elsewhere"
    root.mkdir()
    (root / "2025-08-25HE").mkdir()
    good = [root / "2025-08-25HE" / f"R25-264-{i}.svs" for i in (1, 3)]
    stray = root / "R26-122-1_HE_70.svs"
    for f in good + [stray]:
        f.write_bytes(b"not a slide")
    dripped = root / ".dripped"
    dripped.write_text("\n".join(str(f) for f in [stray] + good) + "\n")

    cfg = Config.from_yaml([str(ROOT / "configs/tuned.yaml")])
    out = root / "ds"
    build_flat_dataset([str(f) for f in [stray] + good], cfg, out,
                       max_tiles_per_slide=1, dripped=dripped)
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["single_slide_batches"] == ["elsewhere"], \
        f"a one-slide batch was not flagged: {prov.get('single_slide_batches')}"


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
