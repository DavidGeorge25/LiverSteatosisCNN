"""Steatosis numeric regression: the restructure must not move a single number.

Records a golden run from the pre-restructure `steatosis` package, then replays
the same run against `mashpath` and compares. The comparison is exact for every
per-tile column, not just the headline fat fraction -- a change that cancels out
in the slide total (one droplet gained here, one lost there) still fails.

Run:
    python tests/regression_steatosis.py --record    # before the move
    python tests/regression_steatosis.py             # after the move

Slides live outside the repo (data/ and *.svs are gitignored), so a missing
slide SKIPS rather than fails; the golden files are committed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GOLDEN = Path(__file__).resolve().parent / "golden"
CONFIG = ROOT / "configs" / "tuned.yaml"

# Two slides with opposite expected behaviour: a MASH slide carries real
# droplets (tests the accept path), a CCl4 slide is fat-free (tests that the
# shape filters still reject what they used to reject).
CASES = [
    ("mash", ROOT / "data" / "R25-264_MASH_HE" / "R25-264-1.svs"),
    ("ccl4", ROOT / "R26-122-1_HE_70.svs"),
]
LIMIT = 40

# Fields that legitimately differ run to run or after the output-layout change.
VOLATILE = {"seconds", "contact_sheet"}


def _import_pipeline():
    """Import from whichever package layout is present.

    During recording only `steatosis` exists; after the move only `mashpath`
    does. Same test file, both sides of the restructure.
    """
    try:
        from mashpath.config import MashConfig as Config
        from mashpath.features.steatosis.pipeline import run_slide
        return Config, run_slide, "mashpath"
    except ImportError:
        from steatosis.config import Config
        from steatosis.pipeline import run_slide
        return Config, run_slide, "steatosis"


def _run(slide_path: Path, out_dir: Path):
    Config, run_slide, pkg = _import_pipeline()
    cfg = Config.from_yaml(CONFIG)
    cfg.slide_path = str(slide_path)
    cfg.output_dir = str(out_dir)
    cfg.limit = LIMIT
    cfg.tiling.save_tiles = False       # numbers are unaffected; skips 40 PNG writes
    cfg.qc.save_per_tile_qc = False
    summary = run_slide(cfg, verbose_tiles=0, detail=False)
    tiles_csv = out_dir / slide_path.stem / "results" / "tiles.csv"
    if not tiles_csv.exists():                       # post-move layout
        tiles_csv = out_dir / slide_path.stem / "steatosis" / "tiles.csv"
    return summary, tiles_csv, pkg


def _clean(summary: dict) -> dict:
    return {k: v for k, v in sorted(summary.items()) if k not in VOLATILE}


def record(scratch: Path) -> int:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    for name, slide in CASES:
        if not slide.exists():
            print(f"  SKIP {name}: {slide} not present")
            continue
        summary, tiles_csv, pkg = _run(slide, scratch / name)
        (GOLDEN / f"{name}_summary.json").write_text(
            json.dumps(_clean(summary), indent=2, sort_keys=True) + "\n"
        )
        (GOLDEN / f"{name}_tiles.csv").write_text(tiles_csv.read_text())
        print(f"  recorded {name} from `{pkg}`: "
              f"macro_fat_fraction={summary['macro_fat_fraction']:.6f} "
              f"macro_count={summary['macro_count']}")
    return 0


def compare(scratch: Path) -> int:
    failures = 0
    checked = 0
    for name, slide in CASES:
        gold_summary = GOLDEN / f"{name}_summary.json"
        if not gold_summary.exists():
            print(f"  SKIP {name}: no golden file (run --record first)")
            continue
        if not slide.exists():
            print(f"  SKIP {name}: {slide} not present")
            continue

        summary, tiles_csv, pkg = _run(slide, scratch / name)
        checked += 1
        expected = json.loads(gold_summary.read_text())
        actual = _clean(summary)

        diffs = [
            f"      {k}: golden={expected.get(k)!r} now={actual.get(k)!r}"
            for k in sorted(set(expected) | set(actual))
            if expected.get(k) != actual.get(k)
        ]
        gold_rows = (GOLDEN / f"{name}_tiles.csv").read_text().splitlines()
        now_rows = tiles_csv.read_text().splitlines()
        if gold_rows != now_rows:
            n = sum(1 for a, b in zip(gold_rows, now_rows) if a != b)
            diffs.append(
                f"      tiles.csv: {n} differing row(s), "
                f"{len(gold_rows)} golden vs {len(now_rows)} now"
            )
            for a, b in zip(gold_rows, now_rows):
                if a != b:
                    diffs.append(f"        golden: {a}")
                    diffs.append(f"        now:    {b}")
                    break

        if diffs:
            failures += 1
            print(f"  FAIL {name} (via `{pkg}`)")
            print("\n".join(diffs))
        else:
            print(f"  PASS {name} (via `{pkg}`): "
                  f"{len(now_rows) - 1} tiles, "
                  f"macro_fat_fraction={summary['macro_fat_fraction']:.6f}, "
                  f"macro_count={summary['macro_count']}, "
                  f"every per-tile column identical")

    if checked == 0:
        print("  nothing checked -- no slides available")
        return 0
    return 1 if failures else 0


def test_steatosis_regression():
    """pytest entry point (asserts instead of returning a code)."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        assert compare(Path(td)) == 0


def main(argv: list[str]) -> int:
    import tempfile
    scratch_env = [a for a in argv if not a.startswith("-")]
    scratch = Path(scratch_env[0]) if scratch_env else Path(tempfile.mkdtemp())
    scratch.mkdir(parents=True, exist_ok=True)
    if "--record" in argv:
        print(f"recording golden baseline (limit={LIMIT} tiles, {CONFIG.name})")
        return record(scratch)
    print(f"steatosis regression (limit={LIMIT} tiles, {CONFIG.name})")
    return compare(scratch)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
