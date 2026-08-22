"""Core tests: config layering, micron conversion, TIFF reading, review flow.

Runs under plain python (no pytest needed):

    python tests/test_core.py

Every test is also a pytest-collectable `test_*` function.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mashpath.config import MashConfig  # noqa: E402
from mashpath.core.config import ReviewConfig, SlideConfig  # noqa: E402
from mashpath.core.io import SlideOutputs  # noqa: E402
from mashpath.core.slide import MissingScaleError, Slide  # noqa: E402
from mashpath.features.steatosis.config import FatConfig  # noqa: E402
from mashpath.features.steatosis.detect import MACRO, detect_fat  # noqa: E402
from mashpath.review import Candidate, export_candidates, load_verdicts  # noqa: E402

MPP = 0.4953  # the cohort's scale, used for the synthetic slides


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def test_split_configs_match_tuned():
    """core.yaml + steatosis.yaml must be the same numbers as tuned.yaml.

    The split is a file-layout change, not a retune. If these ever diverge, a
    run reproduced from the split files stops matching the published one.
    """
    split = MashConfig.from_yaml(
        [ROOT / "configs/core.yaml", ROOT / "configs/steatosis.yaml"]
    )
    tuned = MashConfig.from_yaml(ROOT / "configs/tuned.yaml")
    for section in ("tissue", "tiling", "fat"):
        a, b = getattr(split, section), getattr(tuned, section)
        assert a == b, f"{section} differs:\n  split={a}\n  tuned={b}"


def test_legacy_configs_still_load():
    for name in ("default.yaml", "tuned.yaml"):
        cfg = MashConfig.from_yaml(ROOT / "configs" / name)
        assert cfg.fat.min_area_um2 > 0
        assert cfg.tissue.level >= 0


def test_unknown_key_is_rejected():
    """A typo must fail loudly rather than being silently ignored -- a config
    key that does nothing is how a filter quietly reverts to its default."""
    try:
        MashConfig.from_dict({"fat": {"circulariy_min": 0.6}})
    except KeyError as exc:
        assert "circulariy_min" in str(exc)
    else:
        raise AssertionError("typo in a config key was accepted")


def test_dotted_overrides():
    cfg = MashConfig()
    cfg.apply_overrides({"tissue.level": 1, "fat.white_threshold": 222,
                         "slide.mpp_x": None})
    assert cfg.tissue.level == 1
    assert cfg.fat.white_threshold == 222
    assert cfg.slide.mpp_x is None  # None means "not specified", not "set to None"


# --------------------------------------------------------------------------
# physical scale
# --------------------------------------------------------------------------


def _write_tiff(path: Path, size: int = 640, mpp: float | None = MPP) -> Path:
    import tifffile

    rng = np.random.default_rng(0)
    img = rng.integers(180, 250, size=(size, size, 3), dtype=np.uint8)
    kwargs = {}
    if mpp is not None:
        px_per_cm = 10_000.0 / mpp  # 10,000 um in a centimetre
        kwargs = {"resolution": (px_per_cm, px_per_cm), "resolutionunit": "CENTIMETER"}
    tifffile.imwrite(str(path), img, photometric="rgb", **kwargs)
    return path


def test_tiff_with_embedded_scale():
    """A TIFF carrying resolution tags is opened and its own scale is used."""
    with tempfile.TemporaryDirectory() as td:
        p = _write_tiff(Path(td) / "scaled.tif")
        with Slide(p) as s:
            assert s.mpp_source == "embedded", s.mpp_source
            assert abs(s.mpp_x - MPP) < 1e-3, s.mpp_x
            assert s.dimensions == (640, 640)
            tile = s.read_region((0, 0), 0, (64, 64))
            assert tile.shape == (64, 64, 3) and tile.dtype == np.uint8


def test_tiff_without_scale_fails_loudly():
    """No embedded scale and none configured -> refuse, and say which keys to
    set. Guessing would silently rescale every micron threshold."""
    with tempfile.TemporaryDirectory() as td:
        p = _write_tiff(Path(td) / "unscaled.tif", mpp=None)
        try:
            Slide(p)
        except MissingScaleError as exc:
            msg = str(exc)
            assert "mpp_x" in msg and "mpp_y" in msg, msg
        else:
            raise AssertionError("a slide with no physical scale was accepted")


def test_tiff_scale_from_config():
    with tempfile.TemporaryDirectory() as td:
        p = _write_tiff(Path(td) / "unscaled.tif", mpp=None)
        with Slide(p, SlideConfig(mpp_x=MPP, mpp_y=MPP)) as s:
            assert s.mpp_source == "config"
            assert abs(s.um2_per_pixel(0) - MPP**2) < 1e-9


def test_tiled_tiff_without_scale_is_refused():
    """The realistic case: a scanner TIFF openslide CAN open but that carries
    no microns-per-pixel. It must be refused, then accepted once configured."""
    import tifffile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "tiled_noscale.tif"
        rng = np.random.default_rng(0)
        img = rng.integers(180, 250, size=(1024, 1024, 3), dtype=np.uint8)
        tifffile.imwrite(str(p), img, photometric="rgb", tile=(256, 256))

        try:
            Slide(p)
        except MissingScaleError:
            pass
        else:
            raise AssertionError("a scanner TIFF with no scale was accepted")

        with Slide(p, SlideConfig(mpp_x=MPP, mpp_y=MPP)) as s:
            assert s.mpp_source == "config"


def test_pyramidal_tiff_levels():
    """A multi-level TIFF exposes its pyramid, and a region read at a
    non-zero level returns the right pixels at the right scale."""
    import tifffile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pyramid.tif"
        rng = np.random.default_rng(0)
        img = rng.integers(180, 250, size=(1024, 1024, 3), dtype=np.uint8)
        px_per_cm = 10_000.0 / MPP
        with tifffile.TiffWriter(str(p)) as tw:
            tw.write(img, photometric="rgb", subifds=2,
                     resolution=(px_per_cm, px_per_cm), resolutionunit="CENTIMETER")
            tw.write(img[::2, ::2], photometric="rgb", subfiletype=1)
            tw.write(img[::4, ::4], photometric="rgb", subfiletype=1)

        with Slide(p) as s:
            assert s.level_count == 3, s.level_count
            assert s.level_dimensions == ((1024, 1024), (512, 512), (256, 256))
            assert s.level_downsamples == (1.0, 2.0, 4.0)
            # Level-0 coordinates, level-1 pixels -- openslide's convention.
            got = s.read_region((512, 512), 1, (64, 64))
            assert np.array_equal(got, img[::2, ::2][256:320, 256:320])
            mx, _ = s.mpp_at_level(2)
            assert abs(mx - MPP * 4) < 1e-6


def test_micron_filters_survive_a_level_change():
    """The same micron threshold must mean the same physical size at every
    pyramid level -- that is the whole reason filters are specified in microns.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _write_tiff(Path(td) / "scaled.tif")
        with Slide(p) as s:
            for level in range(s.level_count):
                px = s.um_to_px(50.0, level)
                mx, my = s.mpp_at_level(level)
                microns_back = px * (mx + my) / 2.0
                assert abs(microns_back - 50.0) < max(mx, my), (
                    f"level {level}: 50 um round-tripped to {microns_back:.2f} um"
                )


def test_detector_areas_are_physical():
    """A droplet of known diameter must be measured at its true area in um^2,
    which is what makes the tuned min/max_area_um2 cutoffs meaningful."""
    tile = np.full((512, 512, 3), 120, dtype=np.uint8)  # dark "tissue"
    import cv2

    diameter_um = 30.0
    radius_px = int(round(diameter_um / 2 / MPP))
    cv2.circle(tile, (256, 256), radius_px, (255, 255, 255), -1)

    cfg = FatConfig(method="fixed", white_threshold=200,
                    min_area_um2=20.0, max_area_um2=2000.0)
    res = detect_fat(tile, np.ones((512, 512), dtype=bool), cfg, MPP * MPP)

    assert res.count(MACRO) == 1, f"expected one droplet, got {res.count(MACRO)}"
    expected = np.pi * (diameter_um / 2) ** 2
    got = res.area_um2(MACRO)
    assert abs(got - expected) / expected < 0.05, f"{got:.1f} vs {expected:.1f} um^2"


# --------------------------------------------------------------------------
# review scaffolding
# --------------------------------------------------------------------------


def test_review_package_roundtrip():
    """The candidate -> review -> confirmed-set path, end to end.

    Ballooning and inflammation are stubs, so this uses a stand-in detector's
    output. What it proves is that everything AROUND the detector works: crops
    at a fixed physical size, a manifest, and a confirmed set that excludes
    both rejections and unreviewed rows.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = _write_tiff(td / "slide.tif", size=1024)
        outputs = SlideOutputs.for_slide(td / "out", "slide")
        cands = [
            Candidate("slide", "ballooning", x=200 + i * 60, y=300, width=40,
                      height=40, score=i / 10, measurements={"area_ratio": 2.0 + i})
            for i in range(10)
        ]
        cfg = ReviewConfig(context_um=100.0, max_per_slide=6, strata=3)

        with Slide(p) as s:
            manifest = export_candidates(s, cands, outputs, "ballooning", cfg)
            context_px = s.um_to_px(cfg.context_um, 0)

        assert manifest.exists()
        assert (manifest.parent / "REVIEW.md").exists()

        import csv

        rows = list(csv.DictReader(open(manifest)))
        assert len(rows) == 6, f"expected the 6-candidate cap, got {len(rows)}"
        assert all(r["verdict"] == "" for r in rows), "verdict must start empty"

        img = manifest.parent / rows[0]["image"]
        assert img.exists()
        import cv2

        crop = cv2.imread(str(img))
        assert crop.shape[0] == context_px, (
            f"crop is {crop.shape[0]}px, expected {context_px}px "
            f"for {cfg.context_um} um at {MPP} um/px"
        )

        # A pathologist fills in some verdicts and leaves the rest blank.
        rows[0]["verdict"] = "y"
        rows[1]["verdict"] = "yes"
        rows[2]["verdict"] = "n"
        with open(manifest, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        confirmed = load_verdicts(manifest)
        assert len(confirmed) == 2, f"expected 2 confirmed, got {len(confirmed)}"
        assert confirmed[0].measurements["area_ratio"] > 0, "measurements lost"
        assert confirmed[0].feature == "ballooning"


def test_outputs_layout():
    o = SlideOutputs.for_slide("outputs", "R25-264-1")
    assert o.tiles_dir.as_posix().endswith("outputs/R25-264-1/tiles")
    assert o.feature_dir("steatosis").as_posix().endswith("R25-264-1/steatosis")
    assert o.review_dir("ballooning").as_posix().endswith("ballooning/review")
    try:
        o.feature_dir("fibrosis")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown feature name was accepted")


# --------------------------------------------------------------------------


def test_manifest_contract_is_shared_by_both_writers():
    """Both candidate features must write a manifest the review app can read.

    The regression this guards: ballooning and the generic exporter each had
    their own manifest writer, with column sets that had already diverged
    (`padding_um`/`review_band`/`rank` vs `context_um`). Nothing caught it
    because `Candidate.from_row` reads by name and ignores extras -- so the
    formats could drift until one dropped a column the app needs.
    """
    from mashpath.review import manifest as M
    from mashpath.review.candidates import Candidate

    c = Candidate(slide_id="S1", feature="ballooning", x=10, y=20,
                  width=5, height=6, score=1.5, measurements={"z_area": 2.0})

    fixed = M.row(c, "images/a.png", "overlays/a.png", 150.0, M.CROP_FIXED)
    padded = M.row(c, "images/a.png", "overlays/a.png", 114.9, M.CROP_PADDED,
                   extra={"review_band": "top", "rank": 3})

    for r in (fixed, padded):
        assert not [k for k in M.REQUIRED if k not in r]

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "manifest.csv"
        M.write(p, [fixed, padded])
        back = M.read(p)
        assert len(back) == 2
        assert back[0]["crop_mode"] == M.CROP_FIXED
        assert back[1]["crop_mode"] == M.CROP_PADDED
        # Feature-specific extras survive without becoming required.
        assert back[1]["review_band"] == "top"
        assert M.measurement_columns(back) == ["z_area"]
        assert [c.x for c in M.candidates(p)] == [10, 10]

        # A manifest missing a required column fails at read, with the reason.
        bad = Path(td) / "bad.csv"
        bad.write_text("candidate_id,slide,feature\nx,S1,ballooning\n")
        try:
            M.read(bad)
        except M.ManifestError as exc:
            assert "crop_um" in str(exc)
        else:
            raise AssertionError("a manifest missing crop_um must not load")


def test_verdicts_support_two_reviewers_and_repeats():
    """The verdict store must record what a single manifest column cannot."""
    from mashpath.review import verdicts as V

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "verdicts.csv"
        for cid, a, b in [("c1", "y", "y"), ("c2", "n", "n"),
                          ("c3", "y", "n"), ("c4", "n", "n")]:
            V.append(p, "AB", "S1", "ballooning", cid, a)
            V.append(p, "BB", "S1", "ballooning", cid, b)
        V.append(p, "AB", "S1", "ballooning", "c1", "y")  # deliberate repeat

        rows = V.load(p)
        assert V.reviewers(rows) == ["AB", "BB"]

        inter = V.cohen_kappa(V.latest(rows, "AB"), V.latest(rows, "BB"))
        assert inter["n_shared"] == 4
        assert inter["agreement"] == 0.75

        intra = V.intra_rater(rows, "AB")
        assert intra["n_repeated"] == 1
        # A repeat must not overwrite the first look.
        assert len([r for r in rows if r["candidate_id"] == "c1"
                    and r["reviewer"] == "AB"]) == 2
        assert V.intra_rater(rows, "BB")["n_repeated"] == 0


def test_every_subcommand_can_print_its_help():
    """argparse expands help text with `help % params`, so a bare `%` in a help
    string raises TypeError at print time and nowhere else. `--help` had been
    broken for every command by the word "100% uniform" while every real
    invocation still worked."""
    import argparse

    from mashpath.cli import build_parser

    parser = build_parser()
    parser.format_help()
    subs = [a for a in parser._actions
            if isinstance(a, argparse._SubParsersAction)]
    assert subs, "no subcommands found"
    for name, sub in subs[0].choices.items():
        try:
            sub.format_help()
        except Exception as exc:
            raise AssertionError(f"`{name} --help` raises "
                                 f"{type(exc).__name__}: {exc}") from None


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
