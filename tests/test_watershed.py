"""Splitting touching fat droplets.

Runs under plain python (no pytest needed):

    python tests/test_watershed.py

Synthetic discs, because the property under test is geometric and a real tile
cannot say what the right answer was. The tests that matter are the ones about
what splitting must NOT do: it must not invent droplets in a single scalloped
one, it must not change the total fat area, and it must not touch anything when
it is switched off -- the exported pseudo-label is a set of boundaries, and all
three failures are silent in the fat fraction, which is the number anyone would
look at first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skimage.measure import label  # noqa: E402

from mashpath.features.steatosis.config import FatConfig  # noqa: E402
from mashpath.features.steatosis.detect import (extract_components,  # noqa: E402
                                                split_touching)

MPP = 0.4953
UM2 = MPP * MPP
CFG = FatConfig(watershed=True, min_area_um2=40.0, watershed_min_depth_um=1.5)


def _disc(shape, cx, cy, r):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def test_two_fused_droplets_come_apart():
    img = _disc((60, 100), 40, 30, 10) | _disc((60, 100), 58, 30, 10)
    assert label(img).max() == 1, "the test shape is not fused"
    out, n = split_touching(img, CFG, UM2)
    assert out.max() == 2, f"expected 2 droplets, got {out.max()}"
    assert n == 1, f"n_split should count the component, not the pieces: {n}"


def test_one_droplet_stays_one():
    out, n = split_touching(_disc((60, 100), 50, 30, 12), CFG, UM2)
    assert out.max() == 1, f"a single disc was split into {out.max()}"
    assert n == 0


def test_a_scalloped_droplet_is_not_three_droplets():
    """The over-splitting guard. A real droplet's border is ragged, its
    distance transform carries a ridge of shallow maxima, and a seeding rule
    keyed on separation rather than depth turns one droplet into several. That
    failure raises the count, lowers the mean size, and leaves the fat FRACTION
    almost unchanged -- so nothing downstream would catch it."""
    img = _disc((80, 80), 40, 40, 16)
    rng = np.random.default_rng(0)
    for _ in range(14):                      # bumps welded onto the rim
        a = rng.uniform(0, 2 * np.pi)
        img |= _disc((80, 80), 40 + 16 * np.cos(a), 40 + 16 * np.sin(a), 3)
    assert label(img).max() == 1
    out, n = split_touching(img, CFG, UM2)
    assert out.max() == 1, f"a scalloped single droplet became {out.max()} droplets"
    assert n == 0


def test_splitting_conserves_area():
    """No watershed lines: every input pixel keeps a label. Area is the
    reported quantity and count is the diagnostic; if splitting could delete
    boundary pixels the fat fraction would move and look biological."""
    img = _disc((60, 120), 40, 30, 11) | _disc((60, 120), 60, 30, 11) \
        | _disc((60, 120), 95, 30, 8)
    out, _ = split_touching(img, CFG, UM2)
    assert int((out > 0).sum()) == int(img.sum()), \
        f"{img.sum() - (out > 0).sum()} pixels lost to watershed lines"


def test_a_component_below_the_floor_is_never_offered_to_the_watershed():
    """One droplet's worth of area cannot be two droplets."""
    cfg = FatConfig(watershed=True, min_area_um2=40.0, watershed_min_area_um2=1e6)
    img = _disc((60, 100), 40, 30, 10) | _disc((60, 100), 58, 30, 10)
    out, n = split_touching(img, cfg, UM2)
    assert out.max() == 1 and n == 0, \
        "a component under the size floor was split"


def test_the_depth_threshold_is_in_microns_not_pixels():
    """Same shape, two magnifications: the same split, or the parameter does
    not mean what the config says it means."""
    fine = _disc((120, 200), 80, 60, 20) | _disc((120, 200), 116, 60, 20)
    coarse = _disc((60, 100), 40, 30, 10) | _disc((60, 100), 58, 30, 10)
    a, _ = split_touching(fine, CFG, (MPP / 2) ** 2)
    b, _ = split_touching(coarse, CFG, UM2)
    assert a.max() == b.max() == 2, f"{a.max()} at 2x vs {b.max()} at 1x"


def test_off_is_byte_identical_to_plain_labelling():
    """The default path must not move. `regression_steatosis.py` checks this
    end to end; this checks it at the point the branch is taken."""
    img = _disc((60, 100), 40, 30, 10) | _disc((60, 100), 58, 30, 10)
    tissue = np.ones_like(img)
    rgb = (np.where(img[..., None], 250, 120) * np.ones(3)).astype(np.uint8)
    # Morphology off, so `labels` is a labelling of exactly `img` and the
    # comparison is about the branch rather than about the opening radius.
    plain = dict(open_radius_px=0, close_radius_px=0, fill_holes=False,
                 min_area_um2=40.0)
    off = extract_components(rgb, tissue, FatConfig(watershed=False, **plain), UM2)
    assert np.array_equal(off.labels, label(img)), "the off path changed"
    assert off.n_split == 0
    on = extract_components(rgb, tissue,
                            FatConfig(watershed=True, watershed_min_depth_um=1.5,
                                      **plain), UM2)
    assert on.n_split == 1 and on.labels.max() == 2, \
        f"the on path did not split: {on.labels.max()} labels, {on.n_split} split"


def test_an_empty_tile_does_not_raise():
    out, n = split_touching(np.zeros((32, 32), bool), CFG, UM2)
    assert out.max() == 0 and n == 0


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
