"""The viewer's silent failure modes.

Runs under plain python (no pytest, no GPU, no slide):

    python tests/test_viewer.py

Everything here is a mistake that produces a working-looking viewer:

  geometry    the tissue pyramid and the overlay pyramid are two independent
              tile sources drawn on top of each other. If their level maths
              disagree by one pixel at the ragged right/bottom edge, the
              overlay slides against the tissue as you zoom -- which reads as
              "the model is a bit off", not as a viewer bug.
  scale       the overlay must be a DOWNSAMPLED MASK, never a low-resolution
              INFERENCE. The model has no scale augmentation and reports 0.06x
              the fat at 4x downsample, so a viewer that re-ran it per zoom
              level would show a different percentage at every zoom.
  denominator fat is a fraction OF TISSUE. Measuring it over the whole tile
              instead silently deflates every number, and glass is white so
              the error is systematic rather than noise.
  sparsity    a tile with no fat is never written. If a missing file raised
              instead of reading as zero, clean slides -- the negative
              controls -- would be the ones that failed.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mashpath.viewer.dzi import DeepZoom, MaskTileSource  # noqa: E402
from mashpath.viewer.precompute import _area_histogram, _zonal  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not cond:
        FAILS.append(name)


# ---- geometry -------------------------------------------------------------

def test_geometry() -> None:
    print("\ngeometry")
    W, H = 35856, 40087                      # a real slide from this cohort
    dz = DeepZoom(W, H)
    check("max level is full resolution",
          dz.level_dimensions(dz.max_level) == (W, H),
          str(dz.level_dimensions(dz.max_level)))
    check("each level halves the one above",
          dz.level_dimensions(dz.max_level - 1) == ((W + 1) // 2, (H + 1) // 2))
    check("level 0 is 1x1", dz.level_dimensions(0) == (1, 1))

    # Every tile of a level must tile that level exactly: no gap, no overlap.
    for level in (dz.max_level, dz.max_level - 2, dz.max_level - 5):
        cols, rows = dz.tile_counts(level)
        lw, lh = dz.level_dimensions(level)
        wsum = sum(dz.tile_box(level, c, 0)[4] for c in range(cols))
        hsum = sum(dz.tile_box(level, 0, r)[5] for r in range(rows))
        check(f"level {level} tiles cover it exactly",
              (wsum, hsum) == (lw, lh), f"{wsum}x{hsum} vs {lw}x{lh}")

    # The edge tile is the one that drifts if the maths is wrong.
    cols, rows = dz.tile_counts(dz.max_level)
    x0, y0, w0, h0, tw, th = dz.tile_box(dz.max_level, cols - 1, rows - 1)
    check("last full-res tile stops at the slide edge",
          x0 + w0 == W and y0 + h0 == H, f"{x0 + w0},{y0 + h0} vs {W},{H}")

    # The two pyramids are the same object, built from the same numbers. This
    # is the property that keeps the overlay registered to the tissue.
    mask = MaskTileSource(tempfile.mkdtemp(), W, H)
    check("overlay pyramid matches the tissue pyramid at every level",
          all(mask.dz.level_dimensions(l) == dz.level_dimensions(l)
              for l in range(dz.level_count)))
    check("overlay tile counts match at every level",
          all(mask.dz.tile_counts(l) == dz.tile_counts(l)
              for l in range(dz.level_count)))


def test_odd_sizes() -> None:
    """Slides whose dimensions are not powers of two -- i.e. all of them."""
    print("\ngeometry, awkward dimensions")
    for W, H in [(1, 1), (513, 1), (1000, 3), (33864, 25673), (4096, 4096)]:
        dz = DeepZoom(W, H)
        ok = dz.level_dimensions(dz.max_level) == (W, H)
        cols, rows = dz.tile_counts(dz.max_level)
        x0, y0, w0, h0, _, _ = dz.tile_box(dz.max_level, cols - 1, rows - 1)
        ok = ok and (x0 + w0 == W) and (y0 + h0 == H)
        check(f"{W}x{H} round-trips", ok)


# ---- the mask store -------------------------------------------------------

def test_mask_store() -> None:
    print("\nmask store")
    W = H = 2048
    d = Path(tempfile.mkdtemp())
    (d / "mask" / "l0").mkdir(parents=True)

    # One tile, half fat. Written on the 512 level-0 grid the model ran on.
    tile = np.zeros((512, 512), np.uint8)
    tile[:256, :] = 255
    cv2.imwrite(str(d / "mask" / "l0" / "000512_000512.png"), tile)
    ds4 = np.zeros((H // 4, W // 4), np.uint8)
    ds4[128:160, 128:256] = 255
    cv2.imwrite(str(d / "mask" / "ds4.png"), ds4)

    src = MaskTileSource(d, W, H)
    full = src._read_full(512, 512, 512, 512)
    check("full-resolution read returns the stored tile",
          np.array_equal(full, tile), f"{100 * (full > 127).mean():.1f}% set")

    # The sparsity contract: an unwritten tile is zero, not an error.
    empty = src._read_full(0, 0, 512, 512)
    check("a tile that was never written reads as zero",
          empty.shape == (512, 512) and not empty.any())

    # A region straddling written and unwritten tiles.
    straddle = src._read_full(256, 512, 512, 256)
    check("a straddling read pastes only what exists",
          straddle.shape == (256, 512) and straddle[:, :256].sum() == 0
          and straddle[:, 256:].all())

    # Tile geometry survives the round trip at every level, including the
    # partial tiles at the edges.
    for level in range(src.dz.level_count):
        cols, rows = src.dz.tile_counts(level)
        _, _, _, _, tw, th = src.dz.tile_box(level, cols - 1, rows - 1)
        g = src.tile_gray(level, cols - 1, rows - 1)
        if g.shape != (th, tw):
            check(f"level {level} edge tile is the right size", False,
                  f"{g.shape} vs {(th, tw)}")
            break
    else:
        check("every level's edge tile is the right size", True)

    rgba = src.tile_rgba(src.dz.max_level, 1, 1)
    check("overlay tile carries the mask in alpha",
          rgba.shape[2] == 4 and np.array_equal(rgba[..., 3], tile))
    check("overlay tile is transparent where there is no fat",
          rgba[300, 0, 3] == 0)


def test_non_default_mask_grid() -> None:
    """A resampled slide is tiled on its own spacing, not on 512.

    `precompute` records that spacing as `mask_tile`; if the reader assumed
    512 it would look for mask tiles at coordinates that were never written,
    and the overlay would come back empty on exactly the slides that needed
    resampling -- the ones from a different scanner.
    """
    print("\nresampled mask grid")
    W = H = 2048
    d = Path(tempfile.mkdtemp())
    (d / "mask" / "l0").mkdir(parents=True)
    T = 256                                   # e.g. a finer scanner
    tile = np.full((T, T), 255, np.uint8)
    cv2.imwrite(str(d / "mask" / "l0" / f"{T:06d}_{T:06d}.png"), tile)

    right = MaskTileSource(d, W, H, mask_tile=T)
    got = right._read_full(T, T, T, T)
    check("reader honours a non-512 mask grid", bool(got.all()),
          f"{100 * (got > 127).mean():.0f}% set")

    wrong = MaskTileSource(d, W, H, mask_tile=512)
    check("assuming 512 would have missed it",
          not wrong._read_full(T, T, T, T).any())


def test_downsample_is_of_the_mask() -> None:
    """Zooming out must average the MASK, not re-run anything.

    A quarter-covered region has to read as a quarter-intensity overlay pixel.
    If it were thresholded instead, a zoomed-out clean slide and a zoomed-out
    lightly-fatty slide would look identical, which is exactly the comparison
    the demo exists to make.
    """
    print("\ndownsampling")
    W = H = 2048
    d = Path(tempfile.mkdtemp())
    (d / "mask" / "l0").mkdir(parents=True)
    ds4 = np.zeros((H // 4, W // 4), np.uint8)
    ds4[:, :] = 64                       # a uniform quarter coverage
    cv2.imwrite(str(d / "mask" / "ds4.png"), ds4)
    src = MaskTileSource(d, W, H)
    # Pick a level coarse enough to be served from ds4.
    level = src.dz.max_level - 3          # scale 8
    g = src.tile_gray(level, 0, 0)
    check("partial coverage survives downsampling as partial alpha",
          0 < int(g.mean()) < 200, f"mean alpha {g.mean():.0f}")


# ---- the numbers ----------------------------------------------------------

def test_zonal_and_histogram() -> None:
    print("\nsummaries")
    import pandas as pd
    # Two regions: one clean, one fatty. The slide mean hides that; the map
    # is the thing that does not.
    rows = []
    for i in range(10):
        rows.append({"tile_x": i * 512, "tile_y": 0, "fat_px": 0,
                     "tissue_px": 1000, "fat_fraction": 0.0})
    for i in range(10):
        rows.append({"tile_x": i * 512, "tile_y": 20000, "fat_px": 200,
                     "tissue_px": 1000, "fat_fraction": 0.2})
    df = pd.DataFrame(rows)
    z = _zonal(df, 40000, 40000, 8)
    flat = [v for r in z["grid"] for v in r if v is not None]
    check("zonal map separates a clean region from a fatty one",
          min(flat) == 0.0 and abs(max(flat) - 20.0) < 1e-6,
          f"min {min(flat)} max {max(flat)}")
    check("tile spread reports the split, not just the mean",
          z["tile_fat_percent"]["p10"] < 1 and z["tile_fat_percent"]["p90"] > 19)

    h = _area_histogram(np.array([5.0, 6.0, 50.0, 900.0, 1000.0]))
    check("droplet histogram bins are log-spaced",
          len(h["edges"]) == len(h["counts"]) + 1 and sum(h["counts"]) > 0)
    check("empty droplet set does not crash the histogram",
          _area_histogram(np.array([]))["counts"] == [])


def test_unet_padding() -> None:
    """A free-form region must reach the model on a multiple of 16.

    The U-Net pools 4 times, so an odd height loses a row on the way down and
    the skip concatenation fails:

        ConcatOp : Dimension 1 in both shapes must be equal:
        shape[0] = [1,136,202,256] vs. shape[1] = [1,137,202,256]

    The census feeds exactly 512 px and never sees this; the live spot check
    feeds whatever size the browser viewport is and sees it almost always.
    Geometry first, then the real model, because only the second would have
    caught the bug.
    """
    print("\nU-Net input padding")
    from mashpath.viewer.precompute import POOL_MULTIPLE, pad_for_unet

    ok = True
    for h, w in [(2175, 2158), (513, 1), (1, 1), (512, 512), (137, 202), (2048, 2048)]:
        img = np.zeros((h, w, 3), np.uint8)
        out, oh, ow = pad_for_unet(img)
        if (out.shape[0] % POOL_MULTIPLE or out.shape[1] % POOL_MULTIPLE
                or (oh, ow) != (h, w) or out.shape[0] < h or out.shape[1] < w):
            ok = False
            print(f"     {h}x{w} -> {out.shape[:2]}  WRONG")
    check("every size pads up to a multiple of 16 and reports its original", ok)
    check("an already-aligned image is returned untouched",
          pad_for_unet(np.zeros((512, 512, 3), np.uint8))[0].shape[:2] == (512, 512))

    try:
        import os
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        from mashpath.train.unet import UNetConfig, build_unet
    except Exception as e:
        print(f"  SKIP  real-model check ({type(e).__name__})")
        return

    model = build_unet(UNetConfig(base_filters=4, depth=4))
    bad = np.zeros((1, 137, 202, 3), np.float32)
    try:
        model.predict_on_batch(bad)
        check("an unpadded odd size really does fail", False, "it did not raise")
    except Exception:
        check("an unpadded odd size really does fail", True)

    padded, ih, iw = pad_for_unet(np.zeros((137, 202, 3), np.uint8))
    try:
        p = model.predict_on_batch(padded.astype(np.float32)[None] / 255.0)
        cropped = np.asarray(p[0, ..., 0])[:ih, :iw]
        check("the padded version runs and crops back to the request",
              cropped.shape == (137, 202), str(cropped.shape))
    except Exception as e:
        check("the padded version runs and crops back to the request", False,
              f"{type(e).__name__}: {e}")


def test_meta_contract() -> None:
    """The fields the interface promises, and the one it must never carry.

    The whole argument for this project is a continuous measurement, so a
    0-3 grade or any binned severity category appearing in the artifact --
    where the viewer could pick it up and render it -- is a correctness bug,
    not a style preference.
    """
    print("\nmeta contract")
    import pandas as pd
    df = pd.DataFrame([{"tile_x": 0, "tile_y": 0, "fat_px": 50,
                        "tissue_px": 1000, "fat_fraction": 0.05}])
    z = _zonal(df, 1024, 1024, 4)
    banned = ("grade", "score_0_3", "severity", "stage", "category", "band_label")
    blob = json.dumps(z).lower()
    check("the summary carries no grade or binned severity",
          not any(b in blob for b in banned))
    check("fat is reported as a fraction of tissue, not of the tile",
          abs(z["tile_fat_percent"]["mean"] - 5.0) < 1e-6,
          f"{z['tile_fat_percent']['mean']}")


if __name__ == "__main__":
    test_geometry()
    test_odd_sizes()
    test_mask_store()
    test_non_default_mask_grid()
    test_downsample_is_of_the_mask()
    test_zonal_and_histogram()
    test_unet_padding()
    test_meta_contract()
    print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all checks passed'}")
    sys.exit(1 if FAILS else 0)
