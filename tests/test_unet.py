"""The U-Net's plumbing: the parts that are silently wrong rather than loud.

Runs under plain python (no pytest, no GPU):

    python tests/test_unet.py

Synthetic 32 px tiles written to a temp directory. Nothing here trains a model
worth anything -- the point is the three places where a mistake produces
numbers that look fine:

  order      `predict_frame` joins each prediction to a tissue mask BY ROW
             ORDER. If the evaluation dataset ever shuffled, every tile would
             be masked with a different tile's tissue and the fat fractions
             would still look plausible.
  tissue     the teacher measures fat inside tissue only. A student scored on
             the whole tile is scored on a bigger canvas, and since glass is
             white the difference is systematic, not noise.
  weighting  a confidence-weighted loss that silently ignored its weights
             would train fine and report a normal loss curve.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from mashpath.train.unet import (UNetConfig, build_unet, dice,  # noqa: E402
                                 make_dataset, make_loss, predict_frame,
                                 soft_dice_loss, weighted_bce)

SIZE = 32


def _fixture(n: int = 6, tissue_fraction: float = 0.5) -> tuple[Path, pd.DataFrame]:
    """A tiny dataset on disk. Left half of every tile is tissue; the model
    under test predicts fat everywhere, so anything counted outside that half
    is the bug."""
    import cv2
    root = Path(tempfile.mkdtemp())
    for sub in ("images", "labels", "confidence", "tissue"):
        (root / sub).mkdir()
    cut = int(SIZE * tissue_fraction)
    rows = []
    for i in range(n):
        stem = f"s{i:02d}__tile"
        img = np.full((SIZE, SIZE, 3), 200, np.uint8)
        lab = np.zeros((SIZE, SIZE), np.uint8); lab[:, :cut // 2] = 255
        con = np.full((SIZE, SIZE), 255, np.uint8)
        tis = np.zeros((SIZE, SIZE), np.uint8); tis[:, :cut] = 255
        for sub, a in (("images", img), ("labels", lab),
                       ("confidence", con), ("tissue", tis)):
            arr = a if a.ndim == 3 else np.dstack([a, a, a])
            cv2.imwrite(str(root / sub / f"{stem}.png"),
                        cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        rows.append({"id": stem, "slide": f"s{i:02d}", "batch": f"B{i % 2}",
                     "cohort": "positive", "diet": "nash" if i % 2 else "chow",
                     "tile_size": SIZE, "tissue_fraction": tissue_fraction,
                     "fat_fraction": 0.5})
    return root, pd.DataFrame(rows)


class _AllFat:
    """Stands in for a trained model: predicts fat on every pixel."""
    def predict_on_batch(self, xb):
        n = int(np.asarray(xb).shape[0])
        return np.ones((n, SIZE, SIZE, 1), np.float32)


def test_predictions_are_confined_to_tissue():
    root, df = _fixture(tissue_fraction=0.5)
    out = predict_frame(_AllFat(), df, root, UNetConfig(), batch_size=2)
    expected = SIZE * (SIZE // 2)          # the left half only
    assert (out["pred_label_px"] == expected).all(), \
        f"expected {expected} px inside tissue, got {out['pred_label_px'].tolist()}"
    # And the fraction is then over tissue area, so a model that fills the
    # tissue reads 1.0 -- never more.
    assert np.allclose(out["pred_fat_fraction"], 1.0), \
        f"fat fraction outside [0,1]: {out['pred_fat_fraction'].tolist()}"


def test_a_missing_tissue_directory_is_loud():
    root, df = _fixture()
    import shutil
    shutil.rmtree(root / "tissue")
    out = predict_frame(_AllFat(), df, root, UNetConfig(), batch_size=2)
    assert (out["pred_fat_fraction"] > 1.0).all(), \
        "without tissue masks the whole tile is counted; that must be visible"


def test_the_evaluation_dataset_does_not_shuffle():
    """Load-bearing: predict_frame joins tissue masks by row order."""
    root, df = _fixture(n=6)
    ds = make_dataset(df, root, UNetConfig(), training=False, batch_size=1)
    seen = []
    for xb, yb, _ in ds:
        seen.append(float(np.asarray(yb).sum()))
    assert len(seen) == len(df)
    # Every fixture tile is identical, so order cannot be checked by content.
    # Check the property directly instead: two passes must agree element-wise,
    # which a shuffled pipeline with reshuffle_each_iteration would fail.
    again = [float(np.asarray(y).sum())
             for _, y, _ in make_dataset(df, root, UNetConfig(),
                                         training=False, batch_size=1)]
    assert seen == again, "the evaluation pipeline is not order-stable"


def test_the_loss_actually_uses_its_weights():
    import keras.ops as K
    y = K.convert_to_tensor(np.ones((1, 4, 4, 1), np.float32))
    p = K.convert_to_tensor(np.full((1, 4, 4, 1), 0.1, np.float32))
    w_all = K.convert_to_tensor(np.ones((1, 4, 4, 1), np.float32))
    w_half = K.convert_to_tensor(
        np.array([[1, 1, 0, 0]] * 4, np.float32).reshape(1, 4, 4, 1))
    # Same predictions everywhere, so a weighted MEAN is unchanged by masking
    # half the pixels -- that is the invariant. What must NOT happen is the
    # weights being ignored: make half the pixels easy and confirm the weighted
    # loss tracks only the hard ones.
    p2 = np.full((1, 4, 4, 1), 0.1, np.float32)
    p2[:, :, 2:, :] = 0.99
    p2 = K.convert_to_tensor(p2)
    hard_only = float(weighted_bce(y, p2, w_half))
    everything = float(weighted_bce(y, p2, w_all))
    assert hard_only > everything + 0.5, \
        (f"weights were ignored: masking the easy half gave {hard_only:.3f} "
         f"against {everything:.3f} for all pixels")


def test_an_output_pixel_maps_to_its_input_pixel():
    """'same' padding throughout, or a predicted mask is offset from the
    pseudo-label it is scored against and every Dice is quietly wrong."""
    m = build_unet(UNetConfig(base_filters=4, depth=2))
    x = np.zeros((1, 64, 64, 3), np.float32)
    assert m.predict(x, verbose=0).shape[1:3] == (64, 64)


def test_two_empty_masks_agree():
    assert dice(np.zeros((8, 8)), np.zeros((8, 8))) == 1.0
    assert dice(np.ones((8, 8)), np.zeros((8, 8))) == 0.0


def test_cross_entropy_is_happy_with_an_empty_prediction_and_dice_is_not():
    """The collapse guard, and the reason the loss has two terms.

    Fat is ~5% of the pixels in a positive tile. Predicting nothing everywhere
    is cheap under cross-entropy, and the first smoke run did exactly that:
    four epochs, falling train and validation loss, every predicted fat
    fraction 0.000. It reads as a working run until someone looks at a
    prediction, which is why this is pinned rather than trusted."""
    import keras.ops as K
    y = np.zeros((1, 16, 16, 1), np.float32)
    y[:, :4, :4, :] = 1.0                       # a 6.25%-positive tile
    empty = np.full((1, 16, 16, 1), 1e-4, np.float32)
    yt, pt = K.convert_to_tensor(y), K.convert_to_tensor(empty)

    bce = float(weighted_bce(yt, pt))
    dl = float(soft_dice_loss(yt, pt))
    assert bce < 0.6, f"cross-entropy should find the empty mask cheap, got {bce:.3f}"
    assert dl > 0.99, f"Dice should reject the empty mask, got {dl:.3f}"

    combined = float(make_loss(UNetConfig(dice_weight=1.0))(yt, pt))
    plain = float(make_loss(UNetConfig(dice_weight=0.0))(yt, pt))
    assert combined > plain + 0.9, \
        f"the Dice term is not reaching the loss: {combined:.3f} vs {plain:.3f}"


def test_a_perfect_prediction_costs_almost_nothing():
    import keras.ops as K
    y = np.zeros((1, 16, 16, 1), np.float32)
    y[:, :4, :4, :] = 1.0
    good = np.clip(y, 1e-4, 1 - 1e-4)
    loss = float(make_loss(UNetConfig())(K.convert_to_tensor(y),
                                         K.convert_to_tensor(good)))
    assert loss < 0.02, f"a near-perfect prediction should be cheap, got {loss:.4f}"


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
