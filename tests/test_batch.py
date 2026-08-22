"""Batch-effect infrastructure: splitting, evaluation, normalization, augmentation.

Runs under plain python (no pytest needed):

    python tests/test_batch.py

There are no labels yet, so everything here is against SYNTHETIC data. That is
enough for what these tests are actually for: proving that the guards fire. The
centrepiece is `test_a_batch_reading_cheat_survives_a_tile_split_and_dies_here`,
which builds a frame where the label is confounded with the staining batch --
exactly the situation on disk -- and shows a feature that knows nothing but the
batch scoring AUC ~1.0 under a tile-level split and ~0.5 under this module's.
If that test ever passes trivially, the protection has been lost.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mashpath.config import MashConfig  # noqa: E402
from mashpath.train import (augmenter_from_config, normalizer_from_config)  # noqa: E402
from mashpath.train.augment import ColourJitter, StainJitter, default_pipeline  # noqa: E402
from mashpath.train.evaluate import (auc, batch_separability, binary_metrics,  # noqa: E402
                                     held_out_batch_eval, per_batch_metrics,
                                     report)
from mashpath.train.splits import (BatchSplit, LeakySplitError,  # noqa: E402
                                   assert_no_leakage, leave_one_batch_out,
                                   split_by_batch)
from mashpath.train.stain import (MacenkoNormalizer, NullNormalizer,  # noqa: E402
                                  ReinhardNormalizer, macenko_stain_matrix,
                                  make_normalizer, od_to_rgb, rgb_to_od)


def synthetic_frame(n_batches: int = 4, slides_per_batch: int = 3,
                    tiles_per_slide: int = 40, seed: int = 0) -> pd.DataFrame:
    """A frame shaped like the real one: batch -> slides -> tiles.

    `cheat` is a pure batch signature -- what a model reading stain would see.
    `signal` is weak, real biology, independent of batch.
    `label` is confounded: mostly batch, a little signal.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_batches):
        batch = f"batch{b}"
        # Prevalence rises with batch index, so `cheat` (the index plus
        # noise) is a monotone predictor of the label and nothing else.
        positive_batch = b >= n_batches // 2
        for s in range(slides_per_batch):
            slide = f"{batch}_slide{s}"
            for t in range(tiles_per_slide):
                signal = rng.normal()
                p = 0.75 if positive_batch else 0.25
                label = int(rng.random() < (p if signal > -0.5 else p * 0.5))
                rows.append({"batch": batch, "slide": slide,
                             "tile_id": f"{slide}_{t}",
                             "cheat": b + rng.normal(0, 0.01),
                             "signal": signal, "label": label})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------

def test_split_is_defined_by_batch_names_and_assigns_every_row():
    f = synthetic_frame()
    sp = split_by_batch(f, validation=["batch1"])
    fold = sp.assign(f)
    assert set(fold.unique()) == {"train", "validation"}
    assert (fold[f.batch == "batch1"] == "validation").all()
    assert (fold[f.batch != "batch1"] == "train").all()
    assert len(fold) == len(f)


def test_a_tile_level_split_is_not_expressible():
    """The point of the module. There must be no way to ask for a row split."""
    import inspect

    from mashpath.train import splits as m

    banned = ("test_size", "train_size", "random_state", "shuffle", "n_splits",
              "frac", "indices", "test_fraction", "seed")
    for name, fn in vars(m).items():
        if not callable(fn) or getattr(fn, "__module__", "") != m.__name__:
            continue
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        bad = [p for p in params if p in banned]
        assert not bad, f"{name} accepts {bad}, which permits a row-level split"
    # And a split cannot be built out of row indices.
    assert not hasattr(m, "split_by_index")
    assert not hasattr(m, "train_test_split")


def test_a_batch_on_both_sides_raises_when_the_split_is_made():
    f = synthetic_frame()
    try:
        BatchSplit({"train": ("batch0", "batch1"), "validation": ("batch1",)})
    except LeakySplitError:
        pass
    else:
        raise AssertionError("a batch in two folds was accepted")


def test_a_slide_on_both_sides_raises_too():
    """Slide-level leakage is reported even though the split is by batch.

    A corrupt batch column is the realistic way this happens, and it makes both
    levels leak at once -- so the error has to name both, or whoever reads it
    goes looking for the wrong cause.
    """
    f = synthetic_frame(n_batches=2, slides_per_batch=2, tiles_per_slide=10)
    fold = pd.Series(["train"] * len(f), index=f.index)
    fold[f["batch"] == "batch1"] = "validation"
    # Three tiles of one validation slide handed back to train.
    stray = f.index[f["slide"] == "batch1_slide0"][:3]
    fold[stray] = "train"
    try:
        assert_no_leakage(f, fold)
    except LeakySplitError as exc:
        assert "slide" in str(exc), str(exc)
        assert "batch" in str(exc), str(exc)
    else:
        raise AssertionError("a slide in two folds was accepted")


def test_one_batch_cannot_be_split_at_all():
    f = synthetic_frame(n_batches=1)
    try:
        split_by_batch(f, validation=["batch0"])
    except ValueError as exc:
        assert "at least two" in str(exc)
    else:
        raise AssertionError("a one-batch frame produced a split")


def test_a_batch_left_out_of_every_fold_raises():
    """Silently dropping a batch is how a cohort disappears from a result."""
    f = synthetic_frame(n_batches=3)
    sp = BatchSplit({"train": ("batch0",), "validation": ("batch1",)})
    try:
        sp.assign(f)
    except KeyError as exc:
        assert "batch2" in str(exc)
    else:
        raise AssertionError("an unassigned batch was accepted")


def test_a_frame_without_a_batch_column_refuses_rather_than_guessing():
    f = synthetic_frame().drop(columns=["batch"])
    try:
        split_by_batch(f, validation=["batch0"])
    except KeyError as exc:
        assert "batch" in str(exc)
    else:
        raise AssertionError("a frame with no batch column produced a split")


def test_leave_one_batch_out_covers_every_batch_exactly_once():
    f = synthetic_frame(n_batches=4)
    held = [b for b, _ in leave_one_batch_out(f)]
    assert sorted(held) == ["batch0", "batch1", "batch2", "batch3"]
    for b, sp in leave_one_batch_out(f):
        assert sp.folds["validation"] == (b,)
        assert b not in sp.folds["train"]


def test_summary_reports_prevalence_per_fold():
    f = synthetic_frame()
    sp = split_by_batch(f, validation=["batch1"])
    s = sp.summary(f, label_col="label")
    assert set(s.index) == {"train", "validation"}
    assert s.loc["validation", "batches"] == 1
    assert s["rows"].sum() == len(f)
    assert 0.0 <= s.loc["validation", "prevalence"] <= 1.0


# --------------------------------------------------------------------------
# the confound, demonstrated
# --------------------------------------------------------------------------

def test_a_batch_reading_cheat_survives_a_tile_split_and_dies_here():
    """The whole reason this module exists, as an executable statement.

    `cheat` carries no biology -- it is the batch index and nothing else. Under
    a random tile-level split it looks like an excellent predictor, because the
    same batches sit on both sides and the label is confounded with batch.
    Under a held-out-batch split it is worthless, which is the truth.
    """
    f = synthetic_frame(n_batches=4, seed=3)

    # What a tile-level split would report. Not built with anything in this
    # module -- it cannot be -- so it is spelled out here to be compared against.
    rng = np.random.default_rng(0)
    rows = rng.permutation(len(f))
    holdout = f.iloc[rows[: len(f) // 4]]
    leaky = auc(holdout["label"], holdout["cheat"])

    honest = held_out_batch_eval(
        f, lambda train, test: test["cheat"].to_numpy(), label_col="label")

    assert leaky > 0.70, f"the leaky split should look good, got {leaky:.3f}"
    assert honest["auc"].max() < 0.65, (
        f"a pure batch signature must not survive a batch split, "
        f"got {honest['auc'].to_dict()}")


def test_batch_separability_flags_an_output_that_encodes_batch():
    f = synthetic_frame(n_batches=3)
    sep_cheat = batch_separability(f["cheat"], f["batch"])
    sep_signal = batch_separability(f["signal"], f["batch"])
    assert sep_cheat.map(lambda v: abs(v - 0.5)).max() > 0.4, sep_cheat.to_dict()
    assert sep_signal.map(lambda v: abs(v - 0.5)).max() < 0.15, sep_signal.to_dict()


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def test_auc_matches_hand_computed_values_and_handles_ties():
    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 1.0
    assert auc([0, 0, 1, 1], [0.4, 0.3, 0.2, 0.1]) == 0.0
    assert auc([0, 1], [0.5, 0.5]) == 0.5
    assert abs(auc([0, 0, 1, 1], [1.0, 2.0, 2.0, 3.0]) - 0.875) < 1e-12


def test_auc_is_nan_when_a_held_out_batch_has_one_class_only():
    """Not 0.5. A batch with no positives cannot be scored, and reporting 0.5
    would put a fabricated coin-flip into the mean over batches."""
    v = auc([0, 0, 0], [0.1, 0.2, 0.3])
    assert np.isnan(v)


def test_per_batch_metrics_has_one_row_per_batch_and_real_counts():
    f = synthetic_frame(n_batches=3)
    m = per_batch_metrics(f, f["signal"], label_col="label")
    assert len(m) == 3
    assert m["n"].sum() == len(f)
    assert (m["positives"] <= m["n"]).all()


def test_held_out_eval_rejects_a_wrong_length_prediction():
    f = synthetic_frame(n_batches=2, tiles_per_slide=5)
    try:
        held_out_batch_eval(f, lambda tr, te: np.zeros(3), label_col="label")
    except ValueError as exc:
        assert "scores for" in str(exc)
    else:
        raise AssertionError("a wrong-length prediction was accepted")


def test_report_names_the_worst_batch_not_the_mean():
    f = synthetic_frame(n_batches=3)
    df = held_out_batch_eval(f, lambda tr, te: te["signal"].to_numpy(),
                             label_col="label")
    text = report(df)
    assert "Worst unseen batch" in text
    assert str(df["auc"].idxmin()) in text


# --------------------------------------------------------------------------
# stain normalization
# --------------------------------------------------------------------------

def fake_he(seed: int = 0, stain_matrix=None, scale=(1.0, 1.0),
            size: int = 64) -> np.ndarray:
    """A tile synthesised through the Beer-Lambert model this code assumes.

    Two "staining runs" are the SAME concentration fields rendered through
    different stain matrices and intensities -- which is what actually differs
    between two runs of an H&E protocol, and what normalization claims to undo.
    Building the test image any other way tests something else.
    """
    from mashpath.train.stain import DEFAULT_STAIN_MATRIX

    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(float)
    c_h = 0.25 + 1.20 * np.sin(xs / 6.0) ** 2 + rng.normal(0, 0.03, (size, size))
    c_e = 0.35 + 0.90 * np.cos(ys / 4.5) ** 2 + rng.normal(0, 0.03, (size, size))
    c = np.stack([np.clip(c_h, 0.05, None) * scale[0],
                  np.clip(c_e, 0.05, None) * scale[1]]).reshape(2, -1)
    m = DEFAULT_STAIN_MATRIX if stain_matrix is None else np.asarray(stain_matrix)
    return od_to_rgb((m @ c).T).reshape(size, size, 3)


OTHER_RUN = np.array([[0.62, 0.16],      # a plausible second staining run:
                      [0.68, 0.83],      # same two stains, different vectors
                      [0.39, 0.53]])
OTHER_RUN = OTHER_RUN / np.linalg.norm(OTHER_RUN, axis=0, keepdims=True)


def test_od_round_trips():
    img = fake_he()
    back = od_to_rgb(rgb_to_od(img))
    assert np.abs(back.astype(int) - img.astype(int)).max() <= 2


def test_macenko_finds_two_normalised_stain_vectors():
    m = macenko_stain_matrix(fake_he())
    assert m.shape == (3, 2)
    assert np.allclose(np.linalg.norm(m, axis=0), 1.0, atol=1e-6)
    assert (m >= 0).all()


def test_macenko_refuses_a_tile_with_no_stained_tissue():
    blank = np.full((64, 64, 3), 250, dtype=np.uint8)
    try:
        macenko_stain_matrix(blank)
    except ValueError as exc:
        assert "not enough stained tissue" in str(exc)
    else:
        raise AssertionError("an empty tile produced stain vectors")


def test_normalization_brings_two_differently_stained_copies_closer():
    """The claim normalization makes. Checked, not assumed, for both methods."""
    a = fake_he(0)
    b = fake_he(0, stain_matrix=OTHER_RUN, scale=(1.35, 0.75))
    before = abs(a.mean(axis=(0, 1)) - b.mean(axis=(0, 1))).mean()
    for norm in (MacenkoNormalizer(), ReinhardNormalizer().fit(a)):
        na, nb = norm.transform(a), norm.transform(b)
        after = abs(na.mean(axis=(0, 1)) - nb.mean(axis=(0, 1))).mean()
        assert after < before, (f"{type(norm).__name__} did not reduce the gap: "
                                f"{before:.2f} -> {after:.2f}")


def test_the_default_is_off_and_the_off_path_is_a_real_object():
    cfg = MashConfig.from_yaml([ROOT / "configs/core.yaml"])
    assert cfg.stain.enabled is False
    assert cfg.stain.method == "none"
    norm = normalizer_from_config(cfg.stain)
    assert isinstance(norm, NullNormalizer)
    img = fake_he()
    assert np.array_equal(norm.transform(img), img)


def test_an_unknown_normalizer_name_raises():
    try:
        make_normalizer("vahadane")
    except ValueError as exc:
        assert "unknown stain normalization" in str(exc)
    else:
        raise AssertionError("an unimplemented method was accepted")


# --------------------------------------------------------------------------
# colour augmentation
# --------------------------------------------------------------------------

def test_jitter_changes_colour_stays_in_range_and_keeps_shape():
    img = fake_he()
    for aug in (ColourJitter(probability=1.0), StainJitter(probability=1.0)):
        out = aug(img, np.random.default_rng(1))
        assert out.shape == img.shape and out.dtype == np.uint8
        assert not np.array_equal(out, img), f"{type(aug).__name__} was a no-op"


def test_jitter_is_reproducible_from_its_generator():
    img = fake_he()
    aug = ColourJitter(probability=1.0)
    a = aug(img, np.random.default_rng(7))
    b = aug(img, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_jitter_moves_a_batch_toward_another_batch_appearance():
    """The reason for the strength: the jitter has to span real between-batch
    variation, or it teaches invariance over the wrong range."""
    a = fake_he(0)
    b = fake_he(0, stain_matrix=OTHER_RUN, scale=(1.35, 0.75))
    target = b.mean(axis=(0, 1))
    aug = ColourJitter(probability=1.0)
    gaps = [abs(aug(a, np.random.default_rng(s)).mean(axis=(0, 1)) - target).mean()
            for s in range(60)]
    base = abs(a.mean(axis=(0, 1)) - target).mean()
    assert min(gaps) < base, ("no draw in 60 reached the other batch's "
                              f"appearance (best {min(gaps):.2f} vs {base:.2f})")


def test_augmentation_strength_none_is_an_identity_pipeline():
    img = fake_he()
    assert np.array_equal(default_pipeline("none")(img), img)
    cfg = MashConfig.from_yaml([ROOT / "configs/core.yaml"])
    assert cfg.augment.strength == "strong"
    cfg.augment.strength = "none"
    assert np.array_equal(augmenter_from_config(cfg.augment)(img), img)


def test_an_unknown_augment_strength_raises():
    try:
        default_pipeline("extreme")
    except ValueError as exc:
        assert "unknown strength" in str(exc)
    else:
        raise AssertionError("an unknown strength was accepted")


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
