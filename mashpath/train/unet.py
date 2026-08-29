"""A U-Net on the pseudo-labels, and a loss that treats them as what they are.

Nothing in this file assumes the label is correct. `outputs/dataset_v2` ships a
per-pixel confidence map -- the fraction of nine parameter settings that agreed
that pixel is fat -- and on a positive tile a mean 26% of the labelled area is
contested rather than unanimous. A plain binary cross-entropy would spend the
same gradient on a pixel nine members agreed about and a pixel five of them
argued over, which is the single easiest way to make a weakly supervised model
confidently reproduce its teacher's mistakes.

WHAT IS BEING MEASURED, AND WHAT CANNOT BE. Held-out-batch agreement with the
pseudo-labels is agreement with the TEACHER, not accuracy; a model that matched
the teacher perfectly would have learned its false positives too. So it is
reported as `dice_vs_teacher` and never as Dice, and the numbers that carry
weight are the two that do not come from the teacher:

  the reserved batch    R22-354, four chow against four NASH, stained together.
                        Known diet, never trained on, and not scored by the
                        detector under test. This is the external validation.
  the CCl4 floor        fat-free tissue. The model must not invent droplets on
                        it, and 0.17% is the number to beat.

A model that agrees with the teacher at Dice 0.9, separates chow from NASH, and
holds the floor has learned steatosis. One that agrees at 0.9 and fails either
of the other two has learned the teacher.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("KERAS_BACKEND", "tensorflow")


@dataclass
class UNetConfig:
    """Everything the training run needs. Saved next to the weights."""

    # --- data ---
    crop: int = 256          # random crop from the 512 px tile; see `_crop`
    batch_size: int = 16
    shuffle_buffer: int = 512
    # --- model ---
    base_filters: int = 32
    depth: int = 4
    dropout: float = 0.1
    # --- optimisation ---
    epochs: int = 15
    learning_rate: float = 1e-3
    # --- the two ablation switches ---
    #
    # Both default ON here and each is a claim that has to be measured, not
    # assumed. `docs/BATCH_EFFECTS.md` §4-5 argues normalization and
    # augmentation are opposites: normalizing tries to make every batch look
    # alike and asks the model to trust that it worked, augmenting makes the
    # model stop relying on how a batch looks at all. Stain normalization is
    # therefore OFF by default there and stays off here -- the first number
    # must be the un-normalized baseline anything later has to beat.
    confidence_weighting: bool = True
    augment: str = "strong"          # none | mild | strong
    stain_normalize: str = "none"    # none | macenko | reinhard
    # Pixels below this confidence contribute nothing. 0 keeps every pixel and
    # lets the weight do the work; raising it is a harder, more explicit claim
    # about which pseudo-label pixels are worth learning.
    confidence_floor: float = 0.0
    # Slides held out of the fit, from the TRAINING batches, for early stopping
    # and checkpoint selection. See `_dev_split`.
    dev_fraction: float = 0.12
    # Weight on the soft-Dice term. 0 is plain weighted cross-entropy, which on
    # this class balance collapses to an empty mask -- see `soft_dice_loss`.
    dice_weight: float = 1.0
    seed: int = 0

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ---- model ----------------------------------------------------------------

def build_unet(cfg: UNetConfig, channels: int = 3):
    """Plain U-Net. Deliberately unremarkable.

    The contribution this project is making is the labels and the evaluation,
    not the architecture, and an unusual backbone would make every held-out
    batch number harder to attribute. Padding is 'same' throughout so an output
    pixel maps to its input pixel, which is what lets a predicted mask be
    compared with a pseudo-label without a crop offset.
    """
    from keras import layers, models

    def block(x, f):
        for _ in range(2):
            x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        return x

    inp = layers.Input((None, None, channels))
    skips, x, f = [], inp, cfg.base_filters
    for _ in range(cfg.depth):
        x = block(x, f)
        skips.append(x)
        x = layers.MaxPooling2D(2)(x)
        if cfg.dropout:
            x = layers.Dropout(cfg.dropout)(x)
        f *= 2
    x = block(x, f)
    for skip in reversed(skips):
        f //= 2
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skip])
        if cfg.dropout:
            x = layers.Dropout(cfg.dropout)(x)
        x = block(x, f)
    out = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32")(x)
    return models.Model(inp, out, name="unet")


# ---- loss -----------------------------------------------------------------

def weighted_bce(y_true, y_pred, sample_weight=None):
    """Per-pixel BCE, weighted by how much the ensemble agreed about that pixel.

    Keras applies `sample_weight` for us when it is passed through the dataset,
    so this exists to keep the reduction explicit: the mean is over WEIGHT, not
    over pixels. Dividing by pixel count instead would let a tile whose label is
    almost entirely contested contribute as much gradient as one the ensemble
    was unanimous about, which is the opposite of the intent.
    """
    import keras.ops as K
    eps = 1e-7
    p = K.clip(y_pred, eps, 1.0 - eps)
    bce = -(y_true * K.log(p) + (1.0 - y_true) * K.log(1.0 - p))
    if sample_weight is None:
        return K.mean(bce)
    w = sample_weight
    return K.sum(bce * w) / (K.sum(w) + eps)


def soft_dice_loss(y_true, y_pred, sample_weight=None):
    """1 - soft Dice, confidence-weighted. The term that stops the collapse.

    Fat is roughly 5% of the pixels in a positive tile and 0% in most negative
    ones, and cross-entropy's cheapest minimum on that distribution is an empty
    mask: predicting nothing everywhere scores a low loss and a smooth curve.
    The first smoke run did exactly that -- three epochs, falling training and
    validation loss, and every predicted fat fraction 0.000, which reads as a
    working run right up until someone looks at a prediction.

    Dice has no such minimum. An empty prediction scores 0 overlap against any
    non-empty label however small that label is, so the term is large exactly
    where cross-entropy has gone quiet.

    Weighted by the same confidence map, so a contested pseudo-label pixel is
    worth less here too. Computed over the whole batch rather than per image:
    per-image Dice on a tile whose label is empty is either undefined or 1 by
    convention, and 67% of the negative tiles carry an empty mask.
    """
    import keras.ops as K
    # Smoothing of 1.0, not 1e-6. With a tiny epsilon the empty-label case has a
    # cliff: when the label is empty the loss is 1 - eps/(sum(p)+eps), which
    # keeps falling as the outputs are driven toward 1e-12, so the model is
    # rewarded for numerical extremity rather than for segmentation. Measured on
    # the CCl4 fold, where 67% of tiles carry an empty label: validation loss
    # reached 0.0013 after one epoch, which is that cliff and not a good model.
    # At 1.0 the empty case degrades smoothly to roughly sum(p) and there is
    # nothing to win past predicting nothing.
    smooth = 1.0
    w = K.ones_like(y_true) if sample_weight is None else sample_weight
    num = 2.0 * K.sum(w * y_true * y_pred)
    den = K.sum(w * y_true) + K.sum(w * y_pred)
    return 1.0 - (num + smooth) / (den + smooth)


def make_loss(cfg: "UNetConfig"):
    """`weighted_bce` plus `dice_weight` times the soft-Dice term.

    Both, not one: Dice alone is unstable early -- with a near-empty prediction
    its gradient is dominated by a handful of pixels -- while cross-entropy is
    well behaved everywhere and merely has the wrong minimum. Together, BCE
    supplies the stable early signal and Dice supplies the reason not to
    settle for nothing.
    """
    def loss(y_true, y_pred, sample_weight=None):
        out = weighted_bce(y_true, y_pred, sample_weight)
        if cfg.dice_weight:
            out = out + cfg.dice_weight * soft_dice_loss(y_true, y_pred,
                                                         sample_weight)
        return out
    loss.__name__ = "weighted_bce_dice"
    return loss


def dice(y_true, y_pred, threshold: float = 0.5) -> float:
    """Agreement with the pseudo-label. Named `dice_vs_teacher` at the call
    site, because against a programmatic label this is not accuracy."""
    a = np.asarray(y_true, dtype=bool)
    b = np.asarray(y_pred, dtype=float) >= threshold
    denom = a.sum() + b.sum()
    # Two empty masks agree completely, and on this dataset 67% of negative
    # tiles carry one. Returning 0 there would drag every negative batch's
    # mean down and read as a model failure on exactly the tissue it handles
    # best; NaN would silently drop them. 1.0 is the honest answer to "how much
    # do these two masks agree".
    return 1.0 if denom == 0 else float(2.0 * (a & b).sum() / denom)


# ---- data -----------------------------------------------------------------

def _paths(df: pd.DataFrame, root: Path) -> tuple[list[str], list[str], list[str]]:
    ids = df["id"].tolist()
    return ([str(root / "images" / f"{i}.png") for i in ids],
            [str(root / "labels" / f"{i}.png") for i in ids],
            [str(root / "confidence" / f"{i}.png") for i in ids])


def make_dataset(df: pd.DataFrame, root: str | Path, cfg: UNetConfig,
                 training: bool = True, batch_size: int | None = None):
    """tf.data over a slice of the manifest.

    The slice is chosen by the CALLER, from `mashpath.train.splits`. This
    function takes rows and never decides which rows: given a `frame` and a
    `seed` it could produce a random split, and then nothing downstream could
    distinguish that from the batch-aware one it is supposed to be fed. Passing
    an already-split frame makes the wrong thing unreachable from here.
    """
    import tensorflow as tf

    root = Path(root)
    imgs, labs, confs = _paths(df, root)
    bs = batch_size or cfg.batch_size

    aug = None
    if training and cfg.augment != "none":
        from .augment import default_pipeline
        aug = default_pipeline(cfg.augment)
    norm = None
    if cfg.stain_normalize != "none":
        from .stain import make_normalizer
        norm = make_normalizer(cfg.stain_normalize)

    def _read(p, channels=3):
        return tf.image.decode_png(tf.io.read_file(p), channels=channels)

    def load(ip, lp, cp):
        x = tf.cast(_read(ip), tf.float32) / 255.0
        y = tf.cast(_read(lp)[..., :1], tf.float32) / 255.0
        w = tf.cast(_read(cp)[..., :1], tf.float32) / 255.0
        return x, y, w

    def crop(x, y, w):
        # One crop window for all three, so image, label and confidence stay
        # registered. Concatenating first is the only way tf.image.random_crop
        # can be trusted to use one window.
        s = tf.concat([x, y, w], axis=-1)
        s = tf.image.random_crop(s, (cfg.crop, cfg.crop, 5))
        return s[..., :3], s[..., 3:4], s[..., 4:5]

    def geometric(x, y, w):
        # Free and exactly correct for histology: a liver section has no
        # canonical orientation, so a flip or a quarter turn is another real
        # section rather than a distortion of this one.
        s = tf.concat([x, y, w], axis=-1)
        s = tf.image.random_flip_left_right(s)
        s = tf.image.random_flip_up_down(s)
        s = tf.image.rot90(s, tf.random.uniform((), 0, 4, tf.int32))
        return s[..., :3], s[..., 3:4], s[..., 4:5]

    def colour(x, y, w):
        def _np(a):
            u8 = (np.clip(a, 0, 1) * 255).astype(np.uint8)
            if norm is not None:
                try:
                    u8 = norm(u8)
                except Exception:
                    # A tile with too little stained tissue: both normalizers
                    # refuse rather than guess, and the un-normalized tile is
                    # the honest fallback. Silently substituting the reference
                    # matrix would make it LOOK normalized.
                    pass
            if aug is not None:
                u8 = aug(u8)
            return (u8.astype(np.float32) / 255.0)
        out = tf.numpy_function(_np, [x], tf.float32)
        out.set_shape(x.shape)
        return out, y, w

    def weight(x, y, w):
        if not cfg.confidence_weighting:
            w = tf.ones_like(w)
        elif cfg.confidence_floor > 0:
            w = tf.where(w >= cfg.confidence_floor, w, tf.zeros_like(w))
        return x, y, w

    AUTO = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((imgs, labs, confs))
    if training:
        ds = ds.shuffle(min(cfg.shuffle_buffer, len(imgs)), seed=cfg.seed,
                        reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=AUTO)
    if training:
        ds = ds.map(crop, num_parallel_calls=AUTO)
        ds = ds.map(geometric, num_parallel_calls=AUTO)
        if aug is not None or norm is not None:
            ds = ds.map(colour, num_parallel_calls=AUTO)
    elif norm is not None:
        ds = ds.map(colour, num_parallel_calls=AUTO)
    ds = ds.map(weight, num_parallel_calls=AUTO)
    return ds.batch(bs).prefetch(AUTO)


# ---- evaluation -----------------------------------------------------------

def predict_frame(model, df: pd.DataFrame, root: str | Path, cfg: UNetConfig,
                  threshold: float = 0.5, batch_size: int = 8) -> pd.DataFrame:
    """Per tile: predicted fat fraction and agreement with the pseudo-label.

    Evaluated on the WHOLE 512 px tile, not a crop. The model is fully
    convolutional so nothing stops it, and a metric computed on random crops
    would not be comparable between runs -- the crop, not the model, would be
    part of the number.
    """
    import tensorflow as tf

    root = Path(root)
    out = df.copy().reset_index(drop=True)
    ds = make_dataset(out, root, cfg, training=False, batch_size=batch_size)

    # Predictions are confined to tissue, because the teacher's were. The
    # detector does `white &= tissue_mask` before it measures anything, so a
    # student scored on the whole tile is scored on a larger canvas than the
    # one that produced its labels -- and since glass is white, the difference
    # is not noise. `make_dataset(training=False)` does not shuffle, so the
    # tiles come back in the order of `out` and the mask for row i is row i's.
    tissue_dir = root / "tissue"
    have_tissue = tissue_dir.is_dir()
    if not have_tissue:
        print(f"WARNING: {tissue_dir} is missing; predictions are NOT confined "
              f"to tissue and are not comparable with the teacher's numbers",
              flush=True)

    import cv2
    preds, dices, row = [], [], 0
    for xb, yb, _ in ds:
        pb = model.predict_on_batch(xb)
        for i in range(pb.shape[0]):
            p = np.asarray(pb[i, ..., 0]) >= threshold
            y = np.asarray(yb[i, ..., 0]) >= 0.5
            if have_tissue:
                tp = tissue_dir / f"{out['id'].iloc[row]}.png"
                t = cv2.imread(str(tp), cv2.IMREAD_GRAYSCALE)
                if t is not None and t.shape == p.shape:
                    p = p & (t > 127)
            preds.append(float(p.sum()))
            dices.append(dice(y, p.astype(float), 0.5))
            row += 1
    px = out["tile_size"] ** 2 * out["tissue_fraction"].clip(lower=1e-6)
    out["pred_label_px"] = preds[:len(out)]
    out["pred_fat_fraction"] = out["pred_label_px"] / px
    out["dice_vs_teacher"] = dices[:len(out)]
    return out


def slide_summary(pred: pd.DataFrame) -> pd.DataFrame:
    """One row per slide. Tiles within a slide are not independent, so every
    claim about an animal is made here rather than on the tile table."""
    keep = [c for c in ("batch", "cohort", "diet") if c in pred.columns]
    return (pred.groupby(["slide"] + keep, dropna=False)
                .agg(tiles=("id", "size"),
                     teacher_fat=("fat_fraction", "mean"),
                     model_fat=("pred_fat_fraction", "mean"),
                     dice_vs_teacher=("dice_vs_teacher", "mean"))
                .reset_index())


def external_test(pred: pd.DataFrame) -> dict:
    """Chow vs NASH on the reserved batch: the one number not from the teacher.

    Reported at slide level, because diet is a property of an animal, and
    alongside the teacher's own AUC on the same slides so the two are read
    together. A model scoring 1.000 where the teacher also scores 1.000 has
    inherited a working measurement; a model scoring below it has lost
    something the pseudo-labels contained.
    """
    from .evaluate import auc
    sl = slide_summary(pred)
    if "diet" not in sl.columns:
        return {"error": "no diet column"}
    n = sl[sl["diet"] == "nash"]; c = sl[sl["diet"] == "chow"]
    if n.empty or c.empty:
        return {"error": f"need both diets, have {sl['diet'].unique().tolist()}"}
    y = np.r_[np.ones(len(n)), np.zeros(len(c))]
    return {
        "n_nash": len(n), "n_chow": len(c),
        "model_auc": auc(y, np.r_[n["model_fat"], c["model_fat"]]),
        "teacher_auc": auc(y, np.r_[n["teacher_fat"], c["teacher_fat"]]),
        "model_chow_max": float(c["model_fat"].max()),
        "model_nash_min": float(n["model_fat"].min()),
        "teacher_chow_max": float(c["teacher_fat"].max()),
        "teacher_nash_min": float(n["teacher_fat"].min()),
    }


def floor_test(pred: pd.DataFrame, negative_batch: str = "2026-04-20_CCl4_HE") -> dict:
    """Does the model invent fat on tissue known to have none?

    The teacher's floor is 0.17% mean / 0.44% worst slide. A student above that
    has learned to hallucinate droplets; a student below it is the interesting
    case, and is the only way this design can show the model beating its
    teacher rather than copying it.
    """
    sl = slide_summary(pred)
    neg = sl[sl["batch"] == negative_batch]
    if neg.empty:
        return {"error": f"no slides from {negative_batch!r}"}
    return {"slides": len(neg),
            "model_mean": float(neg["model_fat"].mean()),
            "model_worst": float(neg["model_fat"].max()),
            "teacher_mean": float(neg["teacher_fat"].mean()),
            "teacher_worst": float(neg["teacher_fat"].max())}


def _dev_split(train: pd.DataFrame, fraction: float, seed: int
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Whole slides from the TRAINING batches, for checkpoint selection.

    Early stopping and "save the best epoch" are model selection, and doing
    them on the held-out batch means the held-out batch is not held out --
    the fold's headline number would then be reported on data that chose the
    weights. `BATCH_EFFECTS.md` spends its length on making the wrong split
    unreachable; leaving this one in place would undo that at the last step.

    It is also degenerate on some folds. On the CCl4 fold 67% of held-out tiles
    carry an empty label, so a loss computed there is minimised by predicting
    nothing at all, and the checkpoint chosen would be the emptiest epoch.

    By slide, never by tile -- adjacent tiles share staining and often the same
    hepatocytes across their border. Stratified by (batch, is_negative) so the
    dev set cannot come back all-positive or missing a staining run, and every
    stratum yields at least one slide.
    """
    rng = np.random.default_rng(seed)
    keys = ["batch"] + (["is_negative"] if "is_negative" in train.columns else [])
    dev_slides: list[str] = []
    for _, grp in train.groupby(keys, dropna=False):
        slides = sorted(grp["slide"].unique())
        n = max(1, int(round(len(slides) * fraction)))
        n = min(n, len(slides) - 1) if len(slides) > 1 else 0
        if n:
            dev_slides += list(rng.choice(slides, n, replace=False))
    dev_set = set(dev_slides)
    dev = train[train["slide"].isin(dev_set)]
    fit = train[~train["slide"].isin(dev_set)]
    if dev.empty or fit.empty:
        raise ValueError(f"dev_fraction {fraction} left {len(fit)} fit rows and "
                         f"{len(dev)} dev rows")
    return fit, dev


# ---- one fold -------------------------------------------------------------

def train_fold(manifest: pd.DataFrame, root: str | Path, held_out: str,
               cfg: UNetConfig, out_dir: str | Path, verbose: int = 2):
    """Train on every batch except `held_out`, validate on it.

    The split comes from `split_by_batch`, which is keyword-only on its
    `validation` argument, subtracts the reserved batches from train before
    anything else, and calls `assert_no_leakage` itself. Building the two
    frames by hand here would work and would also be the exact shortcut that
    module exists to make unreachable.
    """
    import keras
    from .splits import split_by_batch

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    split = split_by_batch(manifest, validation=[held_out])
    tr, va = split.subset(manifest, "train"), split.subset(manifest, "validation")
    print(split.summary(manifest).to_string(), flush=True)
    if tr.empty or va.empty:
        raise ValueError(f"fold {held_out!r} has an empty side: "
                         f"{len(tr)} train, {len(va)} validation rows")

    fit_df, dev_df = _dev_split(tr, cfg.dev_fraction, cfg.seed)
    print(f"fit {len(fit_df)} rows / {fit_df.slide.nunique()} slides   "
          f"dev {len(dev_df)} rows / {dev_df.slide.nunique()} slides   "
          f"held-out {len(va)} rows / {va.slide.nunique()} slides", flush=True)

    keras.utils.set_random_seed(cfg.seed)
    model = build_unet(cfg)
    model.compile(optimizer=keras.optimizers.Adam(cfg.learning_rate),
                  loss=make_loss(cfg))
    hist = model.fit(
        make_dataset(fit_df, root, cfg, training=True),
        # Dev, NOT the held-out batch: see `_dev_split`. The held-out batch is
        # scored once, after training, and never chooses a weight.
        validation_data=make_dataset(dev_df, root, cfg, training=False,
                                     batch_size=max(cfg.batch_size // 2, 1)),
        epochs=cfg.epochs, verbose=verbose,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                str(out_dir / "best.keras"), monitor="val_loss",
                save_best_only=True),
            # Restore, so what gets evaluated is the epoch that validated best
            # on staining the model never trained on -- not whichever epoch the
            # loop happened to stop at.
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                          restore_best_weights=True),
            keras.callbacks.CSVLogger(str(out_dir / "history.csv")),
        ])
    import json
    (out_dir / "config.json").write_text(json.dumps(
        {**cfg.to_dict(), "held_out_batch": held_out,
         "train_batches": sorted(split.folds["train"]),
         "fit_rows": len(fit_df), "fit_slides": int(fit_df.slide.nunique()),
         "dev_rows": len(dev_df), "dev_slides": sorted(dev_df.slide.unique()),
         "held_out_rows": len(va)}, indent=2))
    return model, hist


def main(argv: list[str] | None = None) -> int:
    """One fold per invocation, so a SLURM array is the natural driver.

        python -m mashpath.train.unet --manifest DS/manifest.csv --root DS \\
            --held-out 2025-08-25HE --out runs/lobo/2025-08-25HE \\
            --reserved-root DS_reserved

    `--reserved-root` points at the separately exported R22-354 tiles. They are
    not in the training manifest at all -- `build_flat_dataset` refuses to
    export a reserved batch -- so evaluating on them takes a second path that
    someone has to name deliberately.
    """
    import argparse
    import json

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--held-out", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reserved-root", default=None,
                   help="directory of the reserved batch's exported tiles")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--crop", type=int, default=None)
    p.add_argument("--base-filters", type=int, default=None,
                   help="width of the first block; halving it is ~4x cheaper "
                        "and is how this runs on a CPU at all")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--dice-weight", type=float, default=None,
                   help="ablation: 0 is plain cross-entropy, which collapses")
    p.add_argument("--no-confidence", action="store_true",
                   help="ablation: train with every pseudo-label pixel equal")
    p.add_argument("--augment", default=None, choices=["none", "mild", "strong"])
    p.add_argument("--stain-normalize", default=None,
                   choices=["none", "macenko", "reinhard"])
    a = p.parse_args(argv)

    cfg = UNetConfig()
    for k, v in (("epochs", a.epochs), ("crop", a.crop),
                 ("base_filters", a.base_filters),
                 ("batch_size", a.batch_size), ("augment", a.augment),
                 ("stain_normalize", a.stain_normalize),
                 ("dice_weight", a.dice_weight)):
        if v is not None:
            setattr(cfg, k, v)
    if a.no_confidence:
        cfg.confidence_weighting = False

    manifest = pd.read_csv(a.manifest)
    out = Path(a.out)
    model, _ = train_fold(manifest, a.root, a.held_out, cfg, out)

    from .splits import split_by_batch
    va = split_by_batch(manifest, validation=[a.held_out]).subset(
        manifest, "validation")
    pred = predict_frame(model, va, a.root, cfg)
    pred.to_csv(out / "predictions_heldout.csv", index=False)
    summary = {"held_out_batch": a.held_out,
               "dice_vs_teacher": float(pred["dice_vs_teacher"].mean()),
               "floor": floor_test(pred)}

    if a.reserved_root:
        rroot = Path(a.reserved_root)
        rman = pd.read_csv(rroot / "manifest.csv")
        rpred = predict_frame(model, rman, rroot, cfg)
        rpred.to_csv(out / "predictions_reserved.csv", index=False)
        summary["external"] = external_test(rpred)

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float), flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
