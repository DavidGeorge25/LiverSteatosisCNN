"""Held-out-batch evaluation: what the model does on staining it has not seen.

WHY PER BATCH AND NOT POOLED. A pooled metric over a leave-one-batch-out sweep
is an average over the very thing under test. If a model reaches AUC 0.90 on six
batches and 0.55 on the seventh, the mean is 0.85 and reads as a good model; the
0.55 is the actual finding, and it is the number that predicts what happens when
the lab stains a new run next month. Everything here reports per unseen batch
and shows the WORST case next to the mean, because the worst case is the claim
that has to survive.

WHAT THE BATCH-SEPARABILITY COLUMN IS FOR. `batch_separability` asks a different
question from accuracy: how well does the model's own output tell one batch from
the others, ignoring labels entirely? A model that has learned ballooning should
score a MASH tile and a chow tile differently and should not much care which
staining run either came from. A model that has learned stain shows high
separability, and it shows it even when its accuracy looks fine — because with
batch confounded with cohort, reading the stain IS a way to get the labels right
in-distribution. This is the column that catches the failure mode the split is
designed to prevent, in the case where the split alone was not enough.

No sklearn dependency: AUC is the rank-sum identity, which is exact and handles
ties by average ranks.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .splits import BATCH_COL, leave_one_batch_out


def auc(y_true, y_score) -> float:
    """Area under the ROC curve via the Mann-Whitney U identity.

    NaN when one class is absent, which is a real and common state in a
    held-out batch and must not be reported as 0.5.
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(y_score, dtype=float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # Average ranks within tied score groups.
    srt = s[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return (ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def binary_metrics(y_true, y_score, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(y_score, dtype=float)
    pred = s >= threshold
    tp = int((pred & y).sum())
    tn = int((~pred & ~y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return {"n": len(y), "positives": int(y.sum()),
            "prevalence": float(y.mean()) if len(y) else float("nan"),
            "auc": auc(y, s),
            "sensitivity": sens, "specificity": spec,
            "balanced_accuracy": float(np.nanmean([sens, spec])),
            "accuracy": (tp + tn) / len(y) if len(y) else float("nan"),
            "mean_score": float(s.mean()) if len(s) else float("nan")}


def batch_separability(y_score, batches) -> pd.Series:
    """Per batch: AUC of "is this tile from this batch?" against the score alone.

    0.5 means the score carries no information about which staining run a tile
    came from, which is what a model that reads biology should look like.
    Values far from 0.5 -- in either direction -- mean the output has a batch
    offset in it.
    """
    s = np.asarray(y_score, dtype=float)
    b = pd.Series(list(batches))
    return pd.Series({name: auc((b == name).to_numpy(), s)
                      for name in b.unique()}, name="batch_separability")


def per_batch_metrics(frame: pd.DataFrame, y_score, label_col: str = "label",
                      threshold: float = 0.5) -> pd.DataFrame:
    """One row per batch present in `frame`."""
    if label_col not in frame.columns:
        raise KeyError(f"frame has no {label_col!r} column")
    s = np.asarray(y_score, dtype=float)
    if len(s) != len(frame):
        raise ValueError(f"{len(s)} scores for {len(frame)} rows")
    rows = {}
    for b, idx in frame.groupby(BATCH_COL).groups.items():
        pos = frame.index.get_indexer(idx)
        rows[b] = binary_metrics(frame.loc[idx, label_col], s[pos], threshold)
    out = pd.DataFrame(rows).T
    out.index.name = BATCH_COL
    return out


def held_out_batch_eval(frame: pd.DataFrame,
                        fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
                        label_col: str = "label",
                        threshold: float = 0.5) -> pd.DataFrame:
    """Leave-one-batch-out. `fit_predict(train, test) -> score per test row`.

    The callable is handed DataFrames and returns an array, so this harness
    knows nothing about the model — it works the same for a CNN, a logistic
    regression on hand features, or a stub that returns noise, which is what
    makes it testable before any model exists.
    """
    out = []
    for held, split in leave_one_batch_out(frame):
        train = split.subset(frame, "train")
        test = split.subset(frame, "validation")
        scores = np.asarray(fit_predict(train, test), dtype=float)
        if len(scores) != len(test):
            raise ValueError(f"fit_predict returned {len(scores)} scores for "
                             f"{len(test)} held-out rows of batch {held!r}")
        m = binary_metrics(test[label_col], scores, threshold)
        m["held_out_batch"] = held
        m["train_batches"] = len(split.folds["train"])
        m["train_rows"] = len(train)
        out.append(m)
    df = pd.DataFrame(out).set_index("held_out_batch")
    return df


def report(df: pd.DataFrame, title: str = "Held-out-batch evaluation") -> str:
    """Markdown, worst case first — it is the number that has to survive."""
    cols = [c for c in ("n", "prevalence", "auc", "sensitivity", "specificity",
                        "balanced_accuracy") if c in df.columns]
    L = [f"# {title}", "",
         "| held-out batch | " + " | ".join(cols) + " |",
         "| --- |" + " ---: |" * len(cols)]
    for name, r in df.iterrows():
        vals = " | ".join("—" if pd.isna(r[c]) else
                          (f"{r[c]:.0f}" if c == "n" else f"{r[c]:.3f}")
                          for c in cols)
        L.append(f"| {name} | {vals} |")
    if "auc" in df.columns and df["auc"].notna().any():
        worst = df["auc"].idxmin()
        L += ["",
              f"**Worst unseen batch: {worst}, AUC {df['auc'].min():.3f}.** "
              f"Mean over batches {df['auc'].mean():.3f}, "
              f"spread {df['auc'].max() - df['auc'].min():.3f}.",
              "",
              "The mean is the number to quote only if the spread is small. A "
              "wide spread means performance depends on which staining run the "
              "tiles came from, which is the failure this evaluation exists to "
              "make visible."]
    return "\n".join(L) + "\n"
