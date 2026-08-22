"""Held-out validation of parameter selection.

The sweep picks the best of 360 parameter cells by scoring every cell on every
slide. Reporting that winning cell's score on the same slides is circular: with
360 cells and 51 slides, the maximum is partly fitted to sample noise, so the
in-sample separation is optimistically biased.

This module measures that bias and reports an honest out-of-sample estimate:

  * `holdout_validation` -- repeatedly split slides into tune/test, select on
    tune only, evaluate the selected cell on test. The gap between the two is
    the selection optimism.
  * `bootstrap_metrics` -- resample slides to put confidence intervals on a
    fixed cell's metrics.
  * `selection_stability` -- how often each cell wins across resamples, i.e.
    whether "the best parameters" is a stable claim or a coin flip.

Everything operates on the per-slide sweep CSV, so no slide is re-read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

META_COLUMNS = {"slide", "group", "tiles", "macro_pct", "micro_pct", "macro_n"}


def param_keys(df: pd.DataFrame) -> list[str]:
    """The parameter-axis columns of a per-slide sweep frame."""
    return [c for c in df.columns if c not in META_COLUMNS]


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Probability a random positive slide scores above a random negative one.

    Mann-Whitney U / (n_pos * n_neg), ties counted as half. 1.0 means the two
    cohorts are perfectly separable by some threshold; 0.5 means no signal.
    Rank-based, so it does not care about the scale of macro_pct.
    """
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    r_pos = ranks[: pos.size].sum()
    u = r_pos - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def score_cells(
    df: pd.DataFrame,
    slides: Sequence[str] | None = None,
    positive: str = "MASH",
    negative: str = "CCl4",
    compute_auc: bool = True,
) -> pd.DataFrame:
    """Per-cell separation metrics, optionally restricted to a slide subset.

    `compute_auc` is the expensive part (one rank per cell); the repeated-split
    loop turns it off for the selection pass and computes it only on the
    selected cell.
    """
    if slides is not None:
        df = df[df["slide"].isin(set(slides))]
    keys = param_keys(df)

    pos_df = df[df["group"] == positive]
    neg_df = df[df["group"] == negative]
    pos = pos_df.groupby(keys)["macro_pct"]
    neg = neg_df.groupby(keys)["macro_pct"]

    out = pd.DataFrame({
        "pos_mean": pos.mean(),
        "pos_min": pos.min(),
        "neg_mean": neg.mean(),
        "neg_max": neg.max(),
    })

    if compute_auc:
        # AUC needs the raw per-slide values per cell, not an aggregate.
        pos_groups = {k: v.to_numpy() for k, v in pos_df.groupby(keys)["macro_pct"]}
        neg_groups = {k: v.to_numpy() for k, v in neg_df.groupby(keys)["macro_pct"]}
        out["auc"] = pd.Series(
            {k: auc(pos_groups.get(k, []), neg_groups.get(k, [])) for k in out.index}
        )
    else:
        out["auc"] = np.nan

    out = out.reset_index()
    out["margin"] = out["pos_min"] - out["neg_max"]
    out["ratio"] = out["pos_mean"] / out["neg_mean"].replace(0, np.nan)
    return out


def _cell_key(row: pd.Series, keys: Sequence[str]) -> tuple:
    return tuple(row[k] for k in keys)


@dataclass
class HoldoutResult:
    repeats: pd.DataFrame  # one row per split
    optimism: float  # mean(tune metric) - mean(test metric)
    test_mean: float
    test_lo: float
    test_hi: float
    stability: pd.DataFrame  # selection counts per cell


def holdout_validation(
    df: pd.DataFrame,
    positive: str = "MASH",
    negative: str = "CCl4",
    metric: str = "margin",
    n_repeats: int = 300,
    test_frac: float = 0.3,
    seed: int = 0,
) -> HoldoutResult:
    """Repeated slide-level split: select on tune slides, score on test slides.

    Splits are stratified by group, so both cohorts appear on both sides -- the
    metrics are cohort comparisons and are undefined otherwise.
    """
    keys = param_keys(df)
    rng = np.random.default_rng(seed)

    pos_slides = df.loc[df["group"] == positive, "slide"].unique()
    neg_slides = df.loc[df["group"] == negative, "slide"].unique()
    n_pos_test = max(2, int(round(len(pos_slides) * test_frac)))
    n_neg_test = max(2, int(round(len(neg_slides) * test_frac)))

    rows = []
    picks = []
    for _ in range(n_repeats):
        pos_test = rng.choice(pos_slides, n_pos_test, replace=False)
        neg_test = rng.choice(neg_slides, n_neg_test, replace=False)
        test = set(pos_test) | set(neg_test)
        tune = [s for s in df["slide"].unique() if s not in test]

        tuned = score_cells(df, tune, positive, negative, compute_auc=False)
        if tuned.empty or tuned[metric].isna().all():
            continue
        best = tuned.loc[tuned[metric].idxmax()]
        key = _cell_key(best, keys)

        tested = score_cells(df, list(test), positive, negative)
        mask = np.ones(len(tested), dtype=bool)
        for k, v in zip(keys, key):
            mask &= tested[k].to_numpy() == v
        if not mask.any():
            continue
        row_test = tested[mask].iloc[0]

        picks.append(key)
        rows.append({
            **{k: v for k, v in zip(keys, key)},
            "tune_metric": float(best[metric]),
            "test_metric": float(row_test[metric]),
            "test_auc": float(row_test["auc"]),
            "test_pos_mean": float(row_test["pos_mean"]),
            "test_neg_mean": float(row_test["neg_mean"]),
        })

    reps = pd.DataFrame(rows)
    stability = (
        pd.DataFrame(picks, columns=keys)
        .value_counts()
        .rename("times_selected")
        .reset_index()
    )
    stability["pct"] = stability["times_selected"] / len(picks) * 100 if picks else 0

    test_vals = reps["test_metric"].to_numpy() if not reps.empty else np.array([np.nan])
    return HoldoutResult(
        repeats=reps,
        optimism=float(reps["tune_metric"].mean() - reps["test_metric"].mean())
        if not reps.empty else float("nan"),
        test_mean=float(np.nanmean(test_vals)),
        test_lo=float(np.nanpercentile(test_vals, 2.5)),
        test_hi=float(np.nanpercentile(test_vals, 97.5)),
        stability=stability,
    )


def bootstrap_metrics(
    df: pd.DataFrame,
    cell: dict[str, Any],
    positive: str = "MASH",
    negative: str = "CCl4",
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Confidence intervals for one fixed parameter cell, resampling slides.

    Slides are the independent unit -- tiles within a slide are correlated, so
    resampling tiles would understate the interval badly.
    """
    sub = df.copy()
    for k, v in cell.items():
        sub = sub[sub[k] == v]
    if sub.empty:
        raise ValueError(f"no sweep rows match {cell}")

    pos = sub.loc[sub["group"] == positive, "macro_pct"].to_numpy()
    neg = sub.loc[sub["group"] == negative, "macro_pct"].to_numpy()
    rng = np.random.default_rng(seed)

    stats = {"pos_mean": [], "pos_min": [], "neg_mean": [], "neg_max": [],
             "margin": [], "auc": []}
    for _ in range(n_boot):
        p = rng.choice(pos, pos.size, replace=True)
        n = rng.choice(neg, neg.size, replace=True)
        stats["pos_mean"].append(p.mean())
        stats["pos_min"].append(p.min())
        stats["neg_mean"].append(n.mean())
        stats["neg_max"].append(n.max())
        stats["margin"].append(p.min() - n.max())
        stats["auc"].append(auc(p, n))

    rows = []
    observed = {
        "pos_mean": pos.mean(), "pos_min": pos.min(),
        "neg_mean": neg.mean(), "neg_max": neg.max(),
        "margin": pos.min() - neg.max(), "auc": auc(pos, neg),
    }
    for k, v in stats.items():
        arr = np.asarray(v, dtype=float)
        rows.append({
            "metric": k,
            "observed": observed[k],
            "boot_mean": float(np.nanmean(arr)),
            "ci_lo": float(np.nanpercentile(arr, 2.5)),
            "ci_hi": float(np.nanpercentile(arr, 97.5)),
        })
    return pd.DataFrame(rows)
