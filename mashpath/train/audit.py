"""Is a per-tile score reading biology, or reading which staining run it came from?

The question this answers is the one `BATCH_EFFECTS.md` says the whole
evaluation design exists for, asked of a score that already exists rather than
of a model that does not. It needs no labels, which is the point: it can be run
on the detector output sitting on disk today.

Three measurements, because no one of them settles it:

  `batch_separability` (in `evaluate.py`)   Can the score alone say which batch
      a TILE came from? Far from 0.5 means the score carries a stain offset.
      Near 0.5 does not mean it is clean -- a score can be free of tile-level
      batch information and still sit at a different level on every slide.

  `variance_components`   Of the spread in SLIDE medians, how much lies between
      batches rather than within them? A score measuring an animal's disease
      should vary mostly between animals. One measuring the staining run should
      vary mostly between batches. Batch is confounded with cohort here, so a
      high fraction is not proof of a stain effect -- it is the ceiling on how
      much of the cohort difference could be biology.

  `pairwise_group_auc`   Rank every pair of batches by how far apart their slide
      medians sit. A cohort contrast is only evidence if it is larger than the
      contrast between two staining runs picked at random. This is the test that
      makes "MASH scores below CCl4" interpretable, and it is the one that
      cannot be done with two batches on disk.

  `within_group_contrast`   The only clean question available: inside ONE batch,
      does the score separate the treatment groups? Stain is held fixed by
      construction, so an effect here is biology and an absence here is the
      absence of measurable biology -- whatever the cross-batch numbers say.

Nothing here is specific to ballooning. It takes a frame with a score, a batch
column and whatever grouping is of interest.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .evaluate import auc
from .splits import BATCH_COL


def slide_summary(frame: pd.DataFrame, score_col: str = "score",
                  slide_col: str = "slide",
                  keep: tuple[str, ...] = (BATCH_COL, "cohort", "diet")) -> pd.DataFrame:
    """One row per slide: the median of its finite tile scores.

    Median, not mean: the score is a max over cells within a tile and has a
    long right tail, and a single striking cell on one tile should not move a
    slide's summary.
    """
    if score_col not in frame:
        raise KeyError(f"frame has no {score_col!r} column")
    f = frame[np.isfinite(frame[score_col].to_numpy(dtype=float))]
    cols = [c for c in keep if c in f.columns]
    out = (f.groupby([slide_col] + cols, dropna=False)[score_col]
             .agg(["median", "size"])
             .rename(columns={"median": score_col, "size": "tiles"})
             .reset_index())
    return out


def variance_components(values, groups) -> dict[str, float]:
    """Between-group and within-group sums of squares, and the ratio.

    `between_fraction` is 0 when every group sits at the same level and 1 when
    all the spread is between groups. Reported as a fraction of the total
    rather than as an F statistic: there is no null hypothesis worth testing
    here -- batch and cohort are confounded by construction, so a p-value would
    be answering a question nobody asked.
    """
    v = np.asarray(values, dtype=float)
    g = pd.Series(list(groups))
    if len(v) != len(g):
        raise ValueError(f"{len(v)} values for {len(g)} group labels")
    ok = np.isfinite(v)
    v, g = v[ok], g[ok.nonzero()[0]].reset_index(drop=True)
    if len(v) < 2:
        return {"between": float("nan"), "within": float("nan"),
                "between_fraction": float("nan"), "n": int(len(v)),
                "groups": int(g.nunique())}
    grand = v.mean()
    means = pd.Series(v).groupby(g.to_numpy()).transform("mean").to_numpy()
    between = float(((means - grand) ** 2).sum())
    within = float(((v - means) ** 2).sum())
    total = between + within
    return {"between": between, "within": within,
            "between_fraction": between / total if total else float("nan"),
            "n": int(len(v)), "groups": int(g.nunique())}


def pairwise_group_auc(values, groups, min_size: int = 2) -> pd.DataFrame:
    """Every pair of groups, ranked by how far apart the score puts them.

    `separation` is |AUC - 0.5|: direction is not the question here, distance
    is. Pairs smaller than `min_size` a side are dropped, because with two
    observations each a perfect AUC costs nothing and would sit at the top of
    the table meaning nothing. `n_a` and `n_b` are kept in the output so a
    reader can discount the small ones without re-deriving them.
    """
    v = np.asarray(values, dtype=float)
    g = pd.Series(list(groups))
    ok = np.isfinite(v)
    v, g = v[ok], g[ok.nonzero()[0]].reset_index(drop=True)
    rows = []
    for a, b in itertools.combinations(sorted(g.dropna().unique()), 2):
        va, vb = v[(g == a).to_numpy()], v[(g == b).to_numpy()]
        if len(va) < min_size or len(vb) < min_size:
            continue
        rows.append({"a": a, "b": b, "n_a": len(va), "n_b": len(vb),
                     "auc": auc(np.r_[np.ones(len(va)), np.zeros(len(vb))],
                                np.r_[va, vb])})
    out = pd.DataFrame(rows, columns=["a", "b", "n_a", "n_b", "auc"])
    out["separation"] = (out["auc"] - 0.5).abs()
    return out.sort_values("separation", ascending=False).reset_index(drop=True)


def rank_of_pair(pairs: pd.DataFrame, a: str, b: str) -> dict[str, float]:
    """Where one pair sits in the distribution of all of them.

    The number that matters: a cohort difference is only evidence of biology if
    it is bigger than the difference between two staining runs of the same
    tissue, and this says whether it is.
    """
    m = ((pairs["a"] == a) & (pairs["b"] == b)) | \
        ((pairs["a"] == b) & (pairs["b"] == a))
    if not m.any():
        raise KeyError(f"no pair ({a!r}, {b!r}) in the table; it may have been "
                       f"dropped for having too few slides on one side")
    row = pairs[m].iloc[0]
    sep = float(row["separation"])
    return {"auc": float(row["auc"]), "separation": sep,
            "rank": int((pairs["separation"] > sep).sum()) + 1,
            "of": int(len(pairs)),
            "median_separation": float(pairs["separation"].median()),
            "exceeded_by": int((pairs["separation"] > sep).sum())}


def within_group_contrast(frame: pd.DataFrame, group: str, label_col: str,
                          positive: str, negative: str,
                          score_col: str = "score",
                          group_col: str = BATCH_COL) -> dict[str, float]:
    """Inside ONE group (a staining batch), does the score tell two labels apart?

    Returns tile-level AUC and the per-slide medians on each side. Both matter
    and they answer different things: the tile AUC is the powered number, the
    slide medians are the one a reader can check by eye and the level at which
    a claim about animals would be made.
    """
    f = frame[frame[group_col] == group]
    if f.empty:
        raise KeyError(f"no rows with {group_col} == {group!r}")
    pos = f[f[label_col] == positive][score_col].to_numpy(dtype=float)
    neg = f[f[label_col] == negative][score_col].to_numpy(dtype=float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    sl = slide_summary(f, score_col=score_col, keep=(group_col, label_col))
    return {"group": group,
            "tile_auc": auc(np.r_[np.ones(len(pos)), np.zeros(len(neg))],
                            np.r_[pos, neg]) if len(pos) and len(neg) else float("nan"),
            "n_positive_tiles": int(len(pos)), "n_negative_tiles": int(len(neg)),
            "positive_slide_medians": sorted(
                np.round(sl[sl[label_col] == positive][score_col], 3).tolist()),
            "negative_slide_medians": sorted(
                np.round(sl[sl[label_col] == negative][score_col], 3).tolist())}


# ---- driver ---------------------------------------------------------------

def audit_frame(frame: pd.DataFrame, score_col: str = "score",
                batch_col: str = BATCH_COL) -> dict:
    """All four measurements over one frame. Returns tables, prints nothing."""
    from .evaluate import batch_separability
    f = frame[np.isfinite(frame[score_col].to_numpy(dtype=float))]
    sep = (batch_separability(f[score_col], f[batch_col])
           .rename("batch_separability").to_frame())
    sep = f.groupby(batch_col).agg(slides=("slide", "nunique"),
                                   tiles=(score_col, "size"),
                                   median_score=(score_col, "median")) \
           .join(sep)
    # How OFTEN the detector proposes anything is a second channel, and it can
    # move with the staining run even when the scores it returns do not: a
    # batch where nothing passes the vetoes contributes few tiles at any score.
    if {"scored", "n_scored_cells"} <= set(frame.columns):
        looked = frame[frame["scored"].astype(bool)]
        sep["tiles_with_proposal"] = looked.groupby(batch_col)["n_scored_cells"] \
            .apply(lambda v: float((v > 0).mean()))
    sep = sep.sort_values("batch_separability")
    slides = slide_summary(f, score_col=score_col)
    return {"per_batch": sep,
            "slides": slides,
            "variance": variance_components(slides[score_col], slides[batch_col]),
            "pairs": pairwise_group_auc(slides[score_col], slides[batch_col])}


def main(argv: list[str] | None = None) -> int:
    """python -m mashpath.train.audit --frame review_sets/study_frame.csv.gz \\
           --score-dir bal_chunks --score-dir bal_local

    `--score-dir` re-derives every slide's score from its stored cells under
    ONE ranking config before anything is measured. Without it the frame's
    stored scores are used as they are, which mixes whatever configs the runs
    happened to use -- see `review/rescore.py`.
    """
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--frame", default="review_sets/study_frame.csv.gz")
    p.add_argument("--score-dir", action="append", default=[],
                   help="directory of <slide>/ballooning/cells.csv; repeatable, "
                        "later ones win on a name collision")
    p.add_argument("--config", action="append",
                   default=["configs/core.yaml", "configs/ballooning.yaml"])
    p.add_argument("--out-prefix", default="outputs/ballooning_score")
    p.add_argument("--within", nargs=4, metavar=("BATCH", "COLUMN", "POS", "NEG"),
                   default=["2025-03-28_Celina 656D H&E", "diet", "nash", "chow"],
                   help="the within-batch contrast to report")
    p.add_argument("--pair", nargs=2, metavar=("BATCH_A", "BATCH_B"),
                   default=["2025-08-25HE", "2026-04-20_CCl4_HE"],
                   help="the batch pair to locate in the pairwise distribution")
    a = p.parse_args(argv)

    frame = pd.read_csv(a.frame)
    if a.score_dir:
        import glob
        import os

        from ..config import MashConfig
        from ..review.rescore import restore_frame
        paths = {}
        for d in a.score_dir:
            for c in glob.glob(f"{d}/*/ballooning/cells.csv"):
                paths[os.path.basename(os.path.dirname(os.path.dirname(c)))] = c
        cfg = MashConfig.from_yaml(a.config)
        frame = restore_frame(frame, paths, cfg.ballooning.ranking)

    r = audit_frame(frame)
    print("\n--- can the score say which staining run a TILE came from? ---")
    print(r["per_batch"].round(3).to_string())
    s = r["per_batch"]["batch_separability"]
    print(f"mean |AUC-0.5| {(s - 0.5).abs().mean():.3f}, "
          f"worst {(s - 0.5).abs().max():.3f} ({(s - 0.5).abs().idxmax()})")

    v = r["variance"]
    print(f"\n--- spread of the {v['n']} slide medians across {v['groups']} batches ---")
    print(f"between batches {v['between']:.2f}   within batches {v['within']:.2f}   "
          f"between/total {v['between_fraction']:.3f}")

    print("\n--- every batch pair, slide-median AUC ---")
    print(r["pairs"].to_string(index=False,
                               float_format=lambda x: f"{x:.3f}"))
    try:
        loc = rank_of_pair(r["pairs"], *a.pair)
        print(f"\n{a.pair[0]} vs {a.pair[1]}: AUC {loc['auc']:.3f}, "
              f"separation {loc['separation']:.3f} -- rank {loc['rank']} of "
              f"{loc['of']} pairs, median pair {loc['median_separation']:.3f}")
    except KeyError as exc:
        print(f"\npair not located: {exc}")

    batch, col, pos, neg = a.within
    try:
        w = within_group_contrast(frame, batch, col, pos, neg)
        print(f"\n--- inside one batch, stain held fixed: {pos} vs {neg} ---")
        print(f"{batch}: tile AUC {w['tile_auc']:.3f} "
              f"({w['n_positive_tiles']} vs {w['n_negative_tiles']} tiles)")
        print(f"  {pos} slide medians: {w['positive_slide_medians']}")
        print(f"  {neg} slide medians: {w['negative_slide_medians']}")
    except KeyError as exc:
        print(f"\nwithin-batch contrast unavailable: {exc}")

    r["per_batch"].to_csv(f"{a.out_prefix}_per_batch.csv")
    r["slides"].to_csv(f"{a.out_prefix}_slides.csv", index=False)
    r["pairs"].to_csv(f"{a.out_prefix}_batch_pairs.csv", index=False)
    print(f"\nwrote {a.out_prefix}_per_batch.csv, _slides.csv, _batch_pairs.csv")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
