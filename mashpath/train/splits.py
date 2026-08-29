"""Train/validation splitting, by staining batch and by nothing else.

THE PROBLEM THIS EXISTS FOR. In this collection staining batch and experiment
are one-to-one: every MASH slide was cut and stained in one run, every CCl4
slide in another, and so on for all nine batches. A model therefore has two
routes to a good validation score — learn ballooning, or learn which staining
run a tile came from — and a split that puts tiles from the same batch on both
sides cannot tell them apart. It would report the second as if it were the
first, and it would keep doing so all the way to a figure.

A slide-level split does not fix this. Two slides from one batch were stained
together; holding one out tests generalisation to a new ANIMAL, not to a new
stain. Slide-level splitting is the right guard against a different failure
(the same tissue on both sides) and it is enforced here as well, but it is not
sufficient and it is the more tempting of the two mistakes because it looks
careful.

HOW THE WRONG THING IS MADE HARD. There is no function in this module that
takes a test fraction, a row count, a random seed, or a list of indices. The
only way to obtain a split is to name the batches that go in each fold; row
assignment is then derived from the `batch` column and cannot be overridden.
`train_test_split(tiles, test_size=0.2)` has no spelling here. `assert_no_leakage`
is called by `BatchSplit.assign` itself, so a leaking split raises at the point
it is made rather than at the point someone thinks to check.

WHAT IT COSTS. With two batches on disk, a batch-level split means training on
one cohort and validating on the other, and the validation number that comes
back will be poor and hard to interpret. That is the honest reading of what
these data support, and it is the reason for fetching the other seven batches.
A tidier number from a tile-level split would be an artefact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

import pandas as pd

BATCH_COL = "batch"
SLIDE_COL = "slide"

# Batches the model must never see during training.
#
# Dongdong, 2026-08-21: "we do have some test data for which we know the
# relevant information. We would prefer not to expose this data to the model
# during the training phase."
#
# That set is the only place a specificity claim can be made, because it is the
# only data where diet and treatment are known -- which makes it exactly the
# data that is most tempting to peek at, and worth the least once peeked at.
# Reserving it is therefore enforced rather than remembered: a reserved batch
# cannot reach a training fold through any function in this module, and
# `assert_not_trained_on` raises if one ever does.
#
# Add batch names here as the lab identifies them.
#
# 2026-08-29: `2025-03-28_Celina 656D H&E` (accession R22-354) is one of them,
# and it can be named without waiting for the lab, because the criterion above
# identifies it -- it is the ONLY batch in 260 slides whose filenames record
# diet, and all eight of its slides were stained in one run, so it is the only
# place where diet varies and stain does not. That made it the steatosis
# detector's external validation (chow vs NASH, AUC 1.000, p 0.0286, a 38x gap
# on labels the pipeline never saw -- docs/STEATOSIS_NINE_BATCHES.md §1), and
# it is the only normal liver in the collection. Eight slides is a cheap price
# for keeping the one measurement that is not the detector grading its own
# homework.
RESERVED_BATCHES: tuple[str, ...] = ("2025-03-28_Celina 656D H&E",)


class LeakySplitError(RuntimeError):
    """A fold assignment that puts one batch, or one slide, on both sides."""


def _require_columns(frame: pd.DataFrame, *cols: str) -> None:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(
            f"frame has no {missing} column(s); a batch-aware split cannot be "
            f"made without them. Batch comes from `cluster/.dripped` (see "
            f"`mashpath.review.tileset.batch_index`) and is not recoverable "
            f"from the SVS file, so it has to be carried through explicitly.")


@dataclass(frozen=True)
class BatchSplit:
    """Which staining batches belong to which fold.

    Frozen and defined by batch NAMES. There is deliberately no constructor
    that accepts row indices: a split object that could be built from rows
    could be built from a shuffle, and then nothing downstream could tell the
    difference between this and the mistake it exists to prevent.
    """

    folds: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "folds",
                           {k: tuple(v) for k, v in self.folds.items()})
        seen: dict[str, str] = {}
        for fold, batches in self.folds.items():
            if not batches:
                raise ValueError(f"fold {fold!r} has no batches in it")
            for b in batches:
                if b in seen:
                    raise LeakySplitError(
                        f"batch {b!r} is in both {seen[b]!r} and {fold!r}")
                seen[b] = fold

    @property
    def batches(self) -> tuple[str, ...]:
        return tuple(b for v in self.folds.values() for b in v)

    def fold_of(self, batch: str) -> str:
        for fold, batches in self.folds.items():
            if batch in batches:
                return fold
        raise KeyError(f"batch {batch!r} is in no fold of this split")

    def assign(self, frame: pd.DataFrame) -> pd.Series:
        """Fold label per row, derived from `batch`. Never from anything else."""
        _require_columns(frame, BATCH_COL)
        unknown = set(frame[BATCH_COL]) - set(self.batches)
        if unknown:
            raise KeyError(
                f"{sorted(unknown)} appear in the frame but in no fold. Every "
                f"batch must be placed explicitly -- silently dropping one is "
                f"how a cohort goes missing from an evaluation without anyone "
                f"noticing.")
        fold = frame[BATCH_COL].map(self.fold_of)
        assert_no_leakage(frame, fold)
        assert_not_trained_on(frame, fold, self.reserved)
        return fold

    @property
    def reserved(self) -> tuple[str, ...]:
        return self.folds.get("reserved", ())

    def subset(self, frame: pd.DataFrame, fold: str) -> pd.DataFrame:
        if fold not in self.folds:
            raise KeyError(f"no fold {fold!r}; have {sorted(self.folds)}")
        return frame[self.assign(frame) == fold]

    def summary(self, frame: pd.DataFrame, label_col: str | None = None
                ) -> pd.DataFrame:
        """Rows, slides and batches per fold — and the label balance if there
        is one, because a batch split can hand a fold a single class."""
        _require_columns(frame, BATCH_COL)
        f = frame.assign(_fold=self.assign(frame))
        agg = {"rows": (BATCH_COL, "size"), "batches": (BATCH_COL, "nunique")}
        if SLIDE_COL in frame.columns:
            agg["slides"] = (SLIDE_COL, "nunique")
        if label_col and label_col in frame.columns:
            agg["positives"] = (label_col, "sum")
        out = f.groupby("_fold").agg(**agg)
        if "positives" in out:
            out["prevalence"] = (out["positives"] / out["rows"]).round(4)
        return out


def assert_not_trained_on(frame: pd.DataFrame, fold: pd.Series,
                          reserved: Sequence[str] = ()) -> None:
    """Raise if a reserved batch reached a training fold.

    Separate from `assert_no_leakage` because it is a different failure: not a
    split that cannot measure what it claims, but data the lab asked us not to
    look at. It is checked on every assign() so that the answer does not depend
    on anyone remembering to ask.
    """
    reserved = tuple(reserved) or RESERVED_BATCHES
    if not reserved:
        return
    f = frame.assign(_fold=list(fold))
    trained = sorted(set(f.loc[f["_fold"] == "train", BATCH_COL]) & set(reserved))
    if trained:
        raise LeakySplitError(
            f"reserved batch(es) {trained} were assigned to train. These are "
            f"held out at the lab's request and are the only data where diet "
            f"and treatment are known -- training on them destroys the one "
            f"external validation this project has.")


def assert_no_leakage(frame: pd.DataFrame, fold: pd.Series) -> None:
    """Raise if any batch — or any slide — appears in more than one fold."""
    _require_columns(frame, BATCH_COL)
    keys = [BATCH_COL] + ([SLIDE_COL] if SLIDE_COL in frame.columns else [])
    f = frame.assign(_fold=list(fold))
    # Every level that leaks, not just the first one found. A slide straddling
    # folds usually makes its batch straddle too, and reporting only the batch
    # would send someone looking for the wrong cause.
    problems = []
    for key in keys:
        spread = f.groupby(key)["_fold"].nunique()
        bad = spread[spread > 1]
        if len(bad):
            problems.append(f"{len(bad)} {key}(s) in more than one fold: "
                            f"{sorted(bad.index)[:5]}")
    if problems:
        raise LeakySplitError(
            "; ".join(problems) + ". A split at either level lets the model "
            "score well by recognising where a tile came from rather than "
            "what is in it.")


def split_by_batch(frame: pd.DataFrame, *, validation: Sequence[str],
                   test: Sequence[str] = (),
                   reserved: Sequence[str] | None = None) -> BatchSplit:
    """Every batch not named goes to train. Naming is the only interface.

    Keyword-only on purpose: `split_by_batch(df, ["2025-08-25HE"])` should not
    be a thing that runs and quietly means something.

    `reserved` defaults to RESERVED_BATCHES and is subtracted from train before
    anything else happens, so the default behaviour of this function is already
    the safe one. Passing `reserved=()` explicitly is the only way to train on
    them, which is the point: it has to be a decision someone typed.
    """
    _require_columns(frame, BATCH_COL)
    present = list(dict.fromkeys(frame[BATCH_COL]))
    if len(present) < 2:
        raise ValueError(
            f"only {len(present)} staining batch in the frame ({present}). A "
            f"batch-level split needs at least two, and a split that is not at "
            f"batch level is not offered here. Fetch more batches -- see "
            f"docs/BALLOONING_TILESET.md section 6.")
    val, tst = tuple(validation), tuple(test)
    for name, group in (("validation", val), ("test", tst)):
        missing = [b for b in group if b not in present]
        if missing:
            raise KeyError(f"{name} names {missing}, which are not in the frame")
    if not val:
        raise ValueError("validation must name at least one batch")
    res = tuple(RESERVED_BATCHES if reserved is None else reserved)
    res = tuple(b for b in res if b in present)
    named = set(val) | set(tst) | set(res)
    train = tuple(b for b in present if b not in named)
    if not train:
        raise ValueError("every batch was assigned to validation/test/reserved; "
                         "nothing left to train on")
    folds = {"train": train, "validation": val}
    if tst:
        folds["test"] = tst
    if res:
        folds["reserved"] = res
    return BatchSplit(folds)


def leave_one_batch_out(frame: pd.DataFrame,
                        reserved: Sequence[str] | None = None
                        ) -> Iterator[tuple[str, BatchSplit]]:
    """One split per batch, each holding that batch out entirely.

    This is the evaluation that matches the claim people want to make about
    these models — "it works on staining it has not seen" — and it is the only
    one that can support it.
    """
    _require_columns(frame, BATCH_COL)
    present = list(dict.fromkeys(frame[BATCH_COL]))
    res = set(RESERVED_BATCHES if reserved is None else reserved) & set(present)
    # A reserved batch is neither trained on nor swept over: it is not a fold of
    # a development loop, it is the thing the development loop is measured
    # against once, at the end.
    sweep = [b for b in present if b not in res]
    if len(sweep) < 2:
        raise ValueError(f"leave-one-batch-out needs at least 2 non-reserved "
                         f"batches, got {sweep}")
    for b in sweep:
        yield b, split_by_batch(frame, validation=[b], reserved=sorted(res))
