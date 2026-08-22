"""Assemble a review set across slides: what a pathologist's hour buys.

Per-slide banding (`ballooning.rank.stratified_select`) decides which cells on
ONE slide are worth showing. This module decides how a fixed amount of
pathologist time is spent across the whole cohort, which is a different
question with different failure modes:

*All from one slide.* Per-slide sampling run over 50 slides produces 50x more
candidates than anyone will review, and taking the first N of them means one
slide supplies the entire confirmed set. Whatever is idiosyncratic about that
slide -- its staining run, its section thickness, its scanner session --
becomes the definition of ballooning.

*All from one batch.* Subtler and worse, because it survives per-slide
balancing. If every MASH slide was stained in one run and every control in
another, a classifier trained on the result can separate them perfectly on
stain colour without ever learning morphology, and its cross-validation
accuracy will look excellent.

*No repeats.* Without the same crop shown twice, there is no way to tell a
disappointing inter-rater kappa from ordinary within-rater noise, and no way
to answer "is the task hard, or is the review set bad".

*No overlap.* Two reviewers working disjoint sets produce twice the labels and
no measure of whether they agree, which is the one thing needed to know
whether the labels mean anything.

So: allocate by batch, then by slide, then by score band; duplicate a fraction
within each reviewer; overlap a fraction between reviewers; shuffle.

THE TIME BUDGET IS AN ASSUMPTION, not a measurement. `seconds_per_candidate`
defaults to a deliberately conservative guess. The review app records actual
per-candidate seconds in its verdict store, so after the first real session
this should be replaced with the measured median -- `mashpath review-stats`
prints it. Until then, treat the set size as provisional and expect to
over-provision.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import manifest as manifest_mod

# Extra rows added for measurement rather than for labels.
DUP_COLUMN = "duplicate_of"
BATCH_COLUMN = "batch"
REVIEWER_COLUMN = "assigned_to"


@dataclass
class ReviewSetConfig:
    """How to spend a review session."""

    # --- budget ---
    minutes_per_feature: float = 60.0
    # Conservative placeholder. See the module docstring: replace with the
    # measured median once a real session has been recorded. Too small and the
    # session overruns; too large and pathologist time is left unused, so it
    # errs high.
    seconds_per_candidate: float = 8.0

    # --- spread ---
    # No single slide may supply more than this share of the set, however many
    # candidates it proposed. Prevents one florid slide dominating.
    max_slide_fraction: float = 0.20
    # Nor may a single batch, when batches are known.
    max_batch_fraction: float = 0.60
    # Slides contributing fewer than this are dropped rather than included
    # token-ly; a slide with 2 candidates adds noise, not coverage.
    min_per_slide: int = 4

    # --- composition across the score range ---
    # Relative shares per band. The decision boundary is where the classifier
    # is weakest, so `mid` is weighted at least as heavily as `top`.
    band_weights: dict[str, float] = field(default_factory=lambda: {
        "top": 0.30, "mid": 0.30, "low": 0.20, "vetoed": 0.20,
    })

    # --- measurement ---
    # Share of each reviewer's set repeated later in their own queue, for
    # intra-rater consistency. 0.05 of 400 is 20 repeats, enough for a usable
    # agreement estimate without spending much of the hour on it.
    duplicate_fraction: float = 0.05
    # Share of the set shown to EVERY reviewer, for inter-rater kappa. With
    # two reviewers at 0.15, ~15% of candidates get two independent verdicts.
    overlap_fraction: float = 0.15

    reviewers: list[str] = field(default_factory=lambda: ["A", "B"])
    seed: int = 0

    @property
    def total_candidates(self) -> int:
        """How many candidates fit in the budget, before duplicates."""
        return max(1, int(self.minutes_per_feature * 60 / self.seconds_per_candidate))


def infer_batch(slide_id: str, batches: dict[str, str] | None = None) -> str:
    """Which staining/scanning batch a slide belongs to.

    An explicit mapping always wins. Failing that, the accession prefix is
    used -- `R25-264-1` -> `R25-264` -- which on this cohort corresponds to a
    submission and therefore usually to a staining run. That is a HEURISTIC and
    is reported as one: if two batches were stained together, or one batch was
    stained twice, only a mapping from the lab can say so.
    """
    if batches and slide_id in batches:
        return batches[slide_id]
    parts = slide_id.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else slide_id


def _quota(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Split `total` across weighted keys, largest-remainder so it sums."""
    raw = {k: total * w / sum(weights.values()) for k, w in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True):
        if sum(out.values()) >= total:
            break
        out[k] += 1
    return out


def assemble(
    manifests: Sequence[str | Path],
    cfg: ReviewSetConfig,
    batches: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one balanced review set from many per-slide manifests.

    Returns (rows, report). The report is not decoration -- it records what the
    set could NOT satisfy (a band that ran short, a batch that dominates
    anyway), and those caveats belong with the labels when someone later asks
    why a model behaves oddly on one batch.
    """
    rng = random.Random(cfg.seed)

    # slide -> band -> rows
    pools: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for path in manifests:
        for row in manifest_mod.read(path):
            row = dict(row)
            row["_manifest"] = str(path)
            slide = row.get("slide", "")
            band = row.get("review_band") or "unbanded"
            pools[slide][band].append(row)

    slides = sorted(s for s in pools if sum(len(v) for v in pools[s].values())
                    >= cfg.min_per_slide)
    dropped = sorted(set(pools) - set(slides))
    if not slides:
        return [], {"error": "no slide met min_per_slide", "dropped": dropped}

    by_batch: dict[str, list[str]] = defaultdict(list)
    for s in slides:
        by_batch[infer_batch(s, batches)].append(s)

    total = cfg.total_candidates
    # Batch quotas: even split, capped so no batch dominates.
    batch_quota = _quota(total, {b: 1.0 for b in by_batch})
    cap_batch = int(total * cfg.max_batch_fraction)
    for b in batch_quota:
        batch_quota[b] = min(batch_quota[b], cap_batch)

    cap_slide = max(cfg.min_per_slide, int(total * cfg.max_slide_fraction))
    picked: list[dict[str, Any]] = []
    shortfalls: list[str] = []

    for batch, members in sorted(by_batch.items()):
        slide_quota = _quota(batch_quota[batch], {s: 1.0 for s in members})
        for slide in members:
            n = min(slide_quota[slide], cap_slide)
            bands = _quota(n, cfg.band_weights)
            for band, want in bands.items():
                pool = list(pools[slide].get(band, []))
                if len(pool) < want:
                    shortfalls.append(
                        f"{slide}/{band}: wanted {want}, had {len(pool)}"
                    )
                take = rng.sample(pool, min(want, len(pool)))
                for r in take:
                    r[BATCH_COLUMN] = batch
                picked.extend(take)

    rng.shuffle(picked)
    report = {
        "requested": total,
        "selected": len(picked),
        "slides": len(slides),
        "batches": {b: len(v) for b, v in sorted(by_batch.items())},
        "dropped_slides": dropped,
        "shortfalls": shortfalls,
        "seconds_per_candidate_assumed": cfg.seconds_per_candidate,
        "estimated_minutes": round(
            len(picked) * cfg.seconds_per_candidate / 60, 1
        ),
    }
    return picked, report


def split_for_reviewers(
    rows: Sequence[dict[str, Any]], cfg: ReviewSetConfig
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Divide a review set between reviewers, with overlap and duplicates.

    Overlap rows go to everyone and are what inter-rater kappa is computed on.
    Duplicates are appended to a single reviewer's own queue, separated from
    their first appearance, so the second look is not obviously a second look.
    """
    rng = random.Random(cfg.seed + 1)
    who = list(cfg.reviewers) or ["A"]
    items = list(rows)
    rng.shuffle(items)

    n_overlap = int(len(items) * cfg.overlap_fraction)
    overlap, rest = items[:n_overlap], items[n_overlap:]

    out: dict[str, list[dict[str, Any]]] = {r: [] for r in who}
    for i, row in enumerate(rest):
        out[who[i % len(who)]].append(dict(row))
    for r in who:
        out[r].extend(dict(x) for x in overlap)

    for r in who:
        for row in out[r]:
            row[REVIEWER_COLUMN] = r
            row.setdefault(DUP_COLUMN, "")

        n_dup = int(len(out[r]) * cfg.duplicate_fraction)
        if n_dup:
            dups = [dict(x) for x in rng.sample(out[r], min(n_dup, len(out[r])))]
            for d in dups:
                # Same candidate_id on purpose: the verdict store keys on it
                # and counts the repeat via `pass_index`. Marked so analysis
                # can separate deliberate repeats from accidental ones.
                d[DUP_COLUMN] = d["candidate_id"]
            rng.shuffle(out[r])
            # Repeats go in the back half, so the two looks are far apart.
            half = len(out[r]) // 2
            tail = out[r][half:] + dups
            rng.shuffle(tail)
            out[r] = out[r][:half] + tail
        else:
            rng.shuffle(out[r])

    report = {
        "reviewers": who,
        "overlap_candidates": n_overlap,
        "per_reviewer": {r: len(v) for r, v in out.items()},
        "duplicates_per_reviewer": {
            r: sum(1 for x in v if x.get(DUP_COLUMN)) for r, v in out.items()
        },
        "estimated_minutes_each": {
            r: round(len(v) * cfg.seconds_per_candidate / 60, 1)
            for r, v in out.items()
        },
    }
    return out, report


def write_sets(
    per_reviewer: dict[str, list[dict[str, Any]]],
    out_dir: str | Path,
    feature: str,
) -> dict[str, Path]:
    """Write one manifest per reviewer. Returns reviewer -> path.

    Crops are NOT copied. Every row keeps the `image`/`overlay` path from its
    source manifest, and the app resolves them against the source package, so a
    candidate shown to two reviewers is the same file on disk rather than two
    copies that could drift.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for reviewer, rows in per_reviewer.items():
        clean = [{k: v for k, v in r.items() if not k.startswith("_")}
                 for r in rows]
        p = out_dir / f"{feature}_{reviewer}.csv"
        manifest_mod.write(p, clean)
        paths[reviewer] = p
    return paths


def describe(rows: Sequence[dict[str, Any]]) -> str:
    """What a review set actually contains. Print before handing it over."""
    if not rows:
        return "empty review set"
    by_band: dict[str, int] = defaultdict(int)
    by_slide: dict[str, int] = defaultdict(int)
    by_batch: dict[str, int] = defaultdict(int)
    for r in rows:
        by_band[r.get("review_band") or "unbanded"] += 1
        by_slide[r.get("slide", "?")] += 1
        by_batch[r.get(BATCH_COLUMN, "?")] += 1

    n = len(rows)
    lines = [f"{n} candidates"]
    lines.append("  band:  " + ", ".join(
        f"{k}={v} ({v / n:.0%})" for k, v in sorted(by_band.items())))
    lines.append("  batch: " + ", ".join(
        f"{k}={v} ({v / n:.0%})" for k, v in sorted(by_batch.items())))
    top = sorted(by_slide.items(), key=lambda kv: -kv[1])[:5]
    lines.append(f"  slides: {len(by_slide)}, largest contributors: " +
                 ", ".join(f"{k}={v}" for k, v in top))
    worst = top[0][1] / n if top else 0
    if worst > 0.30:
        lines.append(f"  WARNING: one slide supplies {worst:.0%} of the set")
    return "\n".join(lines)
