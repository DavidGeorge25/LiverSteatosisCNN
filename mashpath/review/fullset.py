"""One package: every slide, mark every ballooned cell, then grade the section.

Design decisions that matter, and why.

GROUPED BY SLIDE, not shuffled across slides. The section grade has to be given
after seeing that section, so the fields of a section must arrive together. The
cost is that she will recognise a section's character partway through it; the
mitigation is that sections arrive in random order under opaque ids, so she
never knows which cohort she is in.

UNIFORM SAMPLING, 100% OF IT. NASH-CRN's few-vs-many is a claim about
PREVALENCE, so the fields a grade is based on must be an unbiased sample of the
section. There is no enriched stratum and no way to ask for one -- see the note
on ENRICHMENT below.

BALANCED ACROSS STAINING BATCHES. Cohort and staining batch are one-to-one in
this collection, so anything a model learns from stain is indistinguishable
from anything it learns from biology. Nothing at the sampling stage can undo
that, but spreading the reviewer's time evenly over batches is the one lever
there is: it stops a single batch owning the majority of the labels. Fields are
therefore allocated per BATCH first and split evenly among that batch's slides,
rather than allocated per slide.

TIME IS THE BINDING CONSTRAINT, NOT COVERAGE. The whole draw is sized from a
wall-clock budget at an assumed seconds-per-field, and when more slides arrive
the fields-per-slide falls rather than the total rising. More slides sampled
shallowly beats fewer sampled deeply here, because batch diversity is what this
collection is short of and depth within a section is not.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from ..core.slide import Slide

# --- the time budget -------------------------------------------------------
#
# Every number below is an estimate and one of them is barely that. 13 s/field
# is extrapolated, not measured: the earlier binary tile task was estimated at
# 11 s and this one asks for a circle around each positive cell on top of the
# same yes/no search. The warm-up package exists to replace it with a measured
# number before the main study is committed to.
FIELD_SECONDS = 13.0
GRADE_SECONDS = 45.0          # reading a section montage and choosing 0/1/2

# Target, not ceiling. The ceiling discussed was 5 h; 4 h is what was scoped,
# and the difference is deliberate headroom. If 13 s/field turns out to be 17,
# a set built to 4.0 h lands at ~5.1 h and a set built to 5.0 h lands at ~6.4 h.
# Building to the ceiling at an unvalidated rate spends someone else's evening.
BUDGET_HOURS = 4.0

# No staining batch may own more than this share of the fields.
#
# Cohort and batch are one-to-one here, so a batch that owns most of the labels
# also owns most of one class, and "this stain is negative" becomes the cheapest
# rule a model can learn. Enforced by taking FEWER SLIDES from the dominant
# batch rather than fewer fields per slide, because every section has to rest on
# the same number of fields -- see EQUAL DEPTH below.
MAX_BATCH_SHARE = 0.60

# ...but 60% is only a meaningful bound when there are two batches. With nine it
# can never bind, and the cap silently stops doing anything at exactly the point
# the collection gets interesting: CCl4 came back from the fetch holding 24.5%
# of the slides, more than twice an even share, without tripping a thing.
#
# So the effective cap is the STRICTER of the absolute floor and "no batch may
# hold more than twice an even share". Nine batches -> 22.2%. Two batches ->
# 100%, which falls back to the 0.60 floor. The rule now tightens as batches are
# added instead of loosening.
RELATIVE_BATCH_CAP = 2.0

# Fields per section, floor and ceiling.
#
# EQUAL DEPTH: every section gets the SAME number, and that is a requirement
# rather than a simplification. The section grade is a prevalence judgement, so
# the chance of meeting a ballooned cell at all rises with the number of fields
# behind the grade. If MASH sections rested on 24 fields and CCl4 sections on 8,
# part of any grade difference between them would be sampling depth -- and since
# cohort is confounded with batch, that artefact would be confounded with the
# thing under test. So depth is held constant and balance is bought with slide
# counts instead.
#
# The floor is what a NASH-CRN grade can rest on: below about eight uniform
# fields the sampling noise is larger than the distance between grade 1 and
# grade 2. The ceiling keeps one section to roughly five minutes and its montage
# to a single screen, so the grade is given from fields she can still see.
MIN_PER_SECTION = 8
MAX_PER_SECTION = 24

# Repeats for intra-rater agreement, as a fraction of a section's unique fields.
#
# They sit at the END of their own section's block, which is a compromise the
# section-grouped design forces: a repeat cannot be moved an hour away without
# moving it into a different section, and a field from another section would
# then be sitting in a montage it does not belong to. The gap is therefore
# minutes, not an hour, and the resulting kappa is an UPPER bound on her
# true intra-rater agreement. Stated here so it is not discovered later.
REPEAT_FRACTION = 0.10

# Slides that are in the study whatever the batch balance says.
#
# R22-354 is the only batch holding a within-batch treatment contrast (chow vs
# NASH, same day, same hands, same stain) and the only chow anywhere in the
# 260-slide collection. Every other cohort comparison available here is
# perfectly aliased with its staining batch; this one is not, which makes these
# eight slides worth more per slide than anything else in the set.
ALWAYS_INCLUDE = ("R22-354",)

TILE_PX, TILE_Q = 512, 78

# The context panel is the one that carries "enlarged RELATIVE TO its
# neighbours", so it is not the place to save bytes. It had been rendering at
# 512 px / q66 -- a 3x3 tile region (761 um) squeezed into the same pixel count
# as the 253.6 um field, i.e. a third of the field's magnification at a lower
# quality, and well below the 1024 px / q90 the tile design specifies. 768 at
# the field's own quality is the measured middle: it takes the package from
# 106 MB to ~175 MB, where 1024 would take it to ~236 MB for a panel that is
# inherently a downsample of the field beside it.
CTX_PX, CTX_Q = 768, 78

# ENRICHMENT: deliberately absent, and this is a result rather than an omission.
#
# An earlier draw took a minority stratum from the detector's top decile, on the
# assumption that the score at least correlates with ballooning. Measured across
# all 70 scored slides under one consistent config, it does the opposite:
#
#     chow            3.09   <- normal liver scores HIGHEST
#     NASH (R22-354)  2.90
#     CCl4            2.89
#     other           2.65
#     MASH            1.91   <- the cohort that should score highest is lowest
#
# MASH vs CCl4 comes out at AUC 0.085 -- not "no signal" but a strong signal
# pointing the wrong way. Enriching on it spends her attention worse than chance
# would. There is consequently no `enriched` parameter to set: a stratum that
# cannot be requested cannot quietly come back in a later build, and `key.csv`
# carrying an `enriched` row is a reliable sign that the package is stale.


def n_repeats(n_unique: int, fraction: float | None = None) -> int:
    """How many repeats a section of `n_unique` fields gets.

    The fraction is of PRESENTATIONS, not of unique fields -- "10% repeats"
    means one presentation in ten is a second look at something already seen.
    Taking 10% of the unique count instead gives 2 repeats in a 24-field
    section, which is 7.7% of what she actually sits through, and the intra-rater
    kappa is then computed on fewer pairs than the design claims.
    """
    fraction = REPEAT_FRACTION if fraction is None else fraction
    f = min(max(fraction, 0.0), 0.9)
    return max(1, round(n_unique * f / (1.0 - f)))


def stable_seed(*parts: object) -> int:
    """A seed that is the same in every process.

    `hash()` on a str is salted per interpreter, so `random_state=hash(name)`
    -- which this module used to do -- gave a different draw on every run. The
    package would then differ from its own sampling report.
    """
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def blind_id(slide: str, seed: int) -> str:
    h = hashlib.sha256(f"{seed}:full:{slide}".encode()).hexdigest()
    return "s" + h[:6]


def is_always_include(slide: str, always: Sequence[str] = ALWAYS_INCLUDE) -> bool:
    return any(tag in slide for tag in always)


# --- how many fields each slide gets ---------------------------------------

def plan_allocation(slides: pd.DataFrame, hours: float | None = None,
                    repeat_fraction: float | None = None,
                    min_per: int | None = None,
                    max_per: int | None = None,
                    field_seconds: float | None = None,
                    grade_seconds: float | None = None,
                    max_batch_share: float | None = None,
                    always: Sequence[str] = ALWAYS_INCLUDE,
                    seed: int = 20260819) -> tuple[dict[str, int], dict[str, Any]]:
    """Which slides are in, and how many fields each gets.

    `slides` is one row per slide with columns `slide` and `batch`.

    Two rules, in this order:

    1. Every section gets the SAME number of fields, so that a grade from one
       section rests on as much evidence as a grade from any other.
    2. Batch shares are levelled by dropping slides from over-represented
       batches -- never by giving them shallower sections, which rule 1 forbids,
       and never by dropping a must-include slide.

    Then the common depth is whatever the time budget affords, clamped to
    [min_per, max_per]. If even `min_per` does not fit, more slides are dropped:
    a section too thin to grade is worth less than one fewer section.
    """
    # Resolved here, not bound as default arguments. A wave fetch renders the
    # ceiling with `fullset.MAX_PER_SECTION` read at call time and finalizes
    # with this function; if this one had frozen the value at import, changing
    # the constant would make the two stages disagree about how many fields a
    # section has, and the finalize pass would ask for fields nobody rendered.
    hours = BUDGET_HOURS if hours is None else hours
    repeat_fraction = REPEAT_FRACTION if repeat_fraction is None else repeat_fraction
    min_per = MIN_PER_SECTION if min_per is None else min_per
    max_per = MAX_PER_SECTION if max_per is None else max_per
    field_seconds = FIELD_SECONDS if field_seconds is None else field_seconds
    grade_seconds = GRADE_SECONDS if grade_seconds is None else grade_seconds
    slides = slides.drop_duplicates("slide")[["slide", "batch"]]
    n_batches = slides["batch"].nunique() if len(slides) else 1
    max_batch_share = (min(MAX_BATCH_SHARE, RELATIVE_BATCH_CAP / max(n_batches, 1))
                       if max_batch_share is None else max_batch_share)
    # hours <= 0 means NO time cap: keep every slide and give each the full
    # depth. The reviewer is a collaborator working to the science rather than
    # to a clock, sections are presented in a seeded random order, and her file
    # is written after every answer -- so stopping at any point leaves a fair,
    # batch-balanced subset of a well-formed study, which is strictly better
    # than a study trimmed in advance to a guess about how long she will sit.
    uncapped = hours <= 0
    budget = hours * 3600.0
    inflate = 1.0 + repeat_fraction

    keep = list(slides["slide"])
    batch_of = dict(zip(slides["slide"], slides["batch"]))
    # Deterministic drop order: the same slides go every time this is rebuilt.
    droppable = sorted((s for s in keep if not is_always_include(s, always)),
                       key=lambda s: stable_seed(seed, "drop", s))

    def _drop_one() -> bool:
        """Remove one slide from the batch that currently has the most."""
        counts = pd.Series([batch_of[s] for s in keep]).value_counts()
        for b in counts.index:
            victim = next((s for s in droppable if batch_of[s] == b), None)
            if victim is not None:
                droppable.remove(victim)
                keep.remove(victim)
                dropped.append(victim)
                return True
        return False

    dropped: list[str] = []

    # 1. Batch share. With equal depth per section, share of fields == share of
    #    slides, so this is a slide-count question.
    def _worst_share() -> float:
        counts = pd.Series([batch_of[s] for s in keep]).value_counts()
        return counts.iloc[0] / len(keep)

    over_share = _worst_share()
    while len(keep) > 1 and _worst_share() > max_batch_share:
        if not _drop_one():
            break
    share_dropped = len(dropped)

    # 2. Budget. A section costs its grade plus its floor of fields.
    if uncapped:
        n_sections = len(keep)
        per = max_per
    else:
        floor_cost = grade_seconds + min_per * inflate * field_seconds
        max_sections = max(1, int(budget // floor_cost))
        while len(keep) > max_sections:
            if not _drop_one():
                break
        n_sections = len(keep)
        affordable = (budget - n_sections * grade_seconds) / field_seconds / inflate
        per = int(max(min_per, min(max_per, affordable // max(n_sections, 1))))
    per_slide = {s: per for s in keep}

    unique = per * n_sections
    presentations = n_sections * (per + n_repeats(per, repeat_fraction))
    counts = pd.Series([batch_of[s] for s in keep]).value_counts().to_dict()
    report = {
        "sections": n_sections,
        "fields_per_section": per,
        "unique_fields": unique,
        "presentations": presentations,
        "batch_slides": counts,
        "batch_share": {b: n / n_sections for b, n in counts.items()},
        "share_before": over_share,
        "max_batch_share": max_batch_share,
        "dropped": dropped,
        "dropped_for_share": share_dropped,
        "dropped_for_budget": len(dropped) - share_dropped,
        "hours": (presentations * field_seconds + n_sections * grade_seconds) / 3600.0,
        "budget_hours": None if uncapped else hours,
        "field_seconds": field_seconds,
    }
    return per_slide, report


def _spread(frame: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    """n tiles spread over the section, one per spatial cell before repeating."""
    if len(frame) <= n:
        return frame
    f = frame.copy()
    f["_c"] = (f["y"] // 2048).astype(int) * 10_000 + (f["x"] // 2048).astype(int)
    cells = list(f["_c"].unique())
    rng.shuffle(cells)
    pools = {c: list(f.index[f["_c"] == c]) for c in cells}
    for v in pools.values():
        rng.shuffle(v)
    picked: list[int] = []
    while len(picked) < n and any(pools.values()):
        for c in cells:
            if pools[c]:
                picked.append(pools[c].pop())
                if len(picked) >= n:
                    break
    if len(picked) < n:
        rest = [i for i in f.index if i not in set(picked)]
        rng.shuffle(rest)
        picked += rest[: n - len(picked)]
    return f.loc[picked].drop(columns=["_c"])


def plan_fields(frame: pd.DataFrame, seed: int = 20260819,
                per_slide: dict[str, int] | None = None,
                repeat_fraction: float | None = None,
                only_slides: Iterable[str] | None = None,
                exclude_slides: Iterable[str] | None = None,
                exclude_tiles: Iterable[tuple[str, int, int]] | None = None,
                **budget_kw) -> pd.DataFrame:
    """Choose every field the reviewer will see, slide by slide.

    Every field is a uniform draw over the section's tissue. `per_slide` may be
    given directly; otherwise it is planned from the time budget.
    """
    frame = frame.copy()
    if only_slides is not None:
        frame = frame[frame["slide"].isin(set(only_slides))]
    if exclude_slides:
        frame = frame[~frame["slide"].isin(set(exclude_slides))]
    if exclude_tiles:
        gone = set(exclude_tiles)
        mask = [(s, int(x), int(y)) not in gone
                for s, x, y in zip(frame["slide"], frame["x"], frame["y"])]
        frame = frame[mask]
    if frame.empty:
        raise ValueError("no tiles left to draw from")

    if per_slide is None:
        per_slide, _ = plan_allocation(frame[["slide", "batch"]], seed=seed, **budget_kw)

    out: list[pd.DataFrame] = []
    for name, grp in frame.groupby("slide"):
        n = per_slide.get(name)
        if not n:
            continue
        rng = random.Random(f"{seed}:{name}")
        sel = _spread(grp, n, rng).assign(sample_type="uniform")
        sel = sel.sample(frac=1.0, random_state=stable_seed(seed, "order", name))

        # Repeats sit at the end of the same section's block, as far from their
        # original as a section-grouped design allows.
        n_rep = n_repeats(len(sel), repeat_fraction)
        reps = sel.sample(min(n_rep, len(sel)),
                          random_state=stable_seed(seed, "repeat", name))
        reps = reps.assign(sample_type="repeat")
        sel = pd.concat([sel, reps])
        sel["slide"] = name
        out.append(sel.reset_index(drop=True))
    if not out:
        raise ValueError("no slide received any fields")
    return pd.concat(out, ignore_index=True)


def key_tiles(package_dir: str | Path) -> set[tuple[str, int, int]]:
    """The (slide, x, y) a built package already spent, for disjointness."""
    kp = Path(package_dir) / "key.csv"
    if not kp.exists():
        return set()
    k = pd.read_csv(kp)
    return {(s, int(x), int(y)) for s, x, y in zip(k["slide"], k["x"], k["y"])}


def key_slides(package_dir: str | Path) -> set[str]:
    kp = Path(package_dir) / "key.csv"
    if not kp.exists():
        return set()
    return set(pd.read_csv(kp)["slide"])


def choose_warmup_slides(frame: pd.DataFrame, n: int = 3,
                         seed: int = 20260819,
                         prefer: Iterable[str] = (),
                         always: Sequence[str] = ALWAYS_INCLUDE) -> list[str]:
    """n slides to reserve for the warm-up, one per batch, smallest batch first.

    Held out of the main study entirely rather than sampled disjointly from the
    same sections: if she has already worked through part of a section in the
    warm-up, her grade for that section in the main study is no longer a first
    impression of it. Three slides is a cheap price for that.

    `prefer` is the pool the batch cap is going to discard anyway. Taking the
    warm-up from there first makes it very nearly free -- those slides were
    never going to be labelled. The exception is the scarcest batch, which is
    visited first precisely so the warm-up is not three sections of the one
    cohort that is least likely to contain a ballooned cell: a warm-up she can
    complete without ever meeting a positive cannot tell us whether the images
    are diagnostic.

    Never a must-include slide -- those are in the main study by definition.
    """
    prefer = set(prefer)
    pool = frame.drop_duplicates("slide")[["slide", "batch"]]
    pool = pool[~pool["slide"].map(lambda s: is_always_include(s, always))]
    by_batch: dict[str, list[str]] = {}
    for b, grp in pool.groupby("batch"):
        # Discardable slides first, then a stable shuffle within each group.
        by_batch[b] = sorted(grp["slide"],
                             key=lambda s: (s not in prefer,
                                            stable_seed(seed, "warmup", s)))
    order = sorted(by_batch, key=lambda b: (len(by_batch[b]), b))
    picked: list[str] = []
    # One pass, one slide per batch, scarcest batch first -- so with three slots
    # and two batches the scarce one is represented exactly once rather than
    # twice, which is what a second round-robin pass would do.
    for b in order:
        if len(picked) >= n:
            break
        if by_batch[b]:
            picked.append(by_batch[b].pop(0))
    # Remaining slots come out of the discard pool, so they cost nothing.
    rest = [s for b in order for s in by_batch[b] if s in prefer]
    rest += [s for b in order for s in by_batch[b] if s not in prefer]
    picked += rest[: max(0, n - len(picked))]
    return picked


def section_order(fields: pd.DataFrame, seed: int = 20260819) -> list[str]:
    """Presentation order: batches interleaved, random within each batch.

    A plain shuffle is unbiased but has variance, and variance is expensive here
    because she may stop at any point. In the 146-section draw a plain shuffle
    put only ONE of the four chow sections in the first half -- and chow is the
    only normal liver in the entire 260-slide collection, the tissue the whole
    specificity claim rests on. Losing three-quarters of it to a coin flip is
    not a risk worth running for the sake of purity.

    Interleaving is stratification, not steering: it leaves the expected
    composition of any prefix exactly where a shuffle put it and removes the
    unlucky draws. It also helps the blinding rather than hurting it, because
    consecutive sections now come from DIFFERENT staining batches, so she never
    works through a run of one cohort's characteristic appearance.
    """
    rng = random.Random(f"{seed}:order")
    by_batch: dict[str, list[str]] = {}
    for slide, batch in (fields.drop_duplicates("slide")[["slide", "batch"]]
                         .itertuples(index=False)):
        by_batch.setdefault(batch, []).append(slide)
    for v in by_batch.values():
        rng.shuffle(v)
    # Rarest batch first within each round, so an 8-slide batch is spread over
    # the whole order rather than exhausted in the first few rounds.
    batches = sorted(by_batch, key=lambda b: (len(by_batch[b]), b))
    out: list[str] = []
    while any(by_batch.values()):
        for b in batches:
            if by_batch[b]:
                out.append(by_batch[b].pop())
    return out


def build(frame_csv: str | Path, slide_dirs: list[str], out_dir: str | Path,
          seed: int = 20260819, verbose: bool = True, append: bool = False,
          per_slide: dict[str, int] | None = None,
          only_slides: Iterable[str] | None = None,
          exclude_slides: Iterable[str] | None = None,
          exclude_tiles: Iterable[tuple[str, int, int]] | None = None,
          image_root: str | Path | None = None,
          **budget_kw) -> Path:
    """Render every field the reviewer will see.

    `append` keeps whatever key.csv already holds, so a study can be built wave
    by wave as slides are fetched and deleted again -- 65 GB of slides passing
    through a few GB of disk. Without it, wave two would silently discard the
    fields rendered in wave one.
    """
    import cv2

    frame = pd.read_csv(frame_csv)
    out_dir = Path(out_dir)
    # One render pool, many keys. The warm-up draws different fields from the
    # same slides as the main study, and after a wave fetch the slides are gone
    # -- so its images have to come from wherever the wave put them, not from a
    # directory of its own that nothing ever rendered into.
    image_root = Path(image_root) if image_root else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (image_root / "images").mkdir(parents=True, exist_ok=True)
    (image_root / "context").mkdir(parents=True, exist_ok=True)

    paths = {p.stem: p for d in slide_dirs for p in Path(d).glob("*.svs")}
    fields = plan_fields(frame, seed, per_slide=per_slide, only_slides=only_slides,
                         exclude_slides=exclude_slides, exclude_tiles=exclude_tiles,
                         **budget_kw)

    order = section_order(fields, seed)
    pos = {s: i + 1 for i, s in enumerate(order)}

    key: list[dict[str, Any]] = []
    absent: list[str] = []
    for name in order:
        sel = fields[fields["slide"] == name]
        sid = blind_id(name, seed)

        # Work out what this slide needs BEFORE opening it. A finalize pass
        # after a wave fetch has every image on disk and no SVS anywhere --
        # the slides were deleted to make room for the next wave -- so opening
        # the slide has to be conditional on something actually being missing,
        # not on the slide happening to be present.
        plan = []
        need = False
        for i, r in enumerate(sel.itertuples(), 1):
            x, y, size = int(r.x), int(r.y), int(r.size)
            stem = f"{x:06d}_{y:06d}"
            ip = image_root / "images" / f"{sid}_{stem}.jpg"
            cp = image_root / "context" / f"{sid}_{stem}.jpg"
            plan.append((i, r, x, y, size, stem, ip, cp))
            need = need or not ip.exists() or not cp.exists()
        if need and name not in paths:
            absent.append(name)
            if verbose:
                print(f"  {name}: no slide file and no rendered fields, skipped")
            continue

        sl = Slide(str(paths[name]), None) if need else None
        try:
            dims = sl.dimensions if sl is not None else None
            for i, r, x, y, size, stem, ip, cp in plan:
                fid = f"{sid}_f{i:02d}"
                if not ip.exists():
                    core = sl.read_region((x, y), 0, (size, size))
                    core = cv2.resize(core, (TILE_PX, TILE_PX), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(ip), cv2.cvtColor(core, cv2.COLOR_RGB2BGR),
                                [int(cv2.IMWRITE_JPEG_QUALITY), TILE_Q])
                # Checked separately from the core tile. One guard over both
                # meant that changing the context's size or quality re-rendered
                # nothing at all, because the core was already on disk.
                if not cp.exists():
                    side = size * 3
                    cx = max(0, min(x - size, dims[0] - side))
                    cy = max(0, min(y - size, dims[1] - side))
                    ctx = sl.read_region((cx, cy), 0, (side, side))
                    ctx = cv2.resize(ctx, (CTX_PX, CTX_PX), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(cp), cv2.cvtColor(ctx, cv2.COLOR_RGB2BGR),
                                [int(cv2.IMWRITE_JPEG_QUALITY), CTX_Q])
                key.append(dict(field_id=fid, blind_id=sid, present_order=pos[name],
                                panel=i, slide=name, cohort=r.cohort, batch=r.batch,
                                diet=getattr(r, "diet", "unknown"),
                                split=getattr(r, "split", ""), sample_type=r.sample_type,
                                x=x, y=y, size=size, score=getattr(r, "score", ""),
                                image=f"images/{sid}_{stem}.jpg",
                                context=f"context/{sid}_{stem}.jpg"))
        finally:
            if sl is not None:
                sl.close()
        if verbose:
            print(f"  {name:24s} -> {sid}  {len(sel)} fields"
                  f"{'' if need else '  (already rendered)'}")
    if absent and verbose:
        print(f"\n{len(absent)} slide(s) in the frame produced no fields: "
              f"{', '.join(absent[:5])}")

    k = pd.DataFrame(key)
    kp = out_dir / "key.csv"
    if append and kp.exists():
        prev = pd.read_csv(kp)
        # A slide re-fetched and re-extracted REPLACES its old rows rather than
        # duplicating them, so an interrupted transfer can simply be re-run.
        if len(k):
            prev = prev[~prev["slide"].isin(set(k["slide"]))]
        k = pd.concat([prev, k], ignore_index=True)
        # present_order is assigned within a wave, so without renumbering here
        # every wave would restart at 1 and the sections would interleave.
        order = {s: i + 1 for i, s in enumerate(
            k.drop_duplicates("slide").sort_values("blind_id")["slide"])}
        k["present_order"] = k["slide"].map(order)
    k.to_csv(kp, index=False)
    write_report(k, out_dir / "sampling_report.md")
    if verbose:
        print(f"\n{k.blind_id.nunique()} sections, {len(k)} fields -> {out_dir}")
        print(k.sample_type.value_counts().to_string())
    return out_dir


def write_report(key: pd.DataFrame, path: str | Path) -> Path:
    """What was drawn, by batch / cohort / slide. Never shipped to the browser."""
    path = Path(path)
    n_field, n_sec = len(key), key["blind_id"].nunique()
    hours = (n_field * FIELD_SECONDS + n_sec * GRADE_SECONDS) / 3600.0
    L = [f"# Ballooning study — what was drawn", "",
         f"**{n_sec} sections · {n_field} presentations · "
         f"~{hours:.1f} h at {FIELD_SECONDS:.0f} s/field + {GRADE_SECONDS:.0f} s/grade**",
         ""]
    if "enriched" in set(key.get("sample_type", [])):
        L.append("> **WARNING: an `enriched` stratum is present. This build is stale.**\n")

    L += ["## By staining batch", "",
          "| batch | slides | fields | share |", "| --- | ---: | ---: | ---: |"]
    for b, grp in key.groupby("batch"):
        L.append(f"| {b} | {grp['slide'].nunique()} | {len(grp)} | "
                 f"{100 * len(grp) / n_field:.1f}% |")
    L += ["", "## By cohort", "",
          "| cohort | slides | fields | share |", "| --- | ---: | ---: | ---: |"]
    for c, grp in key.groupby("cohort"):
        L.append(f"| {c} | {grp['slide'].nunique()} | {len(grp)} | "
                 f"{100 * len(grp) / n_field:.1f}% |")
    if "diet" in key:
        L += ["", "## By diet label", "",
              "| diet | slides | fields |", "| --- | ---: | ---: |"]
        for d, grp in key.groupby("diet"):
            L.append(f"| {d} | {grp['slide'].nunique()} | {len(grp)} |")
    L += ["", "## By stratum", "",
          "| sample_type | fields |", "| --- | ---: |"]
    for t, grp in key.groupby("sample_type"):
        L.append(f"| {t} | {len(grp)} |")
    L += ["", "## Per slide", "",
          "| slide | blind id | batch | cohort | split | fields | repeats |",
          "| --- | --- | --- | --- | --- | ---: | ---: |"]
    for s, grp in key.sort_values(["batch", "slide"]).groupby("slide", sort=False):
        L.append(f"| {s} | {grp['blind_id'].iloc[0]} | {grp['batch'].iloc[0]} | "
                 f"{grp['cohort'].iloc[0]} | {grp['split'].iloc[0]} | {len(grp)} | "
                 f"{int((grp['sample_type'] == 'repeat').sum())} |")
    path.write_text("\n".join(L) + "\n")
    return path
