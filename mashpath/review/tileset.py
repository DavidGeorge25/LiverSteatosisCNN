"""Tile-level labelling sets: the sampling frame, the draw, and the package.

WHY THIS EXISTS, and why it is not `review/select.py` with a different crop.

The candidate route asks a pathologist "is THIS cell ballooned?". That question
is only as good as the proposal, and on this material the proposal is not good
enough -- the score does not separate cohorts and the top band was rejected
wholesale. Every candidate-level label inherits that aim: if the detector never
proposes a class of ballooned cell, no amount of labelling reveals it, because
recall is invisible in a design that only ever shows its own proposals.

Tile-level binary labelling breaks that dependency. The question becomes "does
this 512 px field contain at least one ballooned hepatocyte?" -- a question
about the FIELD, which the detector does not get to define. It is the scheme
Heinemann et al. (2019) used to train a ballooning CNN on rodent NASH without
anyone circling a cell, and it is roughly an order of magnitude cheaper per
unit of information than cell-level annotation, which matters when the scarce
resource is one pathologist's afternoon.

The detector still earns a role, but a demoted one: it stratifies the draw.
Tiles it scored high and tiles it scored low both go in the set, so the labels
measure the detector rather than being defined by it. Where she disagrees with
it is the most informative signal available, and a set drawn only from its
top band could never contain a disagreement of the interesting kind.

WHAT THE SCORE MEANS HERE. A tile's score is the MAXIMUM cell score among the
veto-passing hepatocytes inside it -- max, not mean, because the label is
"contains at least one", and a mean would let one striking cell be diluted by
its hundred ordinary neighbours into the same number as a uniformly dull tile.
Tiles containing no veto-passing cell at all score -inf and land in the bottom
band, which is correct: that is the detector saying "nothing here".

Bands are cut at WITHIN-SLIDE percentiles. A slide-level offset in staining or
in section thickness shifts every cell score on that slide together, so a
cohort-wide percentile would sort slides, not tiles, and the top band would
fill up with whichever slides stained darkest.
"""

from __future__ import annotations

import re

import csv
import hashlib
import math
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

from ..core.slide import Slide
from ..core.tiling import Tile, iter_tiles
from ..core.tissue import detect_tissue
from ..core.viz import save_rgb

# Re-exported from `names` so a review package's filenames are defined in one
# dependency-free place -- see names.py for why that split exists.
from .names import FEATURE, MANIFEST_NAME, FRAME_NAME, VERDICTS_NAME  # noqa: F401

# Band names, ordered low to high. `anchor` is deliberately not a percentile
# band: it is an unconditional uniform draw over the frame, and it is the only
# stratum from which the true prevalence can be estimated. Without it the set
# reports the prevalence of its own enrichment and nothing else.
BAND_LOW = "low"
BAND_MID = "mid"
BAND_UPPER = "upper"
BAND_HIGH = "high"
BAND_ANCHOR = "anchor"
SCORE_BANDS = (BAND_LOW, BAND_MID, BAND_UPPER, BAND_HIGH)
ALL_BANDS = (*SCORE_BANDS, BAND_ANCHOR)


# The manifest the app reads carries NO score, band or split. Those live in
# sampling_frame.csv beside it. Splitting the files is not tidiness: the
# reviewer may reasonably open the manifest in a spreadsheet, and a `band`
# column reading "high" next to a tile she has not judged yet would anchor her
# exactly as effectively as printing the score on screen.
MANIFEST_COLUMNS = (
    "order", "tile_id", "slide", "image", "context",
    "tile_um", "context_um", "core_box", "dup_of",
)
FRAME_COLUMNS = (
    "tile_id", "slide", "cohort", "diet", "batch", "split", "band",
    "score", "score_pct", "n_scored_cells", "x", "y", "size",
    "tissue_fraction", "frame_tiles", "frame_patches", "dup_of", "order",
)


class TileSetError(RuntimeError):
    """A set that cannot be built, with the reason a human needs."""


# ---- what the reviewer will be shown --------------------------------------


@dataclass
class TilePlan:
    """Every knob of the draw. No number in this module is hardcoded.

    Defaults are the ones argued for in docs/BALLOONING_TILESET.md; change
    them there and in the YAML together, so the rationale travels with the
    value rather than being reconstructed from a diff.
    """

    n_presentations: int = 1900
    duplicate_fraction: float = 0.10
    test_slide_fraction: float = 0.25

    # Share of the UNIQUE tiles drawn from each band.
    band_shares: dict[str, float] = field(default_factory=lambda: {
        BAND_HIGH: 0.25, BAND_UPPER: 0.20, BAND_MID: 0.15,
        BAND_LOW: 0.15, BAND_ANCHOR: 0.25,
    })
    # Within-slide percentile windows, half-open [lo, hi).
    band_percentiles: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        BAND_LOW: (0.0, 30.0), BAND_MID: (30.0, 70.0),
        BAND_UPPER: (70.0, 90.0), BAND_HIGH: (90.0, 100.0),
    })

    # Tiles must be mostly tissue to be worth asking about. The pipeline's own
    # floor is 0.5, which is right for measuring area -- a half-empty tile
    # still contributes its half -- and wrong here: a field that is 50% glass
    # costs the same 10 seconds of her attention as a full one and carries half
    # the evidence. Raising it is not free: the model must be applied at the
    # same floor at inference, or it will meet fields at test time unlike any
    # it was trained on.
    min_tissue_fraction: float = 0.80

    # Diversity guards.
    max_slide_share: float = 0.06     # no slide may exceed this of the set
    cohort_caps: dict[str, float] = field(default_factory=lambda: {"CCl4": 0.20})
    min_tile_separation: int = 2      # Chebyshev distance in tile-grid units
    min_duplicate_gap: int = 120      # presentations between a tile and its repeat

    # Geometry.
    context_tiles: int = 3            # NxN tiles read around the core tile
    context_max_px: int = 1024
    context_jpeg_quality: int = 90

    seed: int = 20260818

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TilePlan":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        raw = raw.get("tileset", raw)
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(raw) - known)
        if unknown:
            # Same policy as the pipeline config: an unrecognised key is a
            # typo that would otherwise be silently ignored, and a silently
            # ignored sampling parameter is a set nobody can reproduce.
            raise TileSetError(f"unknown tileset key(s) {unknown}; known: {sorted(known)}")
        if "band_percentiles" in raw:
            raw["band_percentiles"] = {
                k: tuple(v) for k, v in raw["band_percentiles"].items()
            }
        return replace(cls(), **raw)

    # -- derived --
    @property
    def n_duplicates(self) -> int:
        return int(round(self.n_presentations * self.duplicate_fraction))

    @property
    def n_unique(self) -> int:
        return self.n_presentations - self.n_duplicates

    def validate(self) -> None:
        s = sum(self.band_shares.values())
        if not math.isclose(s, 1.0, abs_tol=1e-6):
            raise TileSetError(f"band_shares must sum to 1.0, got {s:.4f}")
        missing = sorted(set(ALL_BANDS) - set(self.band_shares))
        if missing:
            raise TileSetError(f"band_shares is missing {missing}")
        for b in SCORE_BANDS:
            if b not in self.band_percentiles:
                raise TileSetError(f"band_percentiles is missing {b!r}")
        if self.max_slide_share <= 0:
            raise TileSetError("max_slide_share must be > 0")


# ---- where a slide comes from ---------------------------------------------


@dataclass(frozen=True)
class SlideSource:
    """One slide, its pixels, and the provenance a stratified draw needs."""

    name: str
    path: Path
    cohort: str
    batch: str
    diet: str = "unknown"
    cells_csv: Path | None = None      # detector output; None = no score

    @property
    def has_score(self) -> bool:
        return self.cells_csv is not None and self.cells_csv.exists()


# Accession prefix -> the experimental model. Derived from the folder layout
# recorded in cluster/.dripped and the lab's naming; anything unlisted stays
# "unknown" rather than being guessed, because a wrong cohort label silently
# corrupts every per-cohort number downstream.
COHORT_BY_PREFIX = {
    "R25-264": "MASH",
    "R26-122": "CCl4",
    "R22-354": "ACLY656",
    "R22-063": "unknown",
    "R25-079": "unknown",
    "R25-151": "unknown",
    "R25-223": "unknown",
    "R25-345": "unknown",
    "R26-041": "unknown",
}


_ACCESSION = re.compile(r"^(R\d{2}-\d{3})")


def _prefix(name: str) -> str:
    """The accession, e.g. R25-264-1 -> R25-264, R22-354_16_ACLY656 -> R22-354.

    Matched as a pattern rather than split on "-". Splitting only worked when
    the character after the accession was another hyphen, so every
    underscore-separated name -- R25-079_1_HE, R25-151_4_HE and the whole of
    R22-354 -- returned itself and fell through to cohort "unknown". R22-354 is
    the batch that matters most here, so it was the most expensive one to lose.
    """
    cleaned = name.replace("--", "-")
    m = _ACCESSION.match(cleaned)
    if m:
        return m.group(1)
    bits = cleaned.split("-")
    return "-".join(bits[:2]) if len(bits) >= 2 else cleaned


def diet_from_name(name: str) -> str:
    """CHOW / NASH when the filename says so, else unknown.

    Only the R22-354 batch encodes diet in the filename. Everywhere else this
    returns "unknown", which is the honest answer and the reason the chow gap
    is as large as it is -- see the inventory in docs/BALLOONING_TILESET.md.
    """
    upper = name.upper()
    if "CHOW" in upper:
        return "chow"
    if "NASH" in upper:
        return "nash"
    return "unknown"


def batch_index(dripped: str | Path) -> dict[str, str]:
    """slide name -> staining batch, read from the cluster's slide list.

    The batch is the containing folder on the cluster, which is how the lab
    groups a staining run. It is not recoverable from the slide file itself,
    so this list is the only record of it -- which is why sampling "across all
    staining batches" needs a file that most of the pipeline never touches.
    """
    out: dict[str, str] = {}
    p = Path(dripped)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = Path(line).parts
        name = Path(line).stem
        # .../slides/<batch>/<slide>.svs -- a slide sitting directly in
        # slides/ has no batch folder and keeps whatever it already had.
        if len(parts) >= 2 and parts[-2] not in ("slides",):
            out.setdefault(name, parts[-2])
    return out


def discover_sources(
    slide_dirs: Sequence[str | Path],
    score_dirs: Sequence[str | Path] = (),
    dripped: str | Path = "cluster/.dripped",
) -> list[SlideSource]:
    """Every readable slide under `slide_dirs`, joined to its detector output.

    A slide with no detector output is kept, not dropped: it can still supply
    anchor-band tiles, and excluding it would concentrate the set on whichever
    slides happened to have been scored.
    """
    batches = batch_index(dripped)
    scores: dict[str, Path] = {}
    for d in score_dirs:
        for cells in sorted(Path(d).glob("*/ballooning/cells.csv")):
            scores[cells.parent.parent.name] = cells

    out: list[SlideSource] = []
    seen: set[str] = set()
    for d in slide_dirs:
        d = Path(d)
        paths = [d] if d.is_file() else sorted(d.glob("*.svs"))
        for p in paths:
            name = p.stem
            if name in seen:
                continue
            seen.add(name)
            out.append(SlideSource(
                name=name,
                path=p,
                cohort=COHORT_BY_PREFIX.get(_prefix(name), "unknown"),
                batch=batches.get(name, d.name),
                diet=diet_from_name(name),
                cells_csv=scores.get(name),
            ))
    return out


# ---- the sampling frame ---------------------------------------------------


def tile_scores(cells_csv: str | Path) -> pd.DataFrame:
    """Per-tile detector score from a cells.csv: max over veto-passing cells.

    `score` is non-null only for cells that survived the ranking vetoes, so
    "no scored cell in this tile" and "a low-scoring cell in this tile" are
    genuinely different states. They are kept apart here (`n_scored_cells`)
    and only collapsed at banding time.
    """
    df = pd.read_csv(cells_csv, usecols=["tile_x", "tile_y", "score"])
    scored = df.dropna(subset=["score"])
    g = scored.groupby(["tile_x", "tile_y"])["score"]
    out = pd.DataFrame({"score": g.max(), "n_scored_cells": g.size()}).reset_index()
    # Tiles present in the run but with nothing veto-passing: the detector
    # looked and proposed nothing. They belong in the frame at the bottom.
    covered = df[["tile_x", "tile_y"]].drop_duplicates()
    out = covered.merge(out, on=["tile_x", "tile_y"], how="left")
    out["n_scored_cells"] = out["n_scored_cells"].fillna(0).astype(int)
    return out.rename(columns={"tile_x": "x", "tile_y": "y"})


def _count_patches(frame: pd.DataFrame) -> int:
    """How many spatially disjoint patches the frame's tiles form.

    Flood fill over 4-connectivity on the tile grid. Reported, not enforced:
    a one-patch frame is a single strip of section and its within-slide
    percentiles are percentiles of that strip, which is a weaker claim than
    the same number computed over spaced windows.
    """
    if frame.empty:
        return 0
    size = int(frame["size"].iloc[0]) or 1
    cells = {(int(x) // size, int(y) // size)
             for x, y in zip(frame["x"], frame["y"])}
    seen: set[tuple[int, int]] = set()
    patches = 0
    for start in cells:
        if start in seen:
            continue
        patches += 1
        stack = [start]
        seen.add(start)
        while stack:
            cx, cy = stack.pop()
            for nb in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
    return patches


def build_frame(
    src: SlideSource, cfg: Any, plan: "TilePlan", verbose: bool = True
) -> pd.DataFrame:
    """The tiles of one slide that are eligible to be drawn, with their band.

    Restricted to the region the detector actually covered when there is one.
    Mixing scored tiles from the covered region with anchor tiles from the
    whole section would confound "band" with "where on the slide", which is
    worse than the narrower frame.

    How narrow depends on how the slide was measured, and the two sources here
    differ. The cluster run (`bal_chunks`) scored ONE contiguous 200-300 tile
    window per slide -- in raster order that is a single horizontal strip, so
    those slides contribute tiles from one band of tissue: one lobule row, one
    staining gradient, one focus plane. The local run (`bal_local`) scores
    several 100-tile windows at spaced offsets instead, which costs the same
    and covers the section. Prefer the latter when adding slides; the strip
    slides are kept because a narrower frame is still a real frame, but they
    are the weaker half of the set and `sampling_report` names which is which.
    """
    slide = Slide(str(src.path), cfg.slide)
    try:
        tissue = detect_tissue(slide, cfg.tissue)
        tiles = list(iter_tiles(slide, tissue, cfg.tiling, limit=cfg.limit))
        frame = pd.DataFrame([
            {"x": t.x, "y": t.y, "size": t.size,
             "tissue_fraction": round(t.tissue_fraction, 4)}
            for t in tiles
        ])
    finally:
        slide.close()

    if frame.empty:
        raise TileSetError(f"{src.name}: no tiles passed the tissue filter")

    n_before, best = len(frame), float(frame["tissue_fraction"].max())
    frame = frame[frame["tissue_fraction"] >= plan.min_tissue_fraction]
    if frame.empty:
        raise TileSetError(
            f"{src.name}: no tile reaches tissue_fraction >= "
            f"{plan.min_tissue_fraction} (best on this slide was {best:.2f})"
        )
    dropped = n_before - len(frame)

    if src.has_score:
        sc = tile_scores(src.cells_csv)
        frame = frame.merge(sc, on=["x", "y"], how="inner")
        if frame.empty:
            raise TileSetError(
                f"{src.name}: cells.csv tile coordinates do not intersect the "
                f"tile grid from {cfg.tiling.tile_size}px/"
                f"{cfg.tiling.min_tissue_fraction} tissue -- the detector was "
                "run with different tiling settings"
            )
    else:
        frame["score"] = np.nan
        frame["n_scored_cells"] = 0

    # Slide-qualified and coordinate-encoded, so a tile_id read off a verdict
    # row three months from now still resolves to a place on a slide without
    # needing the manifest that produced it.
    frame["tile_id"] = [
        f"{src.name}_x{int(x):06d}_y{int(y):06d}"
        for x, y in zip(frame["x"], frame["y"])
    ]
    # Carried as frame columns rather than written back onto SlideSource,
    # which is frozen on purpose: provenance should not acquire new values
    # partway through a build. These ride along into the draw (`rec = dict(row)`
    # keeps every column) and are read off the sequence by the report. They are
    # not in the manifest's column list, so they never reach the reviewer.
    frame["frame_tiles"] = len(frame)
    frame["frame_patches"] = _count_patches(frame)
    frame["slide"] = src.name
    frame["cohort"] = src.cohort
    frame["batch"] = src.batch
    frame["diet"] = src.diet
    frame["scored"] = src.has_score

    # -inf, not NaN: a tile the detector examined and rejected outright ranks
    # BELOW its lowest-scoring proposal, and must be able to reach the low band.
    filled = frame["score"].fillna(-np.inf)
    if src.has_score:
        finite = np.isfinite(filled)
        pct = pd.Series(np.zeros(len(frame)), index=frame.index)
        if finite.any():
            pct[finite] = filled[finite].rank(pct=True) * 100.0
        pct[~finite] = 0.0
        frame["score_pct"] = pct.round(2)
    else:
        frame["score_pct"] = np.nan

    if verbose:
        n_prop = int((frame["n_scored_cells"] > 0).sum())
        print(f"  {src.name:28s} {len(frame):5d} tiles in frame"
              f"  ({n_prop} with >=1 proposal, {dropped} dropped below "
              f"{plan.min_tissue_fraction:.0%} tissue)"
              + ("" if src.has_score else "   [no detector score]"))
    return frame


def assign_bands(frame: pd.DataFrame, plan: TilePlan) -> pd.DataFrame:
    """Label each tile with its within-slide percentile band."""
    frame = frame.copy()
    frame["band"] = ""
    for band in SCORE_BANDS:
        lo, hi = plan.band_percentiles[band]
        sel = (frame["score_pct"] >= lo) & (
            frame["score_pct"] < hi if hi < 100.0 else frame["score_pct"] <= hi
        )
        frame.loc[sel & frame["scored"], "band"] = band
    return frame


# ---- the draw -------------------------------------------------------------


def _rng(plan: TilePlan, salt: str) -> random.Random:
    """A seeded RNG per stage, so adding a stage cannot reshuffle the others."""
    h = hashlib.sha256(f"{plan.seed}:{salt}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _too_close(chosen: dict[str, list[tuple[int, int]]], slide: str,
               x: int, y: int, size: int, min_sep: int) -> bool:
    """True if this tile is within `min_sep` grid steps of one already taken.

    Twenty tiles from one lobule are twenty views of one piece of biology.
    Chebyshev distance on the tile grid, so diagonal neighbours count too.
    """
    if min_sep <= 0:
        return False
    for cx, cy in chosen.get(slide, ()):
        if max(abs(cx - x), abs(cy - y)) < min_sep * size:
            return True
    return False


def draw(frame: pd.DataFrame, plan: TilePlan, verbose: bool = True) -> pd.DataFrame:
    """Draw the unique tiles, relaxing tile separation only if it does not fit.

    `min_tile_separation` is the constraint most likely to bind, and it binds
    invisibly. A contiguous 300-tile detector window is roughly a 20x15 patch
    of the tile grid; requiring two grid steps between sampled tiles caps that
    slide at ~80 tiles in principle and ~50 by greedy draw. With few slides in
    the frame the set then comes up short for a geometric reason that looks
    nothing like a geometric reason in the output. Slides measured as several
    spaced windows rather than one strip relax this a long way -- the windows
    are far enough apart that separation only ever binds within a window.

    Worse, a short draw silently loosens the OTHER guards: the CCl4 cap is
    computed against the target, so a set that lands at 70% of target has a
    cohort share ~1.4x what was asked for. So rather than accept the shortfall,
    step the separation down until the set fits, and say which value was used.
    """
    plan.validate()
    for sep in range(plan.min_tile_separation, -1, -1):
        picked = _draw_at(frame, plan, sep, verbose=False)
        if not picked.attrs["shortfalls"]:
            break
    if verbose:
        for band in (BAND_ANCHOR, BAND_HIGH, BAND_UPPER, BAND_MID, BAND_LOW):
            got = int((picked["band"] == band).sum())
            want = int(round(plan.band_shares[band] * plan.n_unique))
            note = f"  (short {want - got})" if got < want else ""
            print(f"  band {band:7s} target {want:4d}  drew {got:4d}{note}")
        if sep != plan.min_tile_separation:
            print(f"  NOTE: tile separation relaxed {plan.min_tile_separation}"
                  f" -> {sep} grid steps; at {plan.min_tile_separation} the "
                  f"frame could not fill the set. More slides, not a bigger "
                  f"window per slide, is the fix.")
        if picked.attrs["shortfalls"]:
            b = picked.attrs["blocked"]
            worst = max(b, key=b.get) if any(b.values()) else "frame size"
            print(f"  SHORT: drew {len(picked)} of {plan.n_unique}. Tiles "
                  f"turned away: " + ", ".join(f"{k}={v}" for k, v in b.items()))
            print(f"         the binding constraint is {worst!r}. "
                  + {"slide_cap": "too few slides for this target -- add "
                                  "slides, or lower n_presentations.",
                     "cohort_cap": "one cohort is capped and the others cannot "
                                   "cover the rest -- add slides from the "
                                   "uncapped cohorts.",
                     "separation": "the scored window per slide is too small "
                                   "to space tiles this far apart.",
                     "frame size": "the frame has fewer tiles than the plan "
                                   "asks for."}[worst])
    return picked


def _draw_at(frame: pd.DataFrame, plan: TilePlan, min_sep: int,
             verbose: bool = True) -> pd.DataFrame:
    """One draw at a fixed tile separation.

    Round-robin over slides within each band rather than a single pooled
    sample. A pooled draw is proportional to how many tiles each slide
    contributes to the frame, which on this material means the two largest
    sections would supply a fifth of the set between them.
    """
    n_unique = plan.n_unique
    slides = sorted(frame["slide"].unique())
    # The cap can never sit below an even split -- 14 slides at a 6% ceiling
    # can supply 84% of the target, and the draw would come up short for a
    # reason that has nothing to do with the material. It also needs HEADROOM
    # above that split rather than exactly it: bands are filled one after
    # another, so a cap at the exact even share is spent by the early bands and
    # the late ones draw nothing (measured: `mid` and `low` came back empty
    # while every slide sat exactly at its cap). 1.4x lets slides that still
    # have material cover for slides that have run out, which is the whole
    # point of a round-robin. The configured share binds once there are enough
    # slides for it to be the looser of the two.
    even = math.ceil(1.4 * n_unique / max(len(slides), 1))
    slide_cap = max(even, int(round(plan.max_slide_share * n_unique)))
    # Allocated PER BAND, not once over the whole set. A single global ceiling
    # is spent by whichever bands are drawn first, so the capped cohort ends up
    # entirely inside them -- measured: every CCl4 tile landed in `anchor` and
    # `high`, none in `low`, `mid` or `upper`. That defeats the reason for
    # including CCl4 at all. Its hydropic degeneration is the ballooning mimic
    # we most need her opinion on, and we need it across the score range, not
    # only where the detector already shouted.
    band_targets = {b: int(round(plan.band_shares[b] * n_unique))
                    for b in ALL_BANDS}
    cohort_caps = {
        b: {c: int(round(f * band_targets[b])) for c, f in plan.cohort_caps.items()}
        for b in ALL_BANDS
    }

    taken: set[tuple[str, int, int]] = set()
    positions: dict[str, list[tuple[int, int]]] = {}
    per_slide: dict[str, int] = {s: 0 for s in slides}
    per_cohort: dict[str, int] = {}          # reset at the top of each band
    band_caps: dict[str, int] = {}           # the caps in force for this band
    picked: list[dict[str, Any]] = []

    # Which guard turned a tile away, counted so a short draw can name its
    # cause. A set that comes up short for a geometric reason and a set that
    # comes up short because one cohort is capped call for opposite fixes
    # (more slides vs a different plan), and they are indistinguishable from
    # the band totals alone.
    blocked: dict[str, int] = {"slide_cap": 0, "cohort_cap": 0, "separation": 0}

    def eligible(row: dict[str, Any]) -> bool:
        nonlocal per_cohort, band_caps
        key = (row["slide"], int(row["x"]), int(row["y"]))
        if key in taken:
            return False
        if per_slide[row["slide"]] >= slide_cap:
            blocked["slide_cap"] += 1
            return False
        cap = band_caps.get(row["cohort"])
        if cap is not None and per_cohort.get(row["cohort"], 0) >= cap:
            blocked["cohort_cap"] += 1
            return False
        if _too_close(positions, row["slide"], int(row["x"]),
                      int(row["y"]), int(row["size"]), min_sep):
            blocked["separation"] += 1
            return False
        return True

    def take(row: dict[str, Any], band: str) -> None:
        taken.add((row["slide"], int(row["x"]), int(row["y"])))
        positions.setdefault(row["slide"], []).append((int(row["x"]), int(row["y"])))
        per_slide[row["slide"]] += 1
        per_cohort[row["cohort"]] = per_cohort.get(row["cohort"], 0) + 1
        rec = dict(row)
        rec["band"] = band
        picked.append(rec)

    shortfalls: dict[str, int] = {}
    # Anchor first, then richest band down. Order matters because the earlier
    # bands spend the per-slide and per-cohort budget, and the anchor is the
    # one stratum whose scientific job -- estimating the true prevalence, and
    # reweighting the enriched strata back to it -- fails outright if it comes
    # up short. A banded stratum that is short only shifts the enrichment.
    for band in (BAND_ANCHOR, BAND_HIGH, BAND_UPPER, BAND_MID, BAND_LOW):
        target = band_targets[band]
        per_cohort = {}
        band_caps = cohort_caps[band]
        rng = _rng(plan, f"band:{band}:sep{min_sep}")
        if band == BAND_ANCHOR:
            pool = frame           # unconditional: this is the whole point
        else:
            pool = frame[frame["band"] == band]
        by_slide = {
            s: pool[pool["slide"] == s].sample(frac=1.0, random_state=rng.getrandbits(31)
                                               ).to_dict("records")
            for s in slides
        }
        order = list(slides)
        rng.shuffle(order)
        got = 0
        stalled = 0
        while got < target and stalled < len(order):
            stalled = 0
            for s in order:
                if got >= target:
                    break
                bucket = by_slide.get(s) or []
                while bucket:
                    row = bucket.pop()
                    if eligible(row):
                        take(row, band)
                        got += 1
                        break
                else:
                    stalled += 1
        if got < target:
            shortfalls[band] = target - got
        if verbose:
            note = f"  (short {target - got})" if got < target else ""
            print(f"  band {band:7s} target {target:4d}  drew {got:4d}{note}")

    out = pd.DataFrame(picked)
    if out.empty:
        raise TileSetError("the draw produced no tiles; check the frame and plan")
    out.attrs["shortfalls"] = shortfalls
    out.attrs["slide_cap"] = slide_cap
    out.attrs["cohort_caps"] = cohort_caps
    out.attrs["separation"] = min_sep
    out.attrs["blocked"] = blocked
    return out.reset_index(drop=True)


def is_test_slide(name: str, plan: TilePlan) -> bool:
    """Whether a slide is held out, decided from its NAME alone.

    Hashing the name rather than shuffling a list is what makes a pilot set and
    the full set that follows it safe to combine. A shuffle depends on which
    slides happened to be in the draw, so a slide that was train in a 14-slide
    pilot could come out test in a 45-slide set -- and its pilot labels would
    then be training data for a model evaluated on the same slide. With the
    name as the only input, every set built at this seed agrees for all time,
    including sets built months apart from different slide directories.
    """
    h = hashlib.sha256(f"{plan.seed}:split:{name}".encode()).hexdigest()
    return (int(h[:8], 16) % 10_000) / 10_000.0 < plan.test_slide_fraction


def assign_split(picked: pd.DataFrame, plan: TilePlan,
                 verbose: bool = True) -> pd.DataFrame:
    """Hold out whole SLIDES, not tiles.

    A tile-wise split lets a model memorise a slide's stain, focus and section
    thickness from its training tiles and recognise them in its test tiles,
    reporting a generalisation it does not have. Grouping by slide is the only
    split that answers the question we will want to ask -- "does this work on a
    slide it has never seen?".
    """
    picked = picked.copy()
    slides = sorted(picked["slide"].unique())
    test_slides = [s for s in slides if is_test_slide(s, plan)]
    picked["split"] = np.where(picked["slide"].isin(test_slides), "test", "train")

    if verbose:
        n_t = int((picked["split"] == "test").sum())
        print(f"  split: {len(test_slides)}/{len(slides)} test slide(s), "
              f"{n_t} test tiles, {len(picked) - n_t} train tiles")
        # A cohort with no test slide cannot be tested, and a hash split does
        # not guarantee one. Say so rather than letting it surface as an
        # empty subgroup in the evaluation months later.
        for cohort, grp in picked.groupby("cohort"):
            if not (grp["split"] == "test").any() and grp["slide"].nunique() > 1:
                print(f"  NOTE: cohort {cohort!r} has no held-out slide at "
                      f"seed {plan.seed} -- it cannot be evaluated separately")
    return picked


def add_duplicates(picked: pd.DataFrame, plan: TilePlan,
                   verbose: bool = True) -> pd.DataFrame:
    """Repeat ~`duplicate_fraction` of the tiles under a fresh id.

    A repeat gets its OWN tile_id with `dup_of` pointing back, rather than
    being the same id shown twice. That keeps two things apart which the
    verdict store would otherwise conflate: a deliberate repeat (measures
    intra-rater agreement) and the reviewer pressing back to correct herself
    (measures nothing, and must not count as a disagreement).
    """
    # Derived from what was ACTUALLY drawn, not from the plan's target. If a
    # band came up short the session is smaller than planned, and a duplicate
    # count fixed to the target would silently become a much larger share of
    # it -- 10% of her time is the thing being budgeted, not 190 tiles.
    f = plan.duplicate_fraction
    n_dup = int(round(len(picked) * f / (1.0 - f))) if f < 1.0 else 0
    if n_dup <= 0:
        picked = picked.copy()
        picked["dup_of"] = ""
        return picked

    rng = _rng(plan, "duplicates")
    rows: list[dict[str, Any]] = []
    chosen: set[str] = set()
    # Proportional to band so the repeat set spans the same range of material
    # the session does; a repeat set drawn only from easy tiles would report an
    # intra-rater agreement the hard tiles never earned.
    for band, grp in picked.groupby("band"):
        k = int(round(n_dup * len(grp) / len(picked)))
        idx = list(grp.index)
        rng.shuffle(idx)
        for i in idx[:k]:
            rows.append(picked.loc[i].to_dict())
            chosen.add(picked.loc[i, "tile_id"])
    # Per-band rounding lands a few either side of the target; top up from
    # whatever has not already been picked to repeat.
    spare = [i for i in picked.index if picked.loc[i, "tile_id"] not in chosen]
    rng.shuffle(spare)
    while len(rows) < n_dup and spare:
        rows.append(picked.loc[spare.pop()].to_dict())

    picked = picked.copy()
    picked["dup_of"] = ""
    dups = pd.DataFrame(rows[:n_dup])
    if not dups.empty:
        dups["dup_of"] = dups["tile_id"]
        dups["tile_id"] = dups["tile_id"] + "__r2"
    if verbose:
        print(f"  duplicates: {len(dups)} repeat presentations "
              f"({len(dups) / max(len(picked) + len(dups), 1):.0%} of the session)")
    return pd.concat([picked, dups], ignore_index=True)


def order_presentations(rows: pd.DataFrame, plan: TilePlan,
                        verbose: bool = True) -> pd.DataFrame:
    """Interleave into presentation order, repeats well away from originals.

    Also interleaves slides and cohorts: a session blocked by slide lets the
    reviewer calibrate to one section and carry that calibration through
    thirty consecutive tiles, which is a different task from judging each
    field on its own.
    """
    rng = _rng(plan, "order")
    uniq = rows[rows["dup_of"] == ""].copy()
    dups = rows[rows["dup_of"] != ""].copy()

    seq: list[dict[str, Any]] = uniq.sample(
        frac=1.0, random_state=rng.getrandbits(31)
    ).to_dict("records")

    # An original that lands in the last `min_duplicate_gap` positions cannot
    # have its repeat placed far enough away -- there is no room left. Rather
    # than let those pairs land 20 apart (which measures short-term memory, not
    # intra-rater agreement), move the original forward first, swapping it with
    # an ordinary tile. Only ~10% of tiles are originals, so this touches few
    # positions and leaves the rest of the order untouched and random.
    originals = set(dups["dup_of"])
    limit = max(0, len(seq) - plan.min_duplicate_gap)
    free = [i for i in range(limit) if seq[i]["tile_id"] not in originals]
    rng.shuffle(free)
    for i in range(limit, len(seq)):
        if seq[i]["tile_id"] in originals and free:
            j = free.pop()
            seq[i], seq[j] = seq[j], seq[i]

    placed, cramped = 0, 0
    for _, d in dups.iterrows():
        try:
            i = next(k for k, r in enumerate(seq) if r["tile_id"] == d["dup_of"])
        except StopIteration:                   # original was dropped upstream
            continue
        lo = i + plan.min_duplicate_gap
        if lo >= len(seq):
            cramped += 1
            pos = len(seq)
        else:
            pos = rng.randint(lo, len(seq))
        seq.insert(pos, d.to_dict())
        placed += 1

    out = pd.DataFrame(seq)
    out["order"] = range(1, len(out) + 1)
    if verbose:
        gaps = _duplicate_gaps(out)
        if gaps:
            print(f"  order: {len(out)} presentations, repeat gap "
                  f"min {min(gaps)} / median {int(np.median(gaps))}")
        if cramped:
            print(f"  NOTE: {cramped} repeat(s) fell closer than "
                  f"{plan.min_duplicate_gap} -- the session is too short to "
                  f"hold the gap for every pair. Lower min_duplicate_gap or "
                  "raise n_presentations if this is more than a handful.")
    return out.reset_index(drop=True)


def _duplicate_gaps(seq: pd.DataFrame) -> list[int]:
    pos = {r["tile_id"]: i for i, r in seq.iterrows()}
    return [i - pos[r["dup_of"]] for i, r in seq.iterrows()
            if r["dup_of"] and r["dup_of"] in pos]


# ---- writing the package --------------------------------------------------


def _context_window(x: int, y: int, size: int, n: int,
                    dims: tuple[int, int]) -> tuple[int, int, int]:
    """Top-left and side of an n*size window around the tile, clamped in-slide.

    Clamped rather than padded: openslide fills out-of-bounds reads with black,
    and a black L-shape beside the tissue is a distraction in every crop at the
    edge of a section. A clamped window puts the core tile off-centre instead,
    which the drawn box makes obvious.
    """
    w, h = dims
    side = min(n * size, w, h)
    cx = x + size // 2 - side // 2
    cy = y + size // 2 - side // 2
    cx = max(0, min(cx, w - side))
    cy = max(0, min(cy, h - side))
    return cx, cy, side


def render_slide(
    src: SlideSource, rows: pd.DataFrame, out_dir: Path, plan: TilePlan,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Write the core tile and its context view for every row of one slide.

    Opens the slide once for all of its tiles. Reading 1,900 tiles from 50
    slides one openslide handle at a time is most of the runtime otherwise.
    """
    import cv2

    images = out_dir / "images"
    context = out_dir / "context"
    images.mkdir(parents=True, exist_ok=True)
    context.mkdir(parents=True, exist_ok=True)

    slide = Slide(str(src.path), None)
    written: list[dict[str, Any]] = []
    try:
        mpp = slide.mpp_x
        dims = slide.dimensions
        # Unique geometry only: a repeat shows the identical picture, so it
        # points at the file its original already wrote.
        for (x, y, size), grp in rows.groupby(["x", "y", "size"], sort=False):
            x, y, size = int(x), int(y), int(size)
            base = grp.iloc[0]
            stem = base["dup_of"] or base["tile_id"]

            core = slide.read_region((x, y), 0, (size, size))
            save_rgb(images / f"{stem}.png", core)

            cx, cy, side = _context_window(x, y, size, plan.context_tiles, dims)
            ctx = slide.read_region((cx, cy), 0, (side, side))
            scale = min(1.0, plan.context_max_px / side)
            if scale < 1.0:
                ctx = cv2.resize(ctx, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)
            box = [round((x - cx) * scale), round((y - cy) * scale),
                   round(size * scale), round(size * scale)]
            cv2.imwrite(
                str(context / f"{stem}.jpg"),
                cv2.cvtColor(ctx, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), plan.context_jpeg_quality],
            )
            written.append({
                "x": x, "y": y, "size": size, "stem": stem,
                "tile_um": round(size * mpp, 1),
                "context_um": round(side * mpp, 1),
                "core_box": " ".join(str(v) for v in box),
            })
    finally:
        slide.close()
    if verbose:
        print(f"  {src.name:28s} wrote {len(written)} tile+context pairs")
    return written


def write_package(
    seq: pd.DataFrame, out_dir: str | Path, plan: TilePlan,
    sources: dict[str, SlideSource], cfg: Any, verbose: bool = True,
) -> Path:
    """Render every tile and write manifest.csv + sampling_frame.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    geom: dict[tuple[str, int, int], dict[str, Any]] = {}
    for name, grp in seq.groupby("slide", sort=True):
        src = sources[name]
        for rec in render_slide(src, grp, out_dir, plan, verbose):
            geom[(name, rec["x"], rec["y"])] = rec

    manifest_rows, frame_rows = [], []
    for _, r in seq.iterrows():
        g = geom[(r["slide"], int(r["x"]), int(r["y"]))]
        manifest_rows.append({
            "order": int(r["order"]),
            "tile_id": r["tile_id"],
            "slide": r["slide"],
            "image": f"images/{g['stem']}.png",
            "context": f"context/{g['stem']}.jpg",
            "tile_um": g["tile_um"],
            "context_um": g["context_um"],
            "core_box": g["core_box"],
            "dup_of": r["dup_of"],
        })
        frame_rows.append({
            "tile_id": r["tile_id"], "slide": r["slide"], "cohort": r["cohort"],
            "diet": r["diet"], "batch": r["batch"], "split": r["split"],
            "band": r["band"],
            "score": "" if not np.isfinite(r.get("score", np.nan)) else round(float(r["score"]), 4),
            "score_pct": "" if pd.isna(r.get("score_pct")) else round(float(r["score_pct"]), 2),
            "n_scored_cells": int(r.get("n_scored_cells", 0) or 0),
            "x": int(r["x"]), "y": int(r["y"]), "size": int(r["size"]),
            "tissue_fraction": r["tissue_fraction"],
            "dup_of": r["dup_of"], "order": int(r["order"]),
        })

    _write_csv(out_dir / MANIFEST_NAME, MANIFEST_COLUMNS, manifest_rows)
    _write_csv(out_dir / FRAME_NAME, FRAME_COLUMNS, frame_rows)
    tile_um = manifest_rows[0]["tile_um"] if manifest_rows else 0.0
    ctx_um = manifest_rows[0]["context_um"] if manifest_rows else 0.0
    (out_dir / "sampling_report.md").write_text(
        sampling_report(seq, plan, sources, cfg, tile_um, ctx_um)
    )
    if verbose:
        print(f"\n  package -> {out_dir}")
        print(f"    {MANIFEST_NAME}  {len(manifest_rows)} presentations "
              "(no score, no band, no split -- the app reads this)")
        print(f"    {FRAME_NAME}  the same rows WITH score/band/split "
              "(analysis only; do not open it in front of the reviewer)")
    return out_dir


def _write_csv(path: Path, columns: Sequence[str],
               rows: Iterable[dict[str, Any]]) -> Path:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


# ---- the report ------------------------------------------------------------


def sampling_report(seq: pd.DataFrame, plan: TilePlan,
                    sources: dict[str, SlideSource], cfg: Any,
                    tile_um: float = 0.0, context_um: float = 0.0) -> str:
    """What was drawn and from where. Written into the package, not just printed."""
    uniq = seq[seq["dup_of"] == ""]
    L: list[str] = ["# Ballooning tile set — what was drawn", ""]
    L.append(f"- presentations: **{len(seq)}**  "
             f"({len(uniq)} unique tiles + {len(seq) - len(uniq)} repeats)")
    L.append(f"- slides: **{seq['slide'].nunique()}**   "
             f"batches: **{seq['batch'].nunique()}**   "
             f"cohorts: **{seq['cohort'].nunique()}**")
    L.append(f"- tile: {cfg.tiling.tile_size} px at level {cfg.tiling.level} "
             f"= **{tile_um} µm** across; context view "
             f"{plan.context_tiles}×{plan.context_tiles} tiles = "
             f"**{context_um} µm**")
    L.append(f"- seed: `{plan.seed}`")
    L.append("")

    def table(title: str, col: str, extra: str = "") -> None:
        L.append(f"## {title}{extra}")
        L.append("")
        L.append(f"| {col} | tiles | share |")
        L.append("| --- | ---: | ---: |")
        vc = seq[col].value_counts()
        for k, v in vc.items():
            L.append(f"| {k} | {v} | {v / len(seq):.1%} |")
        L.append("")

    table("By band", "band")
    table("By cohort", "cohort")
    table("By staining batch", "batch")
    table("By split", "split")
    table("By diet label", "diet")

    L.append("## By slide")
    L.append("")
    L.append("| slide | cohort | batch | split | tiles | frame | patches |")
    L.append("| --- | --- | --- | --- | ---: | ---: | ---: |")
    for name, grp in seq.groupby("slide"):
        s = sources.get(name)
        if not (s and s.has_score):
            frame_txt, patch_txt = "—", "unscored"
        else:
            patches = int(grp["frame_patches"].iloc[0]) if "frame_patches" \
                in grp else 0
            frame_txt = (str(int(grp["frame_tiles"].iloc[0]))
                         if "frame_tiles" in grp else "?")
            patch_txt = "1 (one strip)" if patches == 1 else str(patches)
        L.append(f"| {name} | {grp['cohort'].iloc[0]} | {grp['batch'].iloc[0]} "
                 f"| {grp['split'].iloc[0]} | {len(grp)} "
                 f"| {frame_txt} | {patch_txt} |")
    L.append("")
    one_strip = [n for n, g in seq.groupby("slide")
                 if (s := sources.get(n)) and s.has_score
                 and "frame_patches" in g
                 and int(g["frame_patches"].iloc[0]) == 1]
    if one_strip:
        L.append(f"{len(one_strip)} slide(s) were scored as a single "
                 "contiguous raster range, so their frame is one horizontal "
                 "strip of section and their band percentiles describe that "
                 "strip rather than the whole slide: "
                 + ", ".join(f"`{n}`" for n in one_strip))
        L.append("")

    short = seq.attrs.get("shortfalls") or {}
    if short:
        L.append("## Bands that could not be filled")
        L.append("")
        for b, n in short.items():
            L.append(f"- `{b}`: {n} short of target — the frame or the "
                     "diversity caps ran out, not a silent truncation.")
        L.append("")
    return "\n".join(L)


# ---- the whole thing -------------------------------------------------------


def read_tile_ids(package: str | Path) -> set[str]:
    """Every unique tile_id already drawn into an existing package.

    Read from `sampling_frame.csv` rather than `manifest.csv` because the frame
    marks repeats: a repeat is the same PIXELS under a second id, so counting
    it would exclude nothing new while inflating the reported overlap.
    """
    f = Path(package) / "sampling_frame.csv"
    if not f.exists():
        raise TileSetError(f"no sampling_frame.csv under {package}")
    df = pd.read_csv(f)
    dup = df.get("dup_of")
    if dup is not None:
        df = df[dup.isna() | (dup == "")]
    return set(df["tile_id"])


def build_set(
    sources: Sequence[SlideSource],
    out_dir: str | Path,
    plan: TilePlan,
    cfg: Any,
    verbose: bool = True,
    exclude: set[str] | None = None,
) -> Path:
    """Frame -> bands -> draw -> split -> duplicates -> order -> package.

    `exclude` drops tiles already drawn into an earlier package. Two sets built
    from one frame at one seed overlap heavily -- the pilot and the full set
    shared 68 tiles, a quarter of the pilot -- and since a pilot label is
    ordinary training data, that overlap is the reviewer judging the same field
    twice for nothing. Excluding makes the packages a clean union.
    """
    plan.validate()
    if not sources:
        raise TileSetError("no slides to sample from")

    if verbose:
        n_scored = sum(1 for s in sources if s.has_score)
        print(f"\nframe: {len(sources)} slide(s), {n_scored} with a detector score")
    frames = []
    for src in sources:
        try:
            frames.append(build_frame(src, cfg, plan, verbose))
        except Exception as exc:                 # one bad slide must not kill the set
            print(f"  {src.name:28s} SKIPPED: {type(exc).__name__}: {exc}")
    if not frames:
        raise TileSetError("every slide failed to produce a frame")
    frame = pd.concat(frames, ignore_index=True)
    if exclude:
        before = len(frame)
        frame = frame[~frame["tile_id"].isin(exclude)]
        if verbose:
            print(f"  excluded {before - len(frame)} tile(s) already drawn "
                  f"into an earlier package")
    frame = assign_bands(frame, plan)

    if verbose:
        print(f"\nframe total: {len(frame)} tiles")
        print("\ndraw:")
    picked = draw(frame, plan, verbose)
    picked = assign_split(picked, plan, verbose)
    rows = add_duplicates(picked, plan, verbose)
    seq = order_presentations(rows, plan, verbose)
    seq.attrs["shortfalls"] = picked.attrs.get("shortfalls", {})

    if verbose:
        print("\nrender:")
    by_name = {s.name: s for s in sources}
    return write_package(seq, out_dir, plan, by_name, cfg, verbose)


def _main(argv: list[str] | None = None) -> int:
    import argparse

    from ..config import MashConfig

    p = argparse.ArgumentParser(
        prog="python -m mashpath.review.tileset",
        description="Build a tile-level binary labelling set for ballooning.",
    )
    p.add_argument("--slide-dir", action="append", required=True,
                   help="directory of slides; repeatable")
    p.add_argument("--score-dir", action="append", default=[],
                   help="pipeline output dir holding <slide>/ballooning/cells.csv; "
                        "repeatable. Slides without one supply anchor tiles only.")
    p.add_argument("--out", required=True, help="where to write the package")
    p.add_argument("--config", action="append", default=None,
                   help="pipeline config (tiling/tissue); repeatable, left to right")
    p.add_argument("--plan", default=None, help="tileset plan YAML")
    p.add_argument("--n", type=int, default=None,
                   help="override the presentation count")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dripped", default="cluster/.dripped",
                   help="cluster slide list, used only to recover staining batch")
    p.add_argument("--exclude", action="append", default=[],
                   help="path to an existing package whose tiles must not be "
                        "drawn again; repeatable. Two sets built from one frame "
                        "at one seed overlap heavily, and a pilot label is "
                        "ordinary training data -- so the overlap is the "
                        "reviewer judging the same field twice for nothing.")
    p.add_argument("--scored-only", action="store_true",
                   help="drop slides with no detector output. Only the anchor "
                        "band can be drawn from an unscored slide, so mixing "
                        "them in means the banded strata and the anchor come "
                        "from different slide populations -- fine for a large "
                        "set, wrong for a pilot whose job is to test whether "
                        "the bands mean anything")
    a = p.parse_args(argv)

    cfg = MashConfig.from_yaml(a.config) if a.config else MashConfig()
    plan = TilePlan.from_yaml(a.plan) if a.plan else TilePlan()
    if a.n:
        plan = replace(plan, n_presentations=a.n)
    if a.seed is not None:
        plan = replace(plan, seed=a.seed)

    sources = discover_sources(a.slide_dir, a.score_dir, a.dripped)
    if a.scored_only:
        kept = [s for s in sources if s.has_score]
        print(f"--scored-only: {len(kept)} of {len(sources)} slides have a "
              "detector score")
        sources = kept
    exclude: set[str] = set()
    for pkg in a.exclude:
        ids = read_tile_ids(pkg)
        exclude |= ids
        print(f"excluding {len(ids)} tile(s) already drawn into {pkg}")
    build_set(sources, a.out, plan, cfg, exclude=exclude or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
