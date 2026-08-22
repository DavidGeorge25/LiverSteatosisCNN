"""Build the review package a pathologist actually opens.

Per candidate: a context crop at a fixed PHYSICAL size, an overlay marking what
the detector proposed, and one manifest row with an empty `verdict` column. The
pathologist fills the column in; `load_verdicts` reads it back and yields the
confirmed set to train on.

The crop is sized in microns, not pixels, because the judgment is comparative:
a ballooned hepatocyte is only callable against its neighbours, so the crop has
to contain them at a predictable physical scale no matter which pyramid level
the detector ran at.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..core.config import ReviewConfig
from ..core.io import SlideOutputs
from ..core.slide import Slide
from ..core.viz import label_panel, outline_mask, save_rgb
from . import manifest as manifest_mod
from .candidates import Candidate

VERDICT_YES = {"y", "yes", "1", "true", "confirm", "confirmed"}
VERDICT_NO = {"n", "no", "0", "false", "reject", "rejected"}


def stratified_sample(
    candidates: Sequence[Candidate], cfg: ReviewConfig
) -> list[Candidate]:
    """Sample up to `max_per_slide`, spread evenly across score bands.

    Uniform random sampling is dominated by borderline candidates, which is
    exactly the population a pathologist can least afford to spend a whole
    session on. Banding by score guarantees the confident and the marginal ends
    are both represented, so the confirmed set spans the range the model will
    meet.
    """
    items = list(candidates)
    if len(items) <= cfg.max_per_slide:
        return sorted(items, key=lambda c: -c.score)

    rng = random.Random(cfg.seed)
    items.sort(key=lambda c: c.score)
    bands = max(1, cfg.strata)
    per_band = max(1, cfg.max_per_slide // bands)
    edges = np.linspace(0, len(items), bands + 1).astype(int)

    picked: list[Candidate] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = items[lo:hi]
        if not band:
            continue
        picked.extend(rng.sample(band, min(per_band, len(band))))

    # Banding rounds down; top up with the highest-scoring leftovers.
    if len(picked) < cfg.max_per_slide:
        chosen = {id(c) for c in picked}
        rest = [c for c in items if id(c) not in chosen]
        rest.sort(key=lambda c: -c.score)
        picked.extend(rest[: cfg.max_per_slide - len(picked)])

    return sorted(picked, key=lambda c: -c.score)


def export_candidates(
    slide: Slide,
    candidates: Sequence[Candidate],
    outputs: SlideOutputs,
    feature: str,
    cfg: ReviewConfig,
    level: int = 0,
) -> Path:
    """Write crops, overlays and the manifest. Returns the manifest path."""
    picked = stratified_sample(candidates, cfg)
    out_dir = outputs.review_dir(feature)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    if cfg.save_overlay:
        (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    context_px = slide.um_to_px(cfg.context_um, level)
    rows: list[dict[str, Any]] = []

    for c in picked:
        half = context_px // 2
        x0, y0 = c.x - half, c.y - half
        crop = slide.read_region((x0, y0), level, (context_px, context_px))

        img_rel = f"images/{c.candidate_id}.png"
        save_rgb(out_dir / img_rel, crop)

        ov_rel = ""
        if cfg.save_overlay:
            box = np.zeros(crop.shape[:2], dtype=bool)
            bx0 = max(0, half - c.width // 2)
            by0 = max(0, half - c.height // 2)
            box[by0 : by0 + c.height, bx0 : bx0 + c.width] = True
            fig = outline_mask(
                crop, box, tuple(cfg.overlay_color), cfg.overlay_thickness
            )
            fig = label_panel(
                fig,
                f"{c.candidate_id}  score={c.score:.3f}  "
                + "  ".join(f"{k}={v:.4g}" for k, v in c.measurements.items()),
            )
            ov_rel = f"overlays/{c.candidate_id}.png"
            save_rgb(out_dir / ov_rel, fig)

        rows.append(manifest_mod.row(
            c, image=img_rel, overlay=ov_rel,
            # A fixed window, so every crop in this package is the same
            # physical size whatever the candidate's own extent.
            crop_um=cfg.context_um, crop_mode=manifest_mod.CROP_FIXED,
        ))

    manifest = manifest_mod.write(out_dir / "manifest.csv", rows)
    _write_instructions(out_dir, feature, len(rows), len(candidates), cfg)
    return manifest


def load_verdicts(
    manifest: str | Path, confirmed_only: bool = True
) -> list[Candidate]:
    """Read a reviewed manifest back.

    Unfilled rows are NOT treated as rejections -- an unreviewed candidate is
    unknown, not negative, and training on it as a negative would teach the
    model exactly the wrong thing.
    """
    out: list[Candidate] = []
    with open(manifest, newline="") as fh:
        for row in csv.DictReader(fh):
            verdict = (row.get("verdict") or "").strip().lower()
            if confirmed_only and verdict not in VERDICT_YES:
                continue
            out.append(Candidate.from_row(row))
    return out


def verdict_summary(manifest: str | Path) -> dict[str, int]:
    """Counts of confirmed / rejected / unreviewed in a manifest."""
    counts = {"confirmed": 0, "rejected": 0, "unreviewed": 0, "other": 0}
    with open(manifest, newline="") as fh:
        for row in csv.DictReader(fh):
            v = (row.get("verdict") or "").strip().lower()
            if not v:
                counts["unreviewed"] += 1
            elif v in VERDICT_YES:
                counts["confirmed"] += 1
            elif v in VERDICT_NO:
                counts["rejected"] += 1
            else:
                counts["other"] += 1
    return counts


def _write_instructions(
    out_dir: Path, feature: str, n: int, n_total: int, cfg: ReviewConfig
) -> None:
    (out_dir / "REVIEW.md").write_text(
        f"""# {feature} candidate review

{n} candidates to review (sampled from {n_total} the detector proposed,
spread across {cfg.strata} score bands so both the confident and the marginal
ends are represented).

These are **proposals, not detections**. The detector cannot make this call on
its own -- that is why you are here. Expect to reject a substantial fraction;
that is the system working, not failing.

## How to review

1. Open `manifest.csv` in Excel or a text editor.
2. For each row, look at `overlays/<candidate_id>.png` (the proposal outlined
   in context) or `images/<candidate_id>.png` (the same crop, unmarked).
   Each crop is {cfg.context_um:.0f} um across, so neighbouring cells are in
   frame at a consistent physical scale.
3. Fill in `verdict`:
     y  - yes, this is {feature}
     n  - no, it is not
     ?  - cannot tell from this crop
4. Optionally add your initials in `reviewer` and anything worth recording in
   `notes` (especially *why* for a rejection -- that is what tells us which
   parameter is wrong).
5. Save as CSV and hand the file back.

Leave a row blank if you did not look at it. Blank means unreviewed, not "no";
only rows marked `y` become training data, and rows marked `n` are used to tell
the detector what it got wrong.
"""
    )
