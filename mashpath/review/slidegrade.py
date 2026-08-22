"""Slide-level NASH-CRN ballooning grade: the round that validates the model.

Field-level counts train a model; they cannot validate it. Every comparable
study -- Heinemann 2019, AIM-NASH, the semaglutide trial re-read -- reports the
same thing: a continuous model score aggregated over a slide, mapped onto the
pathologist's discrete NASH-CRN grade for that slide. That grade is a separate
observation and cannot be derived from tile labels, so it has to be collected.

Two properties matter more than anything else here.

UNIFORM SAMPLING. The fields shown for grading are drawn uniformly from the
slide's tissue, NOT from the enriched score bands the training round uses.
NASH-CRN distinguishes "few" from "many", which is a statement about
PREVALENCE -- so showing a score-enriched sample would inflate every grade by
construction and the validation target would be silently wrong.

BLINDING. Slides are shown under opaque ids in shuffled order. A pathologist
who can tell she is on the CCl4 block grades the block, not the section.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.slide import Slide

# A grading montage is a SAMPLE of the section, not the section. Recorded here
# because the number bounds what the grade can mean: 24 fields of 253.6 um is
# ~1.5 mm^2 against a section of 150-300 mm^2, so this is a ~1% sample and
# "many" means "many in what she was shown".
TILES_PER_SLIDE = 24
THUMB_PX = 400
THUMB_Q = 72


def sample_uniform(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """n tiles spread over one slide's tissue, unconditioned on score."""
    if len(frame) <= n:
        return frame
    # Spatially spread rather than iid: an iid draw of 24 from a few hundred
    # clumps often enough to bias a prevalence judgement.
    f = frame.copy()
    f["_cell"] = (f["y"] // 2048).astype(int) * 10_000 + (f["x"] // 2048).astype(int)
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    cells = list(f["_cell"].unique())
    rng.shuffle(cells)
    # Round-robin over spatial cells: one tile from each before any cell gives a
    # second, so the sample spreads before it repeats.
    remaining = {c: list(f.index[f["_cell"] == c]) for c in cells}
    for lst in remaining.values():
        rng.shuffle(lst)
    while len(picked) < n and any(remaining.values()):
        for c in cells:
            if not remaining[c]:
                continue
            picked.append(remaining[c].pop())
            if len(picked) >= n:
                break
    # Top up if the section simply has fewer spatial cells than tiles wanted.
    # The earlier version returned short here, silently -- CCl4 sections are
    # small and came back with 14 panels where 24 were asked for, which would
    # have made their grades rest on less evidence than the MASH ones without
    # anything in the output saying so.
    if len(picked) < n:
        rest = [i for i in f.index if i not in set(picked)]
        rng.shuffle(rest)
        picked += rest[: n - len(picked)]
    return f.loc[picked].drop(columns=["_cell"])


def blind_id(slide: str, seed: int) -> str:
    """Stable opaque id, so a rebuild does not reshuffle what she has graded."""
    h = hashlib.sha256(f"{seed}:grade:{slide}".encode()).hexdigest()
    return "s" + h[:6]


def build(frame_csv: str | Path, slide_dirs: list[str], out_dir: str | Path,
          seed: int = 20260819, tiles: int = TILES_PER_SLIDE,
          verbose: bool = True) -> Path:
    """Render the montages and write the key. Returns the output directory."""
    import cv2

    frame = pd.read_csv(frame_csv)
    out_dir = Path(out_dir)
    (out_dir / "tiles").mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for d in slide_dirs:
        for p in Path(d).glob("*.svs"):
            paths[p.stem] = p

    key_rows: list[dict[str, Any]] = []
    for name, grp in frame.groupby("slide"):
        if name not in paths:
            if verbose:
                print(f"  {name}: no slide file, skipped")
            continue
        sel = sample_uniform(grp, tiles, seed)
        sid = blind_id(name, seed)
        sl = Slide(str(paths[name]), None)
        try:
            for i, r in enumerate(sel.itertuples(), 1):
                img = sl.read_region((int(r.x), int(r.y)), 0, (int(r.size), int(r.size)))
                img = cv2.resize(img, (THUMB_PX, THUMB_PX), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out_dir / "tiles" / f"{sid}_{i:02d}.jpg"),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                            [int(cv2.IMWRITE_JPEG_QUALITY), THUMB_Q])
                key_rows.append(dict(blind_id=sid, panel=i, slide=name,
                                     cohort=r.cohort, batch=r.batch,
                                     split=r.split, x=int(r.x), y=int(r.y)))
        finally:
            sl.close()
        if verbose:
            print(f"  {name:24s} -> {sid}  {len(sel)} panels")

    key = pd.DataFrame(key_rows)
    key.to_csv(out_dir / "key.csv", index=False)
    if verbose:
        print(f"\n{key.blind_id.nunique()} slides, {len(key)} panels -> {out_dir}")
    return out_dir
