"""The manifest schema: one definition, both writers, one reader.

Ballooning and inflammation generate candidates by completely different
routes, and until this module existed they each wrote their own manifest.
The column sets had already diverged -- ballooning emitted `review_band`,
`rank` and `padding_um`; the generic exporter emitted `context_um` and none of
the others -- and the only reason the review side still loaded both is that
`Candidate.from_row` reads by name and ignores what it does not recognise.
That is not compatibility, it is two formats that have not collided yet.

So: `row()` is the only way to build a manifest row, `write()` the only way to
write the file, and `read()` validates on the way back in. A feature that wants
to record something extra passes it through `extra` and it lands after the
contract columns, before the reviewer's.

CROP GEOMETRY. The two exporters frame a candidate differently and both are
right for their feature -- a fixed window centred on the object, versus a
variable padding around its bounding box. What the review app cannot do is
guess which it got. Every row therefore carries `crop_um` (the physical width
actually written) and `crop_mode` (how it was chosen), so a crop can be scaled
and captioned honestly without the app knowing which detector produced it.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

from .candidates import Candidate

# Written by the detector. `Candidate.to_row` supplies the first block.
IDENTITY = ("candidate_id", "slide", "feature", "x", "y", "width", "height",
            "score", "tile_x", "tile_y")
CROP = ("image", "overlay", "crop_um", "crop_mode")
# Filled in during review. Kept last so a human opening the CSV in Excel finds
# them at the right-hand end rather than hunting between measurement columns.
VERDICT = ("verdict", "reviewer", "notes")

# Without these a manifest cannot be reviewed at all.
REQUIRED = (*IDENTITY, *CROP)

# How the crop was framed.
CROP_FIXED = "fixed_context"   # a window of crop_um centred on the candidate
CROP_PADDED = "bbox_padded"    # the candidate's bbox plus a padding margin


class ManifestError(ValueError):
    """A manifest that cannot be reviewed, with the reason a human needs."""


def row(
    cand: Candidate,
    image: str,
    overlay: str,
    crop_um: float,
    crop_mode: str = CROP_FIXED,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one manifest row. The only supported way to make one.

    `image` and `overlay` are paths RELATIVE to the manifest's own directory,
    so a review package can be copied off the cluster, or handed to a
    pathologist on a memory stick, and still resolve.
    """
    if crop_mode not in (CROP_FIXED, CROP_PADDED):
        raise ManifestError(
            f"unknown crop_mode {crop_mode!r}; "
            f"expected {CROP_FIXED!r} or {CROP_PADDED!r}"
        )
    out = cand.to_row()  # identity + m_* measurements
    out["image"] = image
    out["overlay"] = overlay
    out["crop_um"] = round(float(crop_um), 3)
    out["crop_mode"] = crop_mode
    if extra:
        out.update(extra)
    for col in VERDICT:
        out[col] = ""
    return out


def write(path: str | Path, rows: Sequence[dict[str, Any]]) -> Path:
    """Write a manifest, columns in contract order. Returns the path.

    An empty candidate set still writes a header. A reviewer handed a
    zero-byte file cannot tell "the detector proposed nothing" from "the run
    crashed", and those call for opposite responses.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(",".join((*REQUIRED, *VERDICT)) + "\n")
        return path

    seen = {k for r in rows for k in r}
    ordered = [c for c in REQUIRED if c in seen]
    ordered += sorted(c for c in seen if c.startswith("m_"))
    ordered += [c for c in sorted(seen)
                if c not in ordered and c not in VERDICT and c not in REQUIRED]
    ordered += [c for c in VERDICT if c in seen]

    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ordered, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ordered})
    return path


def read(path: str | Path) -> list[dict[str, Any]]:
    """Read a manifest, validating that it can actually be reviewed.

    Fails on the missing column rather than at the point some later code
    indexes it -- a KeyError three frames into the web app is a much worse
    way to learn that a detector wrote the wrong schema.
    """
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"no manifest at {path}")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        raise ManifestError(
            f"{path} is missing required column(s) {missing}. "
            f"Found: {sorted(rows[0])}. "
            "It was probably written by a detector that predates "
            "mashpath.review.manifest -- re-export the review package."
        )
    return rows


def candidates(path: str | Path) -> list[Candidate]:
    """The candidates in a manifest, as objects."""
    return [Candidate.from_row(r) for r in read(path)]


def measurement_columns(rows: Iterable[dict[str, Any]]) -> list[str]:
    """The `m_*` columns present, without the prefix. For display."""
    seen: set[str] = set()
    for r in rows:
        seen.update(k[2:] for k in r if k.startswith("m_") and r[k] not in (None, ""))
    return sorted(seen)
