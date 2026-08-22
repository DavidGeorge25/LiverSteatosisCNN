"""Review crops and the operator's contact sheet.

Per candidate the reviewer gets two images of the same region: one with the
segmented cell outlined, one untouched. Both, always. An outline is itself a
claim -- it asserts where the cell ends, which on this segmentation is a
territory rather than a membrane (see `segment`'s module docstring) -- and a
reviewer needs to be able to judge the cell without the detector's opinion
drawn over it. Showing only the outlined version would make every verdict
partly a verdict on the outline.

The crop is sized in MICRONS, padded around the cell's own bounding box, so
every crop contains a comparable ring of neighbouring hepatocytes. That is not
cosmetic: ballooning is a relative call, and a crop that does not contain the
cells being compared against cannot be judged at all.

The contact sheet is a separate thing with a separate purpose -- it is for the
operator to sanity check a run at a glance, not for review. It is explicitly
labelled as top-ranked only, because a grid of the most extreme candidates is
exactly the view that makes a bad detector look good.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ...core.slide import Slide
from ...core.viz import grid, label_panel, save_rgb
from .config import CropConfig


def _draw_contour(
    img: np.ndarray, contour: np.ndarray, x0: int, y0: int,
    color: tuple[int, int, int], thickness: int,
) -> None:
    """Draw a level-0 contour onto a crop whose top-left is (x0, y0)."""
    if contour is None or len(contour) < 3:
        return
    pts = (np.asarray(contour) - np.array([x0, y0])).astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def crop_candidate(
    slide: Slide,
    contour: np.ndarray,
    nucleus_contour: np.ndarray,
    cfg: CropConfig,
    mpp: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Read one candidate's context crop. Returns (marked, unmarked, bbox)."""
    pad = int(round(cfg.padding_um / mpp))
    c = np.asarray(contour)
    if c.size == 0:
        raise ValueError("candidate has no contour to crop around")

    x0 = int(c[:, 0].min()) - pad
    y0 = int(c[:, 1].min()) - pad
    x1 = int(c[:, 0].max()) + pad
    y1 = int(c[:, 1].max()) + pad

    w, h = slide.dimensions
    # Clamp to the slide, then read. openslide fills out-of-bounds with black,
    # which would read as tissue; clamping keeps the crop honest even at an
    # edge, at the cost of an off-centre cell there.
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(w, x1), min(h, y1)
    unmarked = slide.read_region((cx0, cy0), 0, (cx1 - cx0, cy1 - cy0))

    marked = unmarked.copy()
    if cfg.draw_outline:
        _draw_contour(marked, contour, cx0, cy0,
                      tuple(cfg.outline_color), cfg.outline_thickness)
    if cfg.draw_nucleus and nucleus_contour is not None:
        _draw_contour(marked, nucleus_contour, cx0, cy0,
                      tuple(cfg.nucleus_color), max(1, cfg.outline_thickness - 1))
    return marked, unmarked, (cx0, cy0, cx1 - cx0, cy1 - cy0)


def write_candidate_images(
    slide: Slide,
    candidate_id: str,
    contour: np.ndarray,
    nucleus_contour: np.ndarray,
    out_dir: Path,
    cfg: CropConfig,
    mpp: float,
) -> tuple[str, str, np.ndarray]:
    """Write the outlined and unmarked crops. Returns their relative paths."""
    marked, unmarked, _ = crop_candidate(
        slide, contour, nucleus_contour, cfg, mpp
    )
    overlay_rel = f"overlays/{candidate_id}.png"
    save_rgb(out_dir / overlay_rel, marked)

    image_rel = ""
    if cfg.save_unmarked:
        image_rel = f"images/{candidate_id}.png"
        save_rgb(out_dir / image_rel, unmarked)
    return image_rel, overlay_rel, marked


def contact_sheet(
    panels: list[np.ndarray], captions: list[str], cfg: CropConfig
) -> np.ndarray:
    """A grid of top candidates, for the operator's sanity check only."""
    thumbs = []
    for img, cap in zip(panels, captions):
        t = cv2.resize(
            img, (cfg.contact_sheet_thumb_px, cfg.contact_sheet_thumb_px),
            interpolation=cv2.INTER_AREA,
        )
        thumbs.append(label_panel(t, cap, height=20))
    return grid(thumbs, cols=cfg.contact_sheet_cols)
