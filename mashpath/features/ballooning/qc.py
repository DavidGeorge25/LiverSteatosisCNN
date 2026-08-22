"""Segmentation QC overlays.

The point of this pass is to answer one question before any feature is built on
top of the segmentation: are these actually hepatocyte boundaries?

So the overlays are written at NATIVE resolution, one file per tile. A
downscaled contact sheet hides exactly the failure this pass exists to catch --
a cell body that leaked across a sinusoid into its neighbour looks fine at 25%
zoom and obviously wrong at 100%.

Three things are drawn, in three colours:

    nucleus outline      what StarDist actually detected
    cell outline         where the expansion put the cell body
    hepatocyte outline   the subset ballooning will be judged over

Non-hepatocyte cells (endothelial, Kupffer, immune) are drawn too rather than
hidden, because seeing them claim their own small territories is how you
confirm they are not being absorbed into the hepatocytes around them.
"""

from __future__ import annotations

import numpy as np
from skimage.segmentation import find_boundaries

from ...core.viz import hstack_panels, label_panel
from .config import QCConfig
from .segment import TileCells


def _thicken(mask: np.ndarray, thickness: int) -> np.ndarray:
    """Widen a 1px boundary mask so it survives at native zoom."""
    if thickness <= 1:
        return mask
    import cv2

    k = np.ones((thickness, thickness), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def _hepatocyte_lut(cells: TileCells, max_label: int) -> np.ndarray:
    """label -> is_hepatocyte, exploiting the 1..N relabelling in `segment`."""
    lut = np.zeros(max_label + 1, dtype=bool)
    n = min(int(cells.is_hepatocyte.size), max_label)
    if n:
        lut[1 : n + 1] = cells.is_hepatocyte[:n]
    return lut


def segmentation_overlay(
    rgb: np.ndarray, cells: TileCells, cfg: QCConfig
) -> np.ndarray:
    """Draw nuclei and cell bodies on a copy of `rgb`."""
    out = rgb.copy()
    cell_lab = cells.cell_labels
    if cell_lab.max() == 0:
        return out

    # `find_boundaries` on a LABEL image separates touching cells from each
    # other; a binary mask would merge them into one blob and hide precisely
    # the boundary being checked.
    bnd = find_boundaries(cell_lab, mode="inner")
    is_hep = _hepatocyte_lut(cells, int(cell_lab.max()))[cell_lab]

    hep_edge = _thicken(bnd & is_hep, cfg.thickness)
    other_edge = _thicken(bnd & ~is_hep, cfg.thickness)
    out[other_edge] = cfg.cell_color
    out[hep_edge] = cfg.hepatocyte_color

    nuc_edge = _thicken(find_boundaries(cells.nucleus_labels, mode="inner"),
                        cfg.thickness)
    out[nuc_edge] = cfg.nucleus_color
    return out


def tile_qc_figure(
    rgb: np.ndarray, cells: TileCells, cfg: QCConfig, caption: str = ""
) -> np.ndarray:
    """Raw H&E beside the overlay -- the only honest way to judge a boundary.

    Without the unmarked panel next to it there is no way to tell a boundary
    the model found from a boundary the drawing implies.
    """
    overlay = segmentation_overlay(rgb, cells, cfg)
    if not cfg.side_by_side:
        return label_panel(overlay, caption) if caption else overlay

    left = label_panel(rgb, "H&E")
    right = label_panel(
        overlay,
        f"nuclei={len(cells)}  "
        f"hepatocytes={int(cells.is_hepatocyte.sum())}  {caption}",
    )
    return hstack_panels([left, right])


def summarize(cells: TileCells, mpp: float) -> dict[str, float]:
    """Per-tile numbers worth printing while eyeballing the overlays."""
    hep = cells.is_hepatocyte
    # Every StarDist detection ends up either kept or in `rejected`, so the two
    # recover the raw yield. Worth reporting: too few seeds is the failure that
    # silently inflates every cell area, because each surviving nucleus then
    # claims the territory of the ones that were missed.
    detected = len(cells) + sum(cells.rejected.values())
    out: dict[str, float] = {
        "cells": float(len(cells)),
        "hepatocytes": float(hep.sum()),
        "detected_in_padded": float(detected),
    }
    if hep.any():
        area = cells.territory_area_um2[hep]
        out.update(
            median_territory_area_um2=float(np.median(area)),
            p10_territory_area_um2=float(np.percentile(area, 10)),
            p90_territory_area_um2=float(np.percentile(area, 90)),
            median_diameter_um=float(np.median(cells.equivalent_diameter_um[hep])),
            median_nucleus_area_um2=float(np.median(cells.nucleus_area_um2[hep])),
            median_nc_ratio=float(np.median(cells.nc_ratio[hep])),
            median_circularity=float(np.median(cells.circularity[hep])),
            median_solidity=float(np.median(cells.solidity[hep])),
        )
    return out
