"""Steatosis-specific QC rendering.

The two-class (macro/micro) panels only mean anything for fat droplets, so
they live with the detector rather than in the shared primitives.
"""

from __future__ import annotations

import numpy as np

from ...core.viz import hstack_panels, label_panel, outline_mask


def two_class_mask_rgb(
    macro: np.ndarray,
    micro: np.ndarray,
    macro_color: tuple[int, int, int] = (255, 0, 0),
    micro_color: tuple[int, int, int] = (0, 190, 255),
) -> np.ndarray:
    """Binary mask panel with the two droplet classes in distinct colors."""
    out = np.zeros((*macro.shape, 3), dtype=np.uint8)
    out[micro.astype(bool)] = micro_color
    out[macro.astype(bool)] = macro_color
    return out


def tile_qc_figure(
    rgb: np.ndarray,
    macro: np.ndarray,
    micro: np.ndarray,
    macro_color: tuple[int, int, int] = (255, 0, 0),
    micro_color: tuple[int, int, int] = (0, 190, 255),
    thickness: int = 1,
    caption: str = "",
) -> np.ndarray:
    """Three-panel tile QC: original | mask | outlined overlay.

    Macro and micro droplets are drawn in different colors in both the mask
    panel and the overlay.
    """
    overlay = outline_mask(rgb, micro, micro_color, thickness)
    overlay = outline_mask(overlay, macro, macro_color, thickness)
    panels = [
        label_panel(rgb, "original"),
        label_panel(
            two_class_mask_rgb(macro, micro, macro_color, micro_color), "mask"
        ),
        label_panel(overlay, caption or "overlay"),
    ]
    return hstack_panels(panels, gap=6)
