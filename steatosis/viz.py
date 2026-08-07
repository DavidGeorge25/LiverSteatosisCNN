"""Visual QC rendering.

Everything here returns or writes plain RGB uint8 images -- no matplotlib
figures in the hot path, so the same helpers work for a single tile overlay
and for a whole-slide thumbnail.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def outline_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw mask boundaries onto a copy of `rgb`."""
    out = rgb.copy()
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def shade_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    """Translucent fill over the masked region."""
    out = rgb.astype(np.float32)
    overlay = np.zeros_like(out)
    overlay[..., 0], overlay[..., 1], overlay[..., 2] = color
    m = mask.astype(bool)[..., None]
    out = np.where(m, out * (1 - alpha) + overlay * alpha, out)
    return out.astype(np.uint8)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Binary mask as a black/white RGB image."""
    m = (mask.astype(bool) * 255).astype(np.uint8)
    return np.dstack([m, m, m])


def label_panel(img: np.ndarray, text: str, height: int = 28) -> np.ndarray:
    """Add a caption bar above an image."""
    bar = np.full((height, img.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        bar,
        text,
        (6, height - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([bar, img])


def hstack_panels(
    panels: list[np.ndarray], gap: int = 8, bg: int = 255
) -> np.ndarray:
    """Horizontally concatenate images of possibly differing heights."""
    if not panels:
        raise ValueError("no panels to stack")
    height = max(p.shape[0] for p in panels)
    padded = []
    for i, p in enumerate(panels):
        pad_h = height - p.shape[0]
        if pad_h:
            p = np.vstack(
                [p, np.full((pad_h, p.shape[1], 3), bg, dtype=np.uint8)]
            )
        padded.append(p)
        if gap and i < len(panels) - 1:
            padded.append(np.full((height, gap, 3), bg, dtype=np.uint8))
    return np.hstack(padded)


def grid(
    panels: list[np.ndarray], cols: int, gap: int = 8, bg: int = 255
) -> np.ndarray:
    """Tile panels into a contact sheet."""
    if not panels:
        raise ValueError("no panels to grid")
    cols = max(1, cols)
    rows: list[np.ndarray] = []
    for i in range(0, len(panels), cols):
        chunk = list(panels[i : i + cols])
        while len(chunk) < cols:
            chunk.append(np.full_like(chunk[0], bg))
        rows.append(hstack_panels(chunk, gap=gap, bg=bg))

    width = max(r.shape[1] for r in rows)
    out: list[np.ndarray] = []
    for i, r in enumerate(rows):
        if r.shape[1] < width:
            r = np.hstack(
                [r, np.full((r.shape[0], width - r.shape[1], 3), bg, dtype=np.uint8)]
            )
        out.append(r)
        if gap and i < len(rows) - 1:
            out.append(np.full((gap, width, 3), bg, dtype=np.uint8))
    return np.vstack(out)


def save_rgb(path: str | Path, rgb: np.ndarray) -> Path:
    """Write an RGB array to disk as PNG/JPEG (cv2 wants BGR)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"failed to write image: {path}")
    return path


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


def tissue_qc_figure(
    thumbnail: np.ndarray,
    mask: np.ndarray,
    outline_color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    caption: str = "",
) -> np.ndarray:
    """Three-panel tissue QC: original | mask | outlined overlay.

    `mask` is resized to the thumbnail's resolution, so the detection level and
    the display level do not have to match.
    """
    h, w = thumbnail.shape[:2]
    m = cv2.resize(
        mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    panels = [
        label_panel(thumbnail, "original"),
        label_panel(mask_to_rgb(m), "tissue mask"),
        label_panel(
            outline_mask(shade_mask(thumbnail, m, outline_color, alpha=0.20), m,
                         outline_color, thickness),
            "overlay",
        ),
    ]
    fig = hstack_panels(panels, gap=12)
    if caption:
        bar = np.full((26, fig.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(
            bar, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (40, 40, 40), 1, cv2.LINE_AA,
        )
        fig = np.vstack([fig, bar])
    return fig
