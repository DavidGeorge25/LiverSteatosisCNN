"""QC rendering for nucleus segmentation and the class split.

Two questions, two pictures, and they answer different things:

`class_overlay` draws every nucleus outlined in its predicted class colour, so
you can check that the labels land on the objects you would have labelled by
eye. It shows whether the classifier is *plausible*.

`distribution_figure` plots the joint distribution of nuclear area against
intensity with the cutoffs drawn on top. It shows whether the split is
*viable* -- if that cloud is one blob, no choice of threshold separates immune
from hepatocyte nuclei and the overlay will look reasonable anyway, because a
line through the middle of a single mode still puts small dark things on one
side. The overlay can only ever confirm; the distribution can falsify.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from skimage.segmentation import find_boundaries

from ...core.viz import hstack_panels, label_panel, save_rgb, shade_mask
from .classify import AMBIGUOUS, CLASS_NAMES, DEBRIS, HEPATOCYTE, IMMUNE
from .config import ClassifyConfig, NucleusQCConfig

# Draw order: hepatocytes first so the rarer classes sit on top.
_DRAW_ORDER = (HEPATOCYTE, AMBIGUOUS, IMMUNE)


def class_colors(cfg: NucleusQCConfig) -> dict[int, tuple[int, int, int]]:
    return {
        IMMUNE: tuple(cfg.immune_color),
        HEPATOCYTE: tuple(cfg.hepatocyte_color),
        AMBIGUOUS: tuple(cfg.ambiguous_color),
        DEBRIS: (130, 130, 130),
    }


def class_image(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Map a 1..N label image to class codes, 0 = background.

    Stored as code+1 so background stays distinguishable from `DEBRIS` (0).
    """
    lut = np.zeros(int(labels.max()) + 1, dtype=np.int16)
    n = min(classes.size, lut.size - 1)
    if n > 0:
        lut[1 : n + 1] = classes[:n].astype(np.int16) + 1
    return np.take(lut, np.clip(labels, 0, lut.size - 1))


def class_overlay(
    rgb: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    cfg: NucleusQCConfig,
) -> np.ndarray:
    """Outline (and optionally shade) each nucleus in its class colour.

    Boundaries come from the label image rather than from a merged binary
    mask, so two nuclei of the same class touching each other keep separate
    outlines instead of fusing into one blob.
    """
    out = rgb.copy()
    if labels.max() == 0:
        return out

    cimg = class_image(labels, classes)
    colors = class_colors(cfg)

    if cfg.fill_alpha > 0:
        for code in _DRAW_ORDER:
            m = cimg == code + 1
            if m.any():
                out = shade_mask(out, m, colors[code], cfg.fill_alpha)

    edges = find_boundaries(labels, mode="inner")
    thickness = max(1, int(cfg.outline_thickness))
    for code in _DRAW_ORDER:
        m = edges & (cimg == code + 1)
        if not m.any():
            continue
        if thickness > 1:
            m = cv2.dilate(
                m.astype(np.uint8), np.ones((thickness, thickness), np.uint8)
            ).astype(bool)
        out[m] = colors[code]
    return out


def legend_bar(
    width: int, cfg: NucleusQCConfig, counts=None, height: int = 30
) -> np.ndarray:
    """A colour key, so class identity is never carried by colour alone."""
    bar = np.full((height, width, 3), 255, dtype=np.uint8)
    colors = class_colors(cfg)
    x = 8
    for code in (IMMUNE, HEPATOCYTE, AMBIGUOUS):
        cv2.rectangle(bar, (x, 9), (x + 16, height - 9), colors[code], -1)
        text = CLASS_NAMES[code]
        if counts is not None:
            text += f" {counts.get(code)} ({counts.fraction(code) * 100:.0f}%)"
        x += 22
        cv2.putText(bar, text, (x, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (40, 40, 40), 1, cv2.LINE_AA)
        x += int(9.2 * len(text)) + 18
    return bar


def tile_class_figure(
    rgb: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    cfg: NucleusQCConfig,
    caption: str = "",
    counts=None,
) -> np.ndarray:
    """Side-by-side original and class overlay, with a key underneath."""
    overlay = class_overlay(rgb, labels, classes, cfg)
    fig = hstack_panels(
        [label_panel(rgb, "H&E"), label_panel(overlay, caption or "predicted class")],
        gap=6,
    )
    return np.vstack([fig, legend_bar(fig.shape[1], cfg, counts)])


# ---- feature-space QC ----------------------------------------------------


def distribution_figure(
    area_um2: np.ndarray,
    intensity: np.ndarray,
    classes: np.ndarray,
    ccfg: ClassifyConfig,
    qcfg: NucleusQCConfig,
    path: str | Path,
    title: str = "",
    subtitle: str = "",
    y_label: str | None = None,
    y_cut: float | None = None,
) -> Path:
    """Joint area-vs-y density with the cutoffs drawn on it.

    `y` is the configured intensity channel by default; pass `y_label` to plot
    something else against area (eccentricity, say) on the same layout.

    The main panel is a single-hue density (magnitude, so one sequential ramp,
    never a rainbow); the marginals are stacked by predicted class, which is
    what ties this figure back to the overlay.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    channel = ccfg.intensity_channel
    keep = np.isfinite(area_um2) & np.isfinite(intensity)
    area, inten, cls = area_um2[keep], intensity[keep], classes[keep]

    if area.size > qcfg.scatter_max_points:
        idx = np.linspace(0, area.size - 1, qcfg.scatter_max_points).astype(int)
        area, inten, cls = area[idx], inten[idx], cls[idx]

    colors = {c: np.array(v) / 255.0 for c, v in class_colors(qcfg).items()}

    fig = plt.figure(figsize=(9.0, 7.4))
    gs = fig.add_gridspec(
        2, 2, width_ratios=(4.2, 1.3), height_ratios=(1.3, 4.2),
        wspace=0.04, hspace=0.04,
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    # --- joint density: magnitude, so one hue light->dark ---
    if area.size:
        ax.hexbin(area, inten, gridsize=70, bins="log", cmap="Blues",
                  mincnt=1, linewidths=0)

    # --- cutoffs ---
    guides = [
        (ax.axvline, ccfg.immune_min_area_um2, "debris floor"),
        (ax.axvline, ccfg.immune_max_area_um2, "immune max area"),
        (ax.axvline, ccfg.hepatocyte_min_area_um2, "hepatocyte min area"),
    ]
    for fn, val, _lab in guides:
        fn(val, color="#444444", lw=1.2, ls="--", alpha=0.85, zorder=5)
    if y_label is None and ccfg.require_dark:
        y_cut = (ccfg.immune_max_intensity if channel == "gray"
                 else ccfg.immune_min_hematoxylin)
    if y_cut is not None:
        ax.axhline(y_cut, color="#444444", lw=1.2, ls=":", alpha=0.85, zorder=5)

    ax.set_xlabel("nuclear area (um$^2$)")
    ax.set_ylabel(y_label or (
        f"mean {channel} intensity"
        + ("  (lower = darker)" if channel == "gray" else "  (higher = darker)")))
    ax.grid(alpha=0.18, lw=0.6)
    ax.set_axisbelow(True)

    # --- marginals, stacked by predicted class ---
    order = [HEPATOCYTE, AMBIGUOUS, IMMUNE]
    present = [c for c in order if np.any(cls == c)]
    a_bins = np.linspace(0, np.percentile(area, 99.5) if area.size else 1, 70)
    i_bins = np.linspace(*(np.percentile(inten, [0.5, 99.5]) if inten.size else (0, 1)), 70)

    if present:
        ax_top.hist([area[cls == c] for c in present], bins=a_bins, stacked=True,
                    color=[colors[c] for c in present], lw=0)
        ax_right.hist([inten[cls == c] for c in present], bins=i_bins, stacked=True,
                      orientation="horizontal",
                      color=[colors[c] for c in present], lw=0)
    for a in (ax_top, ax_right):
        a.grid(alpha=0.15, lw=0.5)
        a.set_axisbelow(True)
    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False)
    ax_top.set_ylabel("nuclei")
    ax_right.set_xlabel("nuclei")
    ax.set_xlim(a_bins[0], a_bins[-1])
    ax.set_ylim(i_bins[0], i_bins[-1])

    handles = [Patch(facecolor=colors[c], label=CLASS_NAMES[c]) for c in order]
    handles += [
        Line2D([], [], color="#444444", ls="--", label="area cutoffs"),
    ]
    if y_cut is not None:
        handles.append(Line2D([], [], color="#444444", ls=":",
                              label=y_label and "cutoff" or "intensity cutoff"))
    ax_right.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.10),
                    frameon=False, fontsize=8)

    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=12, weight="bold")
    if subtitle:
        fig.text(0.02, 0.945, subtitle, ha="left", fontsize=8.5, color="#555555")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=qcfg.scatter_dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return path


def thumbnail_density_figure(
    thumb: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    downsample: float,
    path: str | Path,
    title: str = "",
    radius: int = 2,
    color: tuple[int, int, int] = (0, 176, 80),
) -> Path:
    """Where the immune-classified nuclei landed, over the whole-slide thumbnail.

    A sanity check on spatial pattern before any clustering exists: real
    lobular inflammation is patchy, and a uniform dusting over every lobule
    means the class is picking up resident sinusoidal cells instead.
    """
    out = thumb.copy()
    h, w = out.shape[:2]
    px = np.round(xs / downsample).astype(int)
    py = np.round(ys / downsample).astype(int)
    ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    for x, y in zip(px[ok], py[ok]):
        cv2.circle(out, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    if title:
        out = label_panel(out, title)
    return save_rgb(path, out)
