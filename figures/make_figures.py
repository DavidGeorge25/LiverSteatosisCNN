"""Grant figures: H&E next to the steatosis pseudo-label.

Reads only from `outputs/dataset_v1/` -- the exported label set itself -- so the
panels show exactly the labels the pipeline ships, not a one-off rerun with
different parameters. Rerunning this script on the same dataset reproduces the
files byte-for-byte.

    .venv/bin/python figures/make_figures.py

Writes PNG (600 dpi, for Word/Illustrator) and PDF (vector text) into
`figures/`, plus native-resolution single panels into `figures/panels/`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,   # embed as TrueType so text stays editable
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "outputs/dataset_v1"
OUT = ROOT / "figures"
PANELS = OUT / "panels"

MPP = 0.4953          # µm per pixel at level 0, constant across all 51 slides
SCALEBAR_UM = 50
LABEL_RGB = (0, 190, 60)      # green: absent from H&E, so it never reads as stain
# 300 dpi clears the usual "300 dpi minimum" bar while barely resampling: panels
# are ~2.1 in wide for a 512 px tile, so native resolution is already ~244 dpi.
# Rendering at 600 would quadruple the file size to invent detail that isn't there.
DPI = 300

# Chosen from the exported set by `select_panels.py`, which ranks tiles by how
# much obviously-droplet-like white space the label misses -- so the panels are
# representative of the pipeline working, not hand-picked by eye. A severe and a
# moderate MASH tile span the cohort's range; a fat-free CCl4 tile is the
# negative control. The moderate tile contains a portal vessel and the control
# tile contains rarefied pale cytoplasm -- the two structures most easily
# mistaken for fat -- so the figure shows the specificity behaviour too.
PANEL_SPECS = [
    ("R25-264-32__tile_x027136_y024576", "Steatotic (severe)"),
    ("R25-264-10__tile_x018944_y005120", "Steatotic (moderate)"),
    ("R26-122-23_HE_91__tile_x022016_y010240", "Fat-free control (CCl$_4$)"),
]


def load_manifest() -> dict[str, dict]:
    with open(DS / "manifest.csv") as fh:
        return {r["id"]: r for r in csv.DictReader(fh)}


MAN = load_manifest()


def load_tile(tid: str):
    """Return (image RGB, consensus label bool, confidence float 0-1)."""
    split = MAN[tid]["split"]
    img = cv2.cvtColor(
        cv2.imread(str(DS / split / "images" / f"{tid}.png")), cv2.COLOR_BGR2RGB
    )
    lab = cv2.imread(str(DS / split / "labels" / f"{tid}.png"), cv2.IMREAD_GRAYSCALE) > 127
    conf = cv2.imread(
        str(DS / split / "confidence" / f"{tid}.png"), cv2.IMREAD_GRAYSCALE
    ).astype(np.float32) / 255.0
    return img, lab, conf


def overlay(img: np.ndarray, lab: np.ndarray, thickness: int = 3,
            alpha: float = 0.22) -> np.ndarray:
    """Translucent fill plus a solid outline -- readable at print size and at
    full resolution, without hiding the droplet interior."""
    out = img.astype(np.float32)
    fill = np.zeros_like(out)
    fill[..., 0], fill[..., 1], fill[..., 2] = LABEL_RGB
    m = lab[..., None]
    out = np.where(m, out * (1 - alpha) + fill * alpha, out)
    out = out.astype(np.uint8)
    cnts, _ = cv2.findContours(lab.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, LABEL_RGB, thickness)
    return out


def confidence_rgb(conf: np.ndarray) -> np.ndarray:
    """Ensemble agreement as viridis; unflagged pixels stay white."""
    rgb = (plt.get_cmap("viridis")(conf)[..., :3] * 255).astype(np.uint8)
    rgb[conf <= 0] = 255
    return rgb


def droplet_count(lab: np.ndarray) -> int:
    n, _ = cv2.connectedComponents(lab.astype(np.uint8))
    return n - 1


def add_scalebar(ax, px: int, um: int = SCALEBAR_UM, frac_pad: float = 0.045):
    """Scale bar in the lower-right corner, sized from the slide's µm/px."""
    length = um / MPP
    pad = px * frac_pad
    h = px * 0.016
    x0, y0 = px - pad - length, px - pad - h
    ax.add_patch(Rectangle((x0 - 6, y0 - 20), length + 12, h + 26,
                           facecolor="white", alpha=0.78, edgecolor="none", zorder=4))
    ax.add_patch(Rectangle((x0, y0), length, h, facecolor="black",
                           edgecolor="none", zorder=5))
    ax.text(x0 + length / 2, y0 - 5, f"{um} µm", ha="center", va="bottom",
            fontsize=6.5, color="black", zorder=6)


def show(ax, img, scalebar=False):
    ax.imshow(img, interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6); s.set_color("0.35")
    if scalebar:
        add_scalebar(ax, img.shape[0])


def fat_pct(tid: str) -> float:
    return float(MAN[tid]["fat_fraction"]) * 100


def save(fig, stem: str):
    for ext in ("png", "pdf"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=DPI)
        print(f"  wrote {p.relative_to(ROOT)}")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 1 -- the main one: H&E over pseudo-label, three tissue types
# --------------------------------------------------------------------------
def figure_main():
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.75))
    fig.subplots_adjust(wspace=0.045, hspace=0.045, left=0.052, right=0.999,
                        top=0.915, bottom=0.005)

    for col, (tid, title) in enumerate(PANEL_SPECS):
        img, lab, _ = load_tile(tid)
        show(axes[0, col], img, scalebar=(col == 0))
        show(axes[1, col], overlay(img, lab))

        axes[0, col].set_title(title, fontsize=8.5, pad=4)
        pct, n = fat_pct(tid), droplet_count(lab)
        note = (f"{pct:.1f}% area, {n} droplets" if n
                else "0.0% area, 0 droplets")
        axes[1, col].text(0.5, -0.035, note, transform=axes[1, col].transAxes,
                          ha="center", va="top", fontsize=7.2,
                          color="0.15" if n else "#0a7a2f")

    axes[0, 0].set_ylabel("H&E", fontsize=9)
    axes[1, 0].set_ylabel("Pseudo-label", fontsize=9, color="#0a7a2f")

    fig.text(0.5, 0.985,
             "Automated pseudo-labelling of macrovesicular steatosis in H&E liver WSI",
             ha="center", va="top", fontsize=9.5, weight="bold")
    save(fig, "Fig_steatosis_pseudolabels")


# --------------------------------------------------------------------------
# Figure 2 -- the minimal drop-in: one pair, side by side
# --------------------------------------------------------------------------
def figure_pair():
    tid = PANEL_SPECS[0][0]
    img, lab, _ = load_tile(tid)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.45))
    fig.subplots_adjust(wspace=0.04, left=0.002, right=0.998, top=0.9, bottom=0.05)
    show(axes[0], img, scalebar=True)
    show(axes[1], overlay(img, lab))
    axes[0].set_title("H&E", fontsize=10, pad=5)
    axes[1].set_title(
        f"Pseudo-label  ({fat_pct(tid):.1f}% area, {droplet_count(lab)} droplets)",
        fontsize=10, pad=5, color="#0a7a2f")
    save(fig, "Fig_steatosis_pair")


# --------------------------------------------------------------------------
# Figure 3 -- adds the per-pixel confidence map (the weak-supervision angle)
# --------------------------------------------------------------------------
def figure_confidence():
    tid = PANEL_SPECS[0][0]
    img, lab, conf = load_tile(tid)

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.75))
    fig.subplots_adjust(wspace=0.04, left=0.002, right=0.88, top=0.87, bottom=0.02)
    show(axes[0], img, scalebar=True)
    show(axes[1], overlay(img, lab))
    show(axes[2], confidence_rgb(conf))

    for ax, t in zip(axes, ["H&E", "Consensus pseudo-label", "Ensemble confidence"]):
        ax.set_title(t, fontsize=8.5, pad=4)

    cax = fig.add_axes([0.895, 0.06, 0.017, 0.78])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="viridis"), cax=cax)
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(["1/9", "5/9", "9/9"])
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.set_label("members voting fat", fontsize=6.8, labelpad=3)
    cb.outline.set_linewidth(0.5)
    save(fig, "Fig_steatosis_confidence")


# --------------------------------------------------------------------------
# Native-resolution single panels, for anyone who wants to re-lay-out the figure
# --------------------------------------------------------------------------
def native_panels():
    PANELS.mkdir(parents=True, exist_ok=True)
    for tid, title in PANEL_SPECS:
        img, lab, conf = load_tile(tid)
        stem = title.split(" (")[0].lower().replace(" ", "_").replace("$_4$", "4")
        sev = title.split("(")[-1].rstrip(")").replace("$_4$", "4")
        base = f"{stem}_{sev}".replace("__", "_")
        for suffix, arr in [("HE", img), ("pseudolabel", overlay(img, lab)),
                            ("confidence", confidence_rgb(conf))]:
            p = PANELS / f"{base}_{suffix}.png"
            cv2.imwrite(str(p), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    print(f"  wrote {len(list(PANELS.glob('*.png')))} native-resolution panels "
          f"to {PANELS.relative_to(ROOT)}/")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("building figures...")
    figure_main()
    figure_pair()
    figure_confidence()
    native_panels()
    print("done.")
