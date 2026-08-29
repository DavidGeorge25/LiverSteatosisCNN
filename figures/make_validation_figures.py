"""The two figures the steatosis validation rests on.

    .venv/bin/python figures/make_validation_figures.py

Reads only `outputs/survey_9batches.csv` -- the shipped survey artifact -- so
the figures show the numbers the pipeline actually produced, not a rerun with
different parameters. Same rule as `make_figures.py`.

  Fig_diet_validation   R22-354: 4 chow against 4 NASH, cut and stained in one
                        run. The only place in 260 slides where diet varies and
                        stain does not, and the detector had never seen any of
                        it. This is the external validation.
  Fig_nine_batches      All 259 slides. The tuned config, unretuned, on seven
                        staining runs it was never tuned on.

DESIGN NOTES, because they are decisions and not taste:

**Log x-axis.** The claim is about orders of magnitude -- chow at 0.07-0.20%
against NASH at 7.6-10.5% -- and on a linear axis every negative slide in both
figures collapses onto the spine, which is exactly the range the specificity
argument lives in. The axis is labelled as log and the decade gridlines are
drawn so nobody reads it as linear.

**Every slide is a dot; no box plots.** Four animals a side in Fig A and 23-43
in Fig B. A box plot at n=4 draws quartiles through four points and implies a
distribution that was never measured.

**Two colours in Fig A, one plus an accent in Fig B.** In Fig B the row position
already encodes batch, so nine hues would encode it twice and say nothing; the
negative control is the one row that means something different, so it gets the
accent and the rest recede. Palette is the validated blue/orange pair (CVD
Delta E 24.7, clear of the >=8 gate).
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,      # embed as TrueType so text stays editable
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.linewidth": 0.6,
})

ROOT = Path(__file__).resolve().parent.parent
SURVEY = ROOT / "outputs/survey_9batches.csv"
OUT = ROOT / "figures"
DPI = 300

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
CONTROL_BATCH = "2026-04-20_CCl4"
ACCESSION = re.compile(r"^(R\d{2}-\d{3})")


def load() -> pd.DataFrame:
    d = pd.read_csv(SURVEY)
    d = d[d["macro_pct"].notna()].copy()
    d["diet"] = np.where(d["slide"].str.contains("CHOW", case=False), "chow",
                         np.where(d["slide"].str.contains("NASH", case=False),
                                  "nash", "unknown"))
    return d


def batch_labels(d: pd.DataFrame) -> dict[str, str]:
    """`2025-08-25 (R25-264)`. The accession is derived from the slide names in
    each batch rather than hardcoded, so a relabelled batch cannot end up with
    another batch's accession printed under it."""
    out = {}
    for g, sub in d.groupby("group"):
        accs = {m.group(1) for s in sub["slide"]
                if (m := ACCESSION.match(s.replace("--", "-")))}
        date = re.match(r"(\d{4}-\d{2}-\d{2})", g.replace("P273-", ""))
        stem = date.group(1) if date else g
        out[g] = f"{stem}  ({'/'.join(sorted(accs)) if accs else '?'})"
    return out


def _log_axis(ax, lo=0.03, hi=22.0):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ticks = [0.05, 0.1, 0.5, 1, 5, 10]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=7.5, color=MUTED)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.grid(axis="x", which="major", color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, pad=3)


# ---- Figure A: the diet validation ---------------------------------------

def fig_diet(d: pd.DataFrame) -> None:
    r = d[d["group"] == "2025-03-28_Celina"].sort_values(["diet", "macro_pct"])
    ctrl = d[d["group"] == CONTROL_BATCH]["macro_pct"]
    n_chow = int((r["diet"] == "chow").sum())

    fig, ax = plt.subplots(figsize=(6.1, 3.0))
    _log_axis(ax)

    # The negative control as a band, not a series: it is a reference range
    # from a different cohort, and drawing it as points would invite reading it
    # as a third group in this experiment.
    ax.axvspan(ctrl.min(), ctrl.max(), color="0.90", zorder=1, lw=0)

    ys = np.arange(len(r))[::-1]
    for y, (_, row) in zip(ys, r.iterrows()):
        c = BLUE if row["diet"] == "chow" else ORANGE
        ax.plot([row["macro_pct"]], [y], "o", ms=7, color=c,
                markeredgecolor="white", markeredgewidth=1.2, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([s.replace("R22-354_", "").replace("ACLY656 ", "ACLY656-")
                        for s in r["slide"]], fontsize=7.2, color=INK)
    ax.set_ylim(-1.5, len(r) - 0.2)

    # Group identity as direct labels in the empty right margin, rather than a
    # legend box: nothing here is encoded by colour alone.
    ax.text(19, np.mean(ys[:n_chow]), "chow", fontsize=8.5, color=BLUE,
            weight="bold", va="center", ha="right")
    ax.text(19, np.mean(ys[n_chow:]), "NASH diet", fontsize=8.5, color=ORANGE,
            weight="bold", va="center", ha="right")

    chow_max = float(r[r.diet == "chow"]["macro_pct"].max())
    nash_min = float(r[r.diet == "nash"]["macro_pct"].min())
    # Selective labels: the two points that define the gap, not all eight.
    ax.annotate(f"{chow_max:.2f}%", (chow_max, ys[n_chow - 1]),
                xytext=(chow_max * 1.35, ys[n_chow - 1]), fontsize=7,
                color=MUTED, ha="left", va="center")
    ax.annotate(f"{nash_min:.2f}%", (nash_min, ys[n_chow]),
                xytext=(nash_min * 0.72, ys[n_chow]), fontsize=7,
                color=MUTED, ha="right", va="center")

    # The headline, in the corridor the separation opens up.
    ax.text(np.sqrt(chow_max * nash_min), np.mean(ys[n_chow - 1:n_chow + 1]),
            f"{nash_min / chow_max:.1f}× gap\nAUC 1.000,  p = 0.029",
            ha="center", va="center", fontsize=8, color=INK, linespacing=1.5)

    # Caption for the band, parked under the rows where nothing is plotted.
    ax.text(np.sqrt(ctrl.min() * ctrl.max()), -0.75,
            f"CCl$_4$ negative control (n={len(ctrl)}): {ctrl.min():.2f}–{ctrl.max():.2f}%",
            ha="center", va="center", fontsize=6.8, color=MUTED)

    ax.set_xlabel("macrovesicular fat, % of tissue area   (log scale)",
                  fontsize=8, color=INK, labelpad=5)
    ax.set_title("Detector output against known diet, stain held fixed",
                 fontsize=9.5, color=INK, weight="bold", loc="left", pad=8)
    _save(fig, "Fig_diet_validation")


# ---- Figure B: nine staining batches --------------------------------------

def fig_batches(d: pd.DataFrame) -> None:
    labels = batch_labels(d)
    order = d.groupby("group")["macro_pct"].median().sort_values().index.tolist()
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    _log_axis(ax)

    # Behind the dots and one shade off the surface: it is a reading threshold,
    # not a measurement, and it should not compete with the data.
    ax.axvline(0.5, color="#b9b9b4", lw=0.8, zorder=1)

    for i, g in enumerate(order):
        v = d[d["group"] == g]["macro_pct"].to_numpy()
        y = len(order) - 1 - i
        c = ORANGE if g == CONTROL_BATCH else BLUE
        # Jitter is vertical only, so no dot's x -- the measured value -- moves.
        ax.plot(v, y + rng.uniform(-0.22, 0.22, len(v)), "o", ms=3.4, color=c,
                alpha=0.55, markeredgecolor="none", zorder=3)
        ax.plot([np.median(v)], [y], "|", ms=15, mew=1.8, color=c, zorder=5)
        ax.text(21, y, f"n={len(v)}", fontsize=6.8, color=MUTED,
                va="center", ha="right")

    # Identity in the tick label rather than a floating annotation: the control
    # row is the one that means something different, and a label that travels
    # with the row cannot land on another row's dots.
    ax.set_yticks(range(len(order)))
    ticks = [labels[g] + ("   negative control" if g == CONTROL_BATCH else "")
             for g in order[::-1]]
    ax.set_yticklabels(ticks, fontsize=7.4, color=INK)
    for lab, g in zip(ax.get_yticklabels(), order[::-1]):
        if g == CONTROL_BATCH:
            lab.set_color(ORANGE)
            lab.set_fontweight("bold")
    ax.set_ylim(-1.4, len(order) - 0.3)

    ax.text(0.5, -0.85, "0.5%: below this, a slide-level call is "
                        "indistinguishable from zero",
            fontsize=6.8, color=MUTED, ha="center", va="center")

    ax.set_xlabel("macrovesicular fat per slide, % of tissue area   (log scale)",
                  fontsize=8, color=INK, labelpad=5)
    ax.set_title("259 slides, nine staining runs, parameters unchanged",
                 fontsize=9.5, color=INK, weight="bold", loc="left", pad=8)
    _save(fig, "Fig_nine_batches")


def _save(fig, stem: str) -> None:
    for ext in ("png", "pdf"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=DPI)
        print(f"  {p.relative_to(ROOT)}")
    plt.close(fig)


def main() -> None:
    d = load()
    print(f"{len(d)} slides from {d['group'].nunique()} batches")
    fig_diet(d)
    fig_batches(d)


if __name__ == "__main__":
    main()
