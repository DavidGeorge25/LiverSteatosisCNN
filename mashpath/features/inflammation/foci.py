"""Inflammatory foci as LOCAL EXCESS, not as classified cells.

The obvious design -- classify each nucleus as immune or not, then cluster the
immune ones -- does not work on this material, and the measurement saying so is
worth stating before the code that works around it.

On a CCl4 slide (174,904 nuclei, whole slide), the "immune" class is 48.3% of
all nuclei at 1,134/mm^2, present in every tile. That is not lobular
inflammation; it is the resident sinusoidal lining -- endothelial and Kupffer
nuclei -- which is small and dark exactly like a lymphocyte. The planned fix
was to split them on shape, since lymphocyte nuclei are round and lining nuclei
are flattened. Measured within that class:

    eccentricity p5..p95   0.324 .. 0.898     (smooth, no gap)
    bimodality, ecc        0.501              (< 0.555: one mode)
    bimodality, area       0.530              (< 0.555: one mode)

Both marginals are unimodal. Any cutoff bisects one continuum, and would have
produced a confident-looking "lymphocyte" count that is really a percentile of
the resident population. Separating those cell types on H&E morphometry alone
is not achievable; it wants IHC (CD45, F4/80) or a supervised model trained on
annotated nuclei.

So this module does not try. NASH-CRN scores lobular inflammation by counting
FOCI -- clusters of inflammatory cells -- not by counting cells, and a cluster
is detectable without classifying its members. What distinguishes a focus is
that small dark nuclei are locally much denser than the surrounding parenchyma.
The resident population is roughly uniform, so it sets a baseline; a focus is a
departure from it. That question is answerable on this data, and it degrades
gracefully -- if the baseline is high because the mouse has diffuse infiltrate,
fewer discrete foci are called, which is the correct behaviour rather than a
failure.

    small dark nuclei
      -> local density on a grid, in counts per mm^2
      -> baseline = robust slide-wide density of the same population
      -> excess = local / baseline, thresholded
      -> connected excess regions -> clusters -> foci
      -> portal tracts excluded (they are dense by anatomy, not by disease)

Nothing here needs sklearn; the clustering is a cKDTree flood fill, which is
what DBSCAN reduces to at these sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter, label as cc_label
from scipy.spatial import cKDTree

from ...review.candidates import Candidate
from .classify import IMMUNE
from .config import FociConfig, PortalConfig

FEATURE = "inflammation"


@dataclass
class Focus:
    """One candidate inflammatory focus, in level-0 pixels."""

    x: int
    y: int
    x0: int
    y0: int
    x1: int
    y1: int
    n_cells: int
    density_per_mm2: float
    excess_ratio: float  # local density / slide baseline
    mean_area_um2: float
    mean_eccentricity: float

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass
class FociResult:
    foci: list[Focus] = field(default_factory=list)
    baseline_per_mm2: float = 0.0
    portal_excluded_fraction: float = 0.0
    n_candidate_cells: int = 0
    notes: list[str] = field(default_factory=list)


def _density_grid(
    x_um: np.ndarray, y_um: np.ndarray, bin_um: float, smooth_um: float
) -> tuple[np.ndarray, float, float, float]:
    """Counts per mm^2 on a regular grid. Returns (grid, x0, y0, bin_um)."""
    if x_um.size == 0:
        return np.zeros((1, 1)), 0.0, 0.0, bin_um
    x0, y0 = float(x_um.min()), float(y_um.min())
    nx = max(1, int(np.ceil((x_um.max() - x0) / bin_um)) + 1)
    ny = max(1, int(np.ceil((y_um.max() - y0) / bin_um)) + 1)
    ix = np.clip(((x_um - x0) / bin_um).astype(int), 0, nx - 1)
    iy = np.clip(((y_um - y0) / bin_um).astype(int), 0, ny - 1)
    grid = np.zeros((ny, nx), dtype=np.float64)
    np.add.at(grid, (iy, ix), 1.0)

    if smooth_um > 0:
        # Smoothing in the same physical units as the bin, so the answer does
        # not change when the bin size does.
        grid = gaussian_filter(grid, sigma=max(smooth_um / bin_um, 1e-6))

    # counts per bin -> counts per mm^2
    return grid / ((bin_um / 1000.0) ** 2), x0, y0, bin_um


def detect_portal(
    x_um: np.ndarray,
    y_um: np.ndarray,
    all_x_um: np.ndarray,
    all_y_um: np.ndarray,
    cfg: PortalConfig,
) -> tuple[np.ndarray, float, str]:
    """Mark nuclei sitting in a portal tract. Returns (mask, fraction, note).

    Portal tracts are densely nucleated by anatomy -- bile duct epithelium,
    vessel walls, and a resident lymphocyte population that is NORMAL there.
    Counting them as lobular inflammation would score every healthy liver as
    inflamed, and would do it most strongly on the best-sectioned slides.

    Density here is over ALL nuclei, not just the small dark ones: a portal
    tract is dense in everything, and using the immune subset alone would make
    the exclusion depend on the very classification this module avoids relying
    on.
    """
    if not cfg.enabled or all_x_um.size == 0:
        return np.zeros(x_um.shape, dtype=bool), 0.0, "portal exclusion disabled"

    grid, gx0, gy0, bin_um = _density_grid(
        all_x_um, all_y_um, bin_um=cfg.smoothing_um / 2.0, smooth_um=cfg.smoothing_um
    )
    dense = grid >= cfg.nucleus_density_min

    # Grow the mask by the exclusion radius: the parenchyma immediately around
    # a tract carries spillover lymphocytes that are still portal, not lobular.
    if cfg.exclusion_radius_um > 0:
        pad = int(np.ceil(cfg.exclusion_radius_um / bin_um))
        if pad > 0:
            from scipy.ndimage import binary_dilation, iterate_structure

            struct = iterate_structure(np.ones((3, 3), dtype=bool), pad)
            dense = binary_dilation(dense, structure=struct)

    ix = np.clip(((x_um - gx0) / bin_um).astype(int), 0, dense.shape[1] - 1)
    iy = np.clip(((y_um - gy0) / bin_um).astype(int), 0, dense.shape[0] - 1)
    mask = dense[iy, ix]

    frac = float(dense.mean())
    note = (f"portal exclusion covers {frac:.1%} of the tissue bounding box, "
            f"{int(mask.sum())} of {mask.size} candidate nuclei")
    if frac > cfg.max_excluded_fraction:
        note += (f"  WARNING: above max_excluded_fraction "
                 f"({cfg.max_excluded_fraction:.0%}) -- the exclusion is eating "
                 f"parenchyma, which looks like success and is not. Raise "
                 f"portal.nucleus_density_min.")
    return mask, frac, note


def _cluster(
    x_um: np.ndarray, y_um: np.ndarray, radius_um: float, min_cells: int
) -> np.ndarray:
    """Single-link flood fill within `radius_um`. Returns cluster id per point.

    -1 means unclustered. This is DBSCAN with min_samples=1 on the neighbour
    graph and a size filter afterwards, which at these point counts is both
    faster and one fewer dependency.
    """
    n = x_um.size
    labels = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return labels
    tree = cKDTree(np.column_stack([x_um, y_um]))
    pairs = tree.query_pairs(r=radius_um, output_type="ndarray")

    # Union-find over the neighbour graph.
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra

    roots = np.array([find(i) for i in range(n)])
    uniq, inv, counts = np.unique(roots, return_inverse=True, return_counts=True)
    keep = counts >= min_cells
    remap = np.full(uniq.size, -1, dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    return remap[inv]


def detect_foci(
    nuclei: dict[str, np.ndarray],
    cfg_foci: FociConfig,
    cfg_portal: PortalConfig,
    mpp: float,
    verbose: bool = True,
) -> FociResult:
    """Find inflammatory foci in one slide's nucleus table.

    `nuclei` needs x, y (level-0 px), area_um2, eccentricity and class.
    """
    res = FociResult()
    x_px, y_px = nuclei["x"], nuclei["y"]
    if x_px.size == 0:
        res.notes.append("no nuclei")
        return res

    x_um, y_um = x_px * mpp, y_px * mpp
    is_small_dark = nuclei["class"] == IMMUNE
    res.n_candidate_cells = int(is_small_dark.sum())
    if res.n_candidate_cells == 0:
        res.notes.append("no small-dark nuclei to cluster")
        return res

    cx_um, cy_um = x_um[is_small_dark], y_um[is_small_dark]

    portal_mask, portal_frac, portal_note = detect_portal(
        cx_um, cy_um, x_um, y_um, cfg_portal
    )
    res.portal_excluded_fraction = portal_frac
    res.notes.append(portal_note)
    if verbose:
        print("  " + portal_note)

    keep = ~portal_mask
    kx, ky = cx_um[keep], cy_um[keep]
    idx = np.flatnonzero(is_small_dark)[keep]
    if kx.size == 0:
        res.notes.append("every candidate nucleus fell inside a portal tract")
        return res

    # The baseline is the MEDIAN local density of this same population, which
    # is the resident rate wherever there is no focus. Using the mean would be
    # dragged upward by the foci themselves.
    grid, gx0, gy0, bin_um = _density_grid(
        kx, ky, bin_um=cfg_foci.cluster_radius_um, smooth_um=cfg_foci.cluster_radius_um
    )
    occupied = grid[grid > 0]
    baseline = float(np.median(occupied)) if occupied.size else 0.0
    res.baseline_per_mm2 = baseline
    if verbose:
        print(f"  resident baseline: {baseline:.0f} small-dark nuclei/mm^2")

    if baseline <= 0:
        res.notes.append("baseline density is zero; cannot compute excess")
        return res

    # Only nuclei sitting in genuinely excessive local density are eligible to
    # seed a focus. This is the step that keeps the uniform resident population
    # from clustering into foci everywhere.
    ix = np.clip(((kx - gx0) / bin_um).astype(int), 0, grid.shape[1] - 1)
    iy = np.clip(((ky - gy0) / bin_um).astype(int), 0, grid.shape[0] - 1)
    local = grid[iy, ix]
    excess = local / baseline
    eligible = excess >= cfg_foci.min_excess_ratio
    if verbose:
        print(f"  {int(eligible.sum())}/{kx.size} nuclei above "
              f"{cfg_foci.min_excess_ratio:.1f}x baseline")
    if not eligible.any():
        res.notes.append(
            f"no region reached {cfg_foci.min_excess_ratio:.1f}x the resident "
            "baseline -- no focus called, which on a control slide is correct"
        )
        return res

    ex, ey = kx[eligible], ky[eligible]
    e_idx = idx[eligible]
    e_excess = excess[eligible]
    labels = _cluster(ex, ey, cfg_foci.cluster_radius_um, cfg_foci.min_cells)

    area = nuclei["area_um2"]
    ecc = nuclei["eccentricity"]
    for cid in range(labels.max() + 1 if labels.size else 0):
        sel = labels == cid
        n = int(sel.sum())
        if n < cfg_foci.min_cells or n > cfg_foci.max_cells:
            continue
        fx, fy = ex[sel], ey[sel]
        members = e_idx[sel]
        pad = cfg_foci.bbox_pad_um
        res.foci.append(Focus(
            x=int(round(float(fx.mean()) / mpp)),
            y=int(round(float(fy.mean()) / mpp)),
            x0=int(round((fx.min() - pad) / mpp)),
            y0=int(round((fy.min() - pad) / mpp)),
            x1=int(round((fx.max() + pad) / mpp)),
            y1=int(round((fy.max() + pad) / mpp)),
            n_cells=n,
            density_per_mm2=float(np.median(local[eligible][sel])),
            excess_ratio=float(np.median(e_excess[sel])),
            mean_area_um2=float(area[members].mean()),
            mean_eccentricity=float(ecc[members].mean()),
        ))

    res.foci.sort(key=lambda f: -(f.n_cells * f.excess_ratio))
    if verbose:
        print(f"  {len(res.foci)} focus/foci of >= {cfg_foci.min_cells} cells")
    return res


def to_candidates(res: FociResult, slide_id: str) -> list[Candidate]:
    """Foci as review candidates, highest score first.

    The score is cell count times local excess: a focus is more convincing for
    being both bigger and denser relative to its surroundings than for either
    alone. It is an ordering, not a probability -- which is the whole reason
    these go to a pathologist rather than into a training set directly.
    """
    out: list[Candidate] = []
    for f in res.foci:
        out.append(Candidate(
            slide_id=slide_id,
            feature=FEATURE,
            x=f.x, y=f.y,
            width=max(1, f.width), height=max(1, f.height),
            score=float(f.n_cells * f.excess_ratio),
            measurements={
                "n_cells": float(f.n_cells),
                "density_per_mm2": f.density_per_mm2,
                "excess_ratio": f.excess_ratio,
                "mean_nucleus_area_um2": f.mean_area_um2,
                "mean_eccentricity": f.mean_eccentricity,
            },
        ))
    return out
