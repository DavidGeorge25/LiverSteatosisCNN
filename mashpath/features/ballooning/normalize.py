"""Local z-scoring: each cell against the cells physically around it.

This is the step that makes ballooning measurable at all. Raw absolute values
cannot be compared across a slide, let alone across slides, because two effects
dominate them:

*Staining and scanning.* Section thickness, haematoxylin batch and scanner
white balance move every intensity feature by more than ballooning does.

*Zonation.* In a perfectly normal liver, pericentral hepatocytes are already
larger and paler than periportal ones. An absolute pallor threshold would
"find" ballooning in zone 3 of every healthy animal.

Comparing a cell to its neighbours within ~150 um removes both at once, and it
is also what a pathologist actually does at the microscope -- the call is
"bigger and paler than the cells around it", never "bigger than 800 um^2".

Two choices worth defending:

*Robust statistics by default.* Ballooned cells are outliers in their own
neighbourhood by construction. A mean and standard deviation computed over a
window that contains them is inflated by them, which shrinks the very z-scores
being looked for -- worst exactly where ballooning is worst, since a densely
ballooned neighbourhood would normalize its own signal away. Median and
1.4826*MAD do not have that feedback.

*The reference is hepatocytes only.* Sinusoidal endothelium, Kupffer cells and
lymphocytes are small. Leaving them in the reference population drags the local
median area down and makes every ordinary hepatocyte look enlarged.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.spatial import cKDTree

from .config import NormalizationConfig

# Scale factor making MAD a consistent estimator of the standard deviation for
# normally distributed data, so a robust z and a plain z mean the same thing.
_MAD_TO_SIGMA = 1.4826



def _workers() -> int:
    """How many threads scipy may use for a spatial query.

    NOT -1 ("every core on the machine"). On a cluster node that is every core
    on the NODE, not every core in the allocation -- so a job granted 8 cpus
    spawns 64 threads, steals them from whoever else is on the node, and then
    dies: Alliance's BLIS checks the thread count against what it requested and
    calls abort() when they disagree.

      libblis: A different number of threads was created than was requested.

    It aborts in PASS 2, minutes into a run, after every tile has been
    segmented and the GPU work is finished -- so the failure costs the whole
    job and looks nothing like a threading bug.
    """
    for var in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        val = os.environ.get(var)
        if val and val.isdigit() and int(val) > 0:
            return int(val)
    return os.cpu_count() or 1


def neighbour_distance_um(x_um: np.ndarray, y_um: np.ndarray) -> np.ndarray:
    """Distance from each nucleus to the nearest other nucleus, in microns.

    Reported as a feature in its own right because it is what the territory
    area is really measuring (see `segment`'s module docstring), stated in a
    form that does not pretend to be a cell boundary. Ballooned hepatocytes
    push their neighbours apart, so this rises with ballooning -- but it is a
    spacing measurement and is named as one.
    """
    if x_um.size < 2:
        return np.full(x_um.shape, np.nan)
    tree = cKDTree(np.column_stack([x_um, y_um]))
    # k=2 because the nearest point to any cell is itself, at distance 0.
    dists, _ = tree.query(
        np.column_stack([x_um, y_um]), k=2, workers=_workers()
    )
    return dists[:, 1]


def local_zscores(
    x_um: np.ndarray,
    y_um: np.ndarray,
    values: np.ndarray,
    is_reference: np.ndarray,
    cfg: NormalizationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score every column of `values` against a local neighbourhood.

    `values` is (n_cells, n_features). `is_reference` marks the cells that form
    the comparison population -- normally the hepatocytes.

    Returns (z, n_neighbours, local_centre):
      z              (n_cells, n_features), nan where the neighbourhood is too
                     small to support a score
      n_neighbours   reference cells within the radius, excluding self
      local_centre   the local median (or mean) each cell was compared against,
                     kept so a reviewer can see what "normal" was here
    """
    n_cells, n_feat = values.shape
    z = np.full((n_cells, n_feat), np.nan)
    centre_out = np.full((n_cells, n_feat), np.nan)
    counts = np.zeros(n_cells, dtype=np.int32)
    if n_cells == 0:
        return z, counts, centre_out

    ref_idx = np.flatnonzero(is_reference)
    if ref_idx.size == 0:
        return z, counts, centre_out

    ref_pts = np.column_stack([x_um[ref_idx], y_um[ref_idx]])
    ref_vals = values[ref_idx]
    tree = cKDTree(ref_pts)
    query_pts = np.column_stack([x_um, y_um])
    neighbours = tree.query_ball_point(
        query_pts, r=cfg.radius_um, workers=_workers()
    )

    # Position of each cell within the reference array, so a reference cell can
    # be excluded from its own neighbourhood without an O(n) search per cell.
    pos_in_ref = np.full(n_cells, -1, dtype=np.int64)
    pos_in_ref[ref_idx] = np.arange(ref_idx.size)

    for i, nb in enumerate(neighbours):
        if pos_in_ref[i] >= 0:
            # A cell must not be part of the population it is compared against;
            # with a small neighbourhood that alone biases its own z toward 0.
            nb = [j for j in nb if j != pos_in_ref[i]]
        counts[i] = len(nb)
        if len(nb) < cfg.min_neighbours:
            # Not enough context to make a relative call. The cell is kept with
            # its raw features and no score, rather than being handed a z-score
            # computed off a sample of three.
            continue

        block = ref_vals[nb]  # (k, n_feat)
        if cfg.robust:
            centre = np.nanmedian(block, axis=0)
            scale = np.nanmedian(np.abs(block - centre), axis=0) * _MAD_TO_SIGMA
        else:
            centre = np.nanmean(block, axis=0)
            scale = np.nanstd(block, axis=0)

        # A neighbourhood that is uniform in some feature has scale ~0 there,
        # and dividing by it would turn a rounding difference into an enormous
        # z-score. Floor the scale at a fraction of the centre's magnitude.
        floor = np.abs(centre) * cfg.min_scale_fraction
        scale = np.maximum(scale, floor)
        with np.errstate(invalid="ignore", divide="ignore"):
            zi = (values[i] - centre) / scale
        zi[~np.isfinite(scale) | (scale <= 0)] = np.nan
        z[i] = zi
        centre_out[i] = centre

    return z, counts, centre_out


def summarize(counts: np.ndarray, cfg: NormalizationConfig) -> str:
    """One line on how well the neighbourhood assumption held."""
    if counts.size == 0:
        return "no cells to normalize"
    scored = int((counts >= cfg.min_neighbours).sum())
    return (
        f"local normalization: radius {cfg.radius_um:.0f} um, "
        f"median {int(np.median(counts))} reference neighbours; "
        f"{scored}/{counts.size} cells scored "
        f"({scored / counts.size * 100:.1f}%), "
        f"{counts.size - scored} had < {cfg.min_neighbours} and were kept "
        "unscored"
    )
