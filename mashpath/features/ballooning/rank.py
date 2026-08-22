"""Scoring cells, and choosing which ones a pathologist actually sees.

The score is a weighted sum of LOCAL z-scores on the axes that define
ballooning: bigger than its neighbours, paler than its neighbours, and more
heterogeneous in cytoplasmic texture than its neighbours. Nothing here is
calibrated -- there is no ground truth to fit weights to yet, which is the
whole reason the review round exists. The weights encode a hypothesis about
which direction each feature moves, and the confirmed set is what will replace
them.

The sampling is the part that is easy to get wrong. Handing over the top N
alone produces a review set made entirely of extremes, and a classifier trained
on it learns a boundary no human was ever shown: every confirmed positive is a
florid balloon, every implicit negative is whatever was not sampled, and the
hard middle -- where the actual decisions live -- is absent from the training
data entirely. So mid-ranked and low-ranked cells are sampled deliberately and
mixed in, and the manifest is shuffled so the reviewer cannot infer the score
from the row order and mark accordingly.
"""

from __future__ import annotations

import numpy as np

from .config import RankingConfig

# Bands a candidate can be drawn from. Recorded per candidate so the confirmed
# set can be analysed by band -- "what fraction of mid-band cells did the
# pathologist confirm" is the number that says whether the score is any good.
BAND_TOP = "top"
BAND_MID = "mid"
BAND_LOW = "low"
# Cells the detector vetoed. Sampled so the vetoes themselves get reviewed.
BAND_VETOED = "vetoed"


def score_cells(
    z: np.ndarray, z_names: list[str], cfg: RankingConfig
) -> tuple[np.ndarray, list[str]]:
    """Weighted sum of local z-scores. Returns (score, weights_used).

    A weight naming a feature that was not computed is an error worth
    surfacing, not something to silently skip: a typo in the config would
    otherwise quietly drop an axis and change every ranking without a word.
    """
    missing = [k for k in cfg.weights if k not in z_names]
    if missing:
        raise KeyError(
            f"ranking.weights names features that were not computed: {missing}. "
            f"available: {sorted(z_names)}"
        )

    idx = {name: i for i, name in enumerate(z_names)}
    score = np.zeros(z.shape[0])
    used = []
    for name, w in cfg.weights.items():
        if w == 0:
            continue
        col = z[:, idx[name]]
        # A cell missing one feature keeps its score on the others rather than
        # being knocked out of the ranking entirely; nan would poison the sum.
        score += w * np.nan_to_num(col, nan=0.0)
        used.append(name)
    # A cell with no usable z-scores at all is unrankable, not average.
    unscored = np.all(np.isnan(z[:, [idx[n] for n in used]]), axis=1) if used else \
        np.ones(z.shape[0], dtype=bool)
    score[unscored] = np.nan
    return score, used


def apply_vetoes(
    score: np.ndarray,
    territory_area: np.ndarray,
    local_area_centre: np.ndarray,
    cyto_gray_mean: np.ndarray,
    has_nucleus: np.ndarray,
    is_hepatocyte: np.ndarray,
    solidity: np.ndarray,
    eccentricity: np.ndarray,
    cfg: RankingConfig,
) -> tuple[np.ndarray, dict[str, int]]:
    """Absolute sanity rules, applied whatever the z-scores say.

    These are not ballooning criteria -- they are the small set of things that
    cannot be a ballooned hepatocyte no matter how well they score, and each
    one exists because of a specific failure mode:

      not enlarged  ballooning IS enlargement. Without this the ranking
                    fills with hepatocytes compressed into slivers between fat
                    droplets -- pale and heterogeneous because they are a rim
                    against a white void, while being smaller than their
                    neighbours
      not rounded   same failure, caught on shape: a sliver is ragged and
                    elongated where a balloon is convex and round
      area ratio    a cell many times its local median is a segmentation merge
      pallor        a "cytoplasm" brighter than a fat void IS a fat void, and
                    would otherwise let macrovesicular droplets rank as balloons
      nucleus       ballooned cells keep their nucleus, displaced against the
                    membrane; a territory without one is empty space
      hepatocyte    only hepatocytes can balloon
    """
    vetoed = np.zeros(score.shape, dtype=bool)
    counts: dict[str, int] = {}

    def veto(mask: np.ndarray, name: str) -> None:
        fresh = mask & ~vetoed
        counts[name] = int(fresh.sum())
        vetoed[fresh] = True

    veto(~is_hepatocyte, "not_hepatocyte")
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = territory_area / local_area_centre
    veto(np.isfinite(ratio) & (ratio < cfg.min_area_ratio), "not_enlarged")
    veto(solidity < cfg.min_solidity, "not_convex")
    veto(eccentricity > cfg.max_eccentricity, "too_elongated")
    veto(np.isfinite(ratio) & (ratio > cfg.max_area_ratio), "merged")
    veto(cyto_gray_mean > cfg.max_cyto_gray_mean, "too_bright_is_void")
    if cfg.require_nucleus:
        veto(~has_nucleus, "no_nucleus")

    out = score.copy()
    out[vetoed] = np.nan
    return out, counts


def stratified_select(
    score: np.ndarray, cfg: RankingConfig, vetoed: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the review set. Returns (indices, band label per index).

    The bands are DISJOINT PERCENTILE WINDOWS, not "top N then sample what is
    left". The earlier version took the top `top_n` by rank and then tried to
    sample the 40-60th and 0-20th percentiles from the remainder, which is
    fine while `top_n` is small against the scorable population and silently
    degenerate when it is not. Measured on a 40-tile MASH run: 242 cells were
    scorable, `top_n=150` consumed everything above the 38th percentile, and
    the middle band came back EMPTY -- so the reviewer would have seen only
    florid extremes and near-zeros on the one slide that has disease, with
    nothing from the decision boundary the classifier has to learn.

    Cutting the range into windows first and sampling inside each makes that
    unrepresentable: a band can be short, but it cannot be cannibalised by the
    band above it.

    `vetoed` marks cells the detector REJECTED outright (not enlarged, not
    convex, too bright to be cytoplasm...). A few are sampled deliberately.
    Without them every candidate a pathologist sees has already passed the
    vetoes, the confirmed set contains no evidence about whether those
    rejections were correct, and a veto that is quietly discarding real
    ballooning is invisible -- it removes ~70% of cells on this cohort, which
    is far too much to take on trust.
    """
    rng = np.random.default_rng(cfg.seed)
    valid = np.flatnonzero(np.isfinite(score))
    if valid.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=object)

    sv = score[valid]
    picked: list[int] = []
    bands: list[str] = []

    def sample_window(lo_pct: float, hi_pct: float, n: int, band: str) -> None:
        """Sample `n` from a half-open percentile window of the score."""
        if n <= 0:
            return
        lo, hi = np.percentile(sv, [lo_pct, hi_pct])
        # Half-open above so adjacent windows cannot both claim a boundary
        # cell; the topmost window closes so the maximum is reachable.
        inside = (sv >= lo) & (sv <= hi if hi_pct >= 100.0 else sv < hi)
        pool = valid[inside]
        if pool.size == 0:
            return
        chosen = rng.choice(pool, size=min(n, pool.size), replace=False)
        for i in np.atleast_1d(chosen).tolist():
            picked.append(int(i))
            bands.append(band)

    # Highest-scoring window first, but as a WINDOW -- the top band is still
    # the most ballooned-looking cells, it just cannot eat the ones below it.
    sample_window(cfg.top_percentile_min, 100.0, cfg.top_n, BAND_TOP)
    sample_window(*cfg.mid_percentile_range, cfg.mid_n, BAND_MID)
    sample_window(0.0, cfg.low_percentile_max, cfg.low_n, BAND_LOW)

    if vetoed is not None and cfg.vetoed_n > 0:
        pool = np.flatnonzero(vetoed)
        if pool.size:
            chosen = rng.choice(pool, size=min(cfg.vetoed_n, pool.size),
                                replace=False)
            for i in np.atleast_1d(chosen).tolist():
                picked.append(int(i))
                bands.append(BAND_VETOED)

    idx = np.asarray(picked, dtype=int)
    band_arr = np.asarray(bands, dtype=object)

    # Shuffle so the manifest's row order carries no information about the
    # score. A reviewer working down a list sorted by confidence will mark the
    # top differently from the bottom whether or not they mean to.
    perm = rng.permutation(idx.size)
    return idx[perm], band_arr[perm]


def summarize(
    score: np.ndarray, vetoes: dict[str, int], bands: np.ndarray
) -> str:
    """One block on what was rankable and what got sent for review."""
    finite = int(np.isfinite(score).sum())
    lines = [
        f"ranked:  {finite}/{score.size} cells scorable"
        + (f"; vetoed " + ", ".join(f"{k}={v}" for k, v in vetoes.items() if v)
           if any(vetoes.values()) else "")
    ]
    if bands.size:
        counts = {b: int((bands == b).sum())
                  for b in (BAND_TOP, BAND_MID, BAND_LOW, BAND_VETOED)}
        lines.append(
            f"review:  {bands.size} candidates "
            f"(top={counts[BAND_TOP]}, mid={counts[BAND_MID]}, "
            f"low={counts[BAND_LOW]}, vetoed={counts[BAND_VETOED]}), "
            "order shuffled"
        )
    return "\n".join(lines)
