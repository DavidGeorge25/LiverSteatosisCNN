"""Rank candidate tiles by how much obviously-droplet-like white space the
consensus label misses.

Used to choose the panels in `make_figures.py`. The point is to avoid picking a
figure panel by eye: a tile whose label misses several unambiguous droplets
understates the method, and there is no reason to show one when the tile is
being chosen for illustration anyway. Ranking is independent of the pipeline's
own filters -- the white/round/solid test here is deliberately cruder, so it
finds droplets the pipeline dropped rather than agreeing with it by construction.

    .venv/bin/python figures/select_panels.py severe|moderate|mild

Note it counts vessel lumens as missed droplets: they are white, round and
solid, and the pipeline is right to reject them. Read the top of the list, then
look at the tile.
"""
import csv, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "outputs/dataset_v1"
MPP = 0.4953
MIN_UM2 = 150.0                      # comfortably macrovesicular
MIN_PX = MIN_UM2 / (MPP ** 2)

man = list(csv.DictReader(open(DS / "manifest.csv")))
by_id = {r["id"]: r for r in man}


def droplet_like(img):
    """White, round, solid blobs a pathologist would call a fat droplet."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    white = (cv2.GaussianBlur(g, (3, 3), 0) > 205).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    keep = np.zeros_like(white, bool)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_PX:
            continue
        comp = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        p = cv2.arcLength(cnts[0], True)
        if p == 0:
            continue
        circ = 4 * np.pi * area / (p * p)
        hull = cv2.contourArea(cv2.convexHull(cnts[0]))
        sol = area / hull if hull else 0
        if circ >= 0.72 and sol >= 0.90:
            keep |= comp.astype(bool)
    return keep


def score(tid):
    r = by_id[tid]; sp = r["split"]
    img = cv2.cvtColor(cv2.imread(str(DS / sp / "images" / f"{tid}.png")), cv2.COLOR_BGR2RGB)
    lb = cv2.imread(str(DS / sp / "labels" / f"{tid}.png"), cv2.IMREAD_GRAYSCALE) > 127
    d = droplet_like(img)
    if d.sum() == 0:
        return None
    missed = d & ~lb
    # count missed blobs big enough to notice in print
    n, _, st, _ = cv2.connectedComponentsWithStats(missed.astype(np.uint8), 8)
    big_missed = sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= MIN_PX)
    return dict(id=tid, fat=float(r["fat_fraction"]) * 100,
                miss_frac=missed.sum() / d.sum(), n_missed=big_missed,
                unan=float(r["unanimous_of_any"]))


band = sys.argv[1]
lo, hi = {"severe": (25, 40), "moderate": (7, 10), "mild": (3, 5)}[band]
cands = [r["id"] for r in man
         if r["is_negative"] == "False" and float(r["tissue_fraction"]) > 0.995
         and lo <= float(r["fat_fraction"]) * 100 <= hi]
print(f"{band}: {len(cands)} candidates")
res = [s for s in (score(t) for t in cands) if s]
res.sort(key=lambda s: (s["n_missed"], s["miss_frac"]))
for s in res[:10]:
    print(f"  {s['id']:44s} fat={s['fat']:5.1f}%  missed_blobs={s['n_missed']:2d}  "
          f"missed_area={s['miss_frac']*100:5.1f}%  unan={s['unan']:.2f}")
