"""Why is each unlabeled white region in a tile unlabeled?

    .venv/bin/python figures/why_unlabeled.py <tile-id>

"Several obvious droplets aren't outlined" is the first thing anyone says about
a pseudo-label figure, and the answer is usually one of three very different
things: the region is below the macrovesicular size floor and is excluded by
design, it is a vessel lumen and is rejected on purpose, or a shape filter
rejected a real droplet. Those need different responses, so this attributes
every unlabeled white region to the specific filter that dropped it.

Measurement comes from the pipeline's own code, so the numbers mean exactly what
`configs/tuned.yaml` means by them.
"""
import csv, sys
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mashpath.config import MashConfig as Config
from mashpath.features.steatosis.detect import extract_components, classify

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "outputs/dataset_v1"
tid = sys.argv[1]
MPP = 0.4953
UM2_PER_PX = MPP ** 2

man = {r["id"]: r for r in csv.DictReader(open(DS / "manifest.csv"))}
sp = man[tid]["split"]
img = cv2.cvtColor(cv2.imread(str(DS / sp / "images" / f"{tid}.png")), cv2.COLOR_BGR2RGB)
lab = cv2.imread(str(DS / sp / "labels" / f"{tid}.png"), cv2.IMREAD_GRAYSCALE) > 127

cfg = Config.from_yaml(ROOT / "configs/tuned.yaml").fat
tissue = np.ones(img.shape[:2], bool)
cs = extract_components(img, tissue, cfg, UM2_PER_PX)

print(f"{tid}   consensus label = {lab.sum()/lab.size*100:.2f}% of tile")
print(f"white components found: {len(cs.components)}   threshold {cs.threshold:.0f}")
print()
hdr = f"{'area um2':>9} {'diam um':>8} {'circ':>5} {'sol':>5} {'ecc':>5} {'border':>6} {'in label':>8}  verdict"
print(hdr); print("-" * len(hdr))

rows = []
for c in cs.components:
    klass, reason = classify(c, cfg)
    inlabel = lab[cs.labels == c.label].mean() > 0.5
    rows.append((c, klass, reason, inlabel))

# only the ones a reader would notice: macro-sized, i.e. >= the size floor
big = [r for r in rows if r[0].area_um2 >= cfg.min_area_um2]
big.sort(key=lambda r: -r[0].area_um2)
for c, klass, reason, inlabel in big:
    verdict = f"MACRO" if klass == "macro" else f"rejected: {reason}"
    print(f"{c.area_um2:9.0f} {c.equivalent_diameter_um:8.1f} {c.circularity:5.2f} "
          f"{c.solidity:5.2f} {c.eccentricity:5.2f} {str(c.touches_border):>6} "
          f"{str(inlabel):>8}  {verdict}")

print()
n_macro = sum(1 for r in big if r[1] == "macro")
n_rej = len(big) - n_macro
print(f"macro-sized components: {len(big)}   accepted {n_macro}   rejected {n_rej}")
from collections import Counter
print("rejection reasons:", Counter(r[2] for r in big if r[1] != "macro"))
small = [r for r in rows if r[0].area_um2 < cfg.min_area_um2]
print(f"below the {cfg.min_area_um2:.0f} um2 macro floor (microvesicular, excluded by design): {len(small)}")
