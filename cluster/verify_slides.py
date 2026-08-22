"""Check every slide opens, carries a scale, and reports its tissue area.

Run this the moment a transfer finishes, before submitting anything. A slide
that openslide cannot open, or that carries no MPP, will fail inside an array
task -- after the scheduler has granted the node -- and the failure reads like
a pipeline bug. Catching it here costs seconds per slide and no allocation.

The MPP check is the one that matters. Every filter in this pipeline is
specified in microns; a slide with no embedded scale would either be refused
(correct) or silently processed at the wrong scale if someone had passed
--mpp to work around it. Reporting the scale per slide makes a heterogeneous
cohort visible before it becomes a mysterious batch effect.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("MASHPATH_ROOT", "."))

from mashpath.core.slide import MissingScaleError, Slide  # noqa: E402


def main(argv: list[str]) -> int:
    root = Path(argv[0]) if argv else Path(
        os.environ.get("MASHPATH_SLIDES", "slides")
    )
    paths = sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in {".svs", ".tif", ".tiff", ".ndpi"}
    )
    if not paths:
        print(f"no slides under {root}")
        return 1

    print(f"{len(paths)} slide(s) under {root}\n")
    print(f"{'slide':<34}{'level0':>16}{'mpp':>9}{'obj':>6}  {'GB':>5}  status")
    bad = 0
    mpps: dict[float, int] = {}
    for p in paths:
        size_gb = p.stat().st_size / 1e9
        try:
            s = Slide(str(p))
            w, h = s.dimensions
            mpp = round(s.mpp_x, 4)
            mpps[mpp] = mpps.get(mpp, 0) + 1
            obj = s.objective_power or "?"
            print(f"{p.name:<34}{f'{w}x{h}':>16}{mpp:>9.4f}{obj:>6}  "
                  f"{size_gb:>5.2f}  ok ({s.mpp_source})")
        except MissingScaleError as e:
            bad += 1
            print(f"{p.name:<34}{'-':>16}{'-':>9}{'-':>6}  {size_gb:>5.2f}  "
                  f"NO SCALE: {e}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"{p.name:<34}{'-':>16}{'-':>9}{'-':>6}  {size_gb:>5.2f}  "
                  f"UNREADABLE: {type(e).__name__}: {e}")

    print(f"\n{len(paths) - bad}/{len(paths)} readable")
    if len(mpps) > 1:
        print("\nWARNING: mixed scales in this cohort --")
        for m, n in sorted(mpps.items()):
            print(f"    {m:.4f} um/px  x{n} slide(s)")
        print("  Micron-specified filters stay correct across these, but any")
        print("  per-pixel comparison between them will not. Check whether the")
        print("  groups correspond to scanner or batch before pooling them.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
