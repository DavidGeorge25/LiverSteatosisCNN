"""Shared machinery: slide reading, tissue detection, tiling, config, viz, io.

Everything here is feature-agnostic. If a change here would only ever matter to
one of steatosis/ballooning/inflammation, it belongs in that feature instead.

THREADING. OpenCV is set single-threaded at import, once, for everything that
imports `mashpath.core`. This is not a micro-optimisation -- it is the fix for a
hang that cost most of a night of cluster time on 2026-08-29.

On Fir, `export_slide` ran 90 tiles in 15 seconds and then stopped dead mid-loop
for the remaining 59 minutes of its wall clock, writing nothing and burning 95%
of one core. 54 of 252 slides did it, all of them the large ones (623 MB-1.4 GB;
the 133-283 MB slides never did), and the same slides complete in ~26 s on a
laptop. `cluster/fir_env.sh` documents the same signature for the ballooning
pipeline -- "stopped mid-tile ... it is the thread interaction, not the data" --
and pins OpenBLAS, BLIS and OMP to one thread each. OpenCV keeps its OWN pool,
which none of those variables touch, and on a 2-CPU allocation its workers
contend with the OMP team skimage and scipy bring.

Set to 0 rather than 1: `setNumThreads(0)` disables the parallel framework
outright, where 1 still dispatches through it. It costs nothing here -- every
filter in this codebase runs on a 512x512 tile and is memory-bound at that size.
Measured on 60 real tiles through the full nine-member ensemble: 193 ms/tile
with OpenCV's default 10 threads against 195 ms/tile single-threaded, a 1%
difference. The pool was never buying anything to disable.
"""

from __future__ import annotations


def _single_threaded_opencv() -> None:
    """Called at import. Idempotent, and silent when cv2 cannot oblige."""
    try:
        import cv2
    except Exception:          # a missing cv2 is a different, louder problem
        return
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass                   # a build without threading support: nothing to do


_single_threaded_opencv()
