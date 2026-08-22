"""The filenames a review package is made of. Deliberately dependency-free.

Split out of `tileset` so the labelling app can be handed to a pathologist as a
folder that runs on a stock Python. `tileset` builds sets: it needs pandas,
numpy and OpenSlide to read slides and draw from them. `tiles_app` only shows
finished PNGs and appends CSV rows, which is stdlib work -- but it imported
these three strings from `tileset`, and that one import pulled the whole
scientific stack in behind them. A reviewer would then have needed a working
OpenSlide install to look at pictures.
"""

from __future__ import annotations

FEATURE = "ballooning"
MANIFEST_NAME = "manifest.csv"      # what the app reads; no score, no band
FRAME_NAME = "sampling_frame.csv"   # the same rows WITH score/band/split
VERDICTS_NAME = "verdicts.csv"      # appended as she works
