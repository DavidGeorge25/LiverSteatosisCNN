"""Whole-slide viewer: an exhaustive, native-resolution measurement and a way to look at it.

`precompute.py` produces the number and the mask; `server.py` serves them;
`dzi.py` is the tile geometry both the tissue and the overlay answer to.
Nothing here re-runs the model at a display scale -- see `dzi` for why that
distinction is the whole design.
"""
