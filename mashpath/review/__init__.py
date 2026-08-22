"""Pathologist review of machine-generated candidates.

Steatosis can be pseudo-labelled outright: fat is a white void with a clean
geometric signature, so a threshold plus shape filters produces training labels
with no human in the loop. Ballooning and lobular inflammation cannot. There is
no threshold that finds them -- ballooning is judged relative to neighbouring
cells, and inflammation is confounded by portal tracts that contain immune
cells normally.

So those two features *propose* and a pathologist *disposes*: the detector
emits candidates with a context crop and an overlay, the pathologist marks each
one confirmed or rejected, and the model trains on the confirmed set.
"""

from .candidates import Candidate, CandidateSet
from .export import export_candidates, load_verdicts, stratified_sample

__all__ = [
    "Candidate",
    "CandidateSet",
    "export_candidates",
    "load_verdicts",
    "stratified_sample",
]
