"""mashpath -- computational pathology for the mouse MASH model.

Three NASH-CRN features, one shared core:

  steatosis     fat droplets leave white voids in H&E; found by thresholding
                plus shape filters, and fully pseudo-labelled with no human
                annotation.
  ballooning    enlarged pale rounded hepatocytes; no void, no clean geometric
                signature, judged relative to the neighbouring cells.
  inflammation  clusters of small dark immune cells in the parenchyma,
                confounded by portal tracts, which contain them normally.

Only steatosis can be pseudo-labelled outright. The other two generate
*candidates* for a pathologist to confirm (see `mashpath.review`); the model is
then trained on the confirmed set.
"""

__version__ = "0.2.0"
