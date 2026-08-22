"""Prime the StarDist weight cache. Run ONCE on a login node before any job.

Compute nodes have no outbound network. `StarDist2D.from_pretrained()` silently
assumes it can reach the internet, so a job that has never seen the weights
fails several minutes in -- after the scheduler has granted an H100 and the
slide has been opened -- with a connection error that reads like a bug in the
pipeline rather than a missing cache.

Priming is cheap and idempotent. Run it after any change to $KERAS_HOME.
"""

import os
import sys

sys.path.insert(0, os.environ.get("MASHPATH_ROOT", "."))

from mashpath.core.segmentation import DEFAULT_MODEL, load_model  # noqa: E402


def main() -> int:
    cache = os.environ.get("KERAS_HOME", "~/.keras")
    print(f"caching {DEFAULT_MODEL} into {cache}")
    model = load_model(DEFAULT_MODEL)
    print(f"  ok: {model.name}")
    print(f"  thresholds: prob={model.thresholds.prob:.6f} nms={model.thresholds.nms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
