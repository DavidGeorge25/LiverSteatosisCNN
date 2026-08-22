#!/bin/bash
# Build the mashpath environment on an Alliance cluster (Fir / Nibi / Rorqual).
#
# Run ONCE, on a login node:   bash cluster/setup_fir.sh
#
# Why not `pip install -r requirements.txt`: the PyPI TensorFlow wheel is built
# against a CUDA that the cluster does not provide. It installs without
# complaint and then runs StarDist on CPU -- a ~30x slowdown that reports
# itself as nothing at all. The Alliance wheelhouse ships a TF built against
# the cluster's CUDA, and `--no-index` is what forces pip to use it.
set -euo pipefail

VENV="${VENV:-$HOME/venvs/mashpath}"

module --force purge
module load StdEnv/2023
module load gcc opencv/4.11.0 python/3.11 openslide/4.0.0 cuda/12.2

python -m venv --clear "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip

# From the Alliance wheelhouse: everything with a compiled/CUDA component.
pip install --no-index \
    numpy scipy pandas matplotlib scikit-image tifffile pyyaml tqdm numba \
    tensorflow openslide-python

# StarDist and csbdeep are pure Python and are not always mirrored; allow PyPI
# for these two only, and --no-deps so they cannot drag in a second TensorFlow
# on top of the cluster build.
pip install --no-deps stardist csbdeep

python - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"tensorflow {tf.__version__}, GPUs visible: {len(gpus)}")
if not gpus:
    print("  NOTE: no GPU here -- expected on a login node. Check inside a job.")
from stardist.models import StarDist2D
print("stardist import OK")
import openslide; print("openslide", openslide.__library_version__)
PY

echo
echo "environment ready: $VENV"
echo "Next: cache the StarDist weights while you still have a network:"
echo "  source $VENV/bin/activate && python cluster/cache_model.py"
