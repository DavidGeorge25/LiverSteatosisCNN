#!/bin/bash
#SBATCH --job-name=mash-bal-final
#SBATCH --account=def-wangd123
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/%u/logs/balfinal_%j.out
#SBATCH --error=/scratch/%u/logs/balfinal_%j.out
#
# Pass 2: merge the chunks, z-score every cell against its spatial neighbours,
# rank, and write the review package.
#
# NO GPU. StarDist is finished by this point; what remains is a cKDTree query
# and some pandas. It took 3.8 s on a 40-tile slide. Requesting a GPU here
# would queue behind the GPU partition for nothing.
#
# More memory than a chunk task, because this is the one step that holds every
# cell on the slide at once -- that is inherent, not incidental: the z-score
# neighbourhood crosses tile boundaries, so it cannot be chunked.
#
# `merge_chunks` refuses to run if the tile ranges have a gap. A hole would
# silently change every z-score near it, and the run would look successful.
set -euo pipefail
source "${MASHPATH_ROOT:-/scratch/$USER/mashpath}/cluster/fir_env.sh"
mkdir -p "/scratch/$USER/logs"
: "${SLIDE:?set SLIDE=/path/to/slide.svs via --export}"

cd "$MASHPATH_ROOT"
time python -m mashpath.features.ballooning.cli finalize \
    --slide "$SLIDE" \
    --config configs/core.yaml --config configs/ballooning.yaml --config configs/production.yaml \
    --output-dir "$MASHPATH_OUT/ballooning"
