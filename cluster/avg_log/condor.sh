#!/bin/bash

source ~/miniforge3/etc/profile.d/conda.sh
conda activate plainLM

# Job specific vars
config=$1
job_idx=$2 # CONDOR job arrays range from 0 to n-1
cluster=$3

export HOME="/home/najroldi"
export TMPDIR="/fast/najroldi/tmp/$job_cluster/$job_idx"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor_cache"

# per-job cache isolation
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR"

# Execute python script
python avg_log.py \
  --config=$config \
  --job_idx=$job_idx \
  --job_cluster=$cluster
