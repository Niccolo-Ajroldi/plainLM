#!/bin/bash

source ~/miniforge3/etc/profile.d/conda.sh
conda activate plainLM

# Job
job_idx=$1 # CONDOR job array index ranges from 0 to n-1
job_cluster=$2 # CONDOR job cluster

# Args
row_idx=${3}
save_path=${4}

echo "row_idx=$row_idx"
echo "save_path=$save_path"

export HOME=/home/najroldi
export TMPDIR=/fast/najroldi/tmp/${job_cluster}/${job_idx}
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor_cache"
mkdir -p "$TMPDIR"

python interplolate.py \
  --row_idx=$row_idx \
  --save_path=$save_path

