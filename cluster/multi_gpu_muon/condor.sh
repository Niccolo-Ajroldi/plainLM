#!/bin/bash

source ~/miniforge3/etc/profile.d/conda.sh
conda activate plainLM

# Job specific vars
config=$1
job_idx=$2  # CONDOR job array index ranges from 0 to n-1
job_cluster=$3  # CONDOR job cluster

export HOME=/home/najroldi
export TMPDIR=/tmp/najroldi/${job_cluster}/${job_idx}
mkdir -p "$TMPDIR"

mkdir -p "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Launch torch distributed run on 8 devices
torchrun \
  --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 \
  --standalone --nnodes=1 --nproc_per_node=8 \
  train.py --config=$config --job_idx=$job_idx --job_cluster=$job_cluster
