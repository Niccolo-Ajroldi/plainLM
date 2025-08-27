#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /home/atatjer/src/_plainLM/.venv/bin/activate

# Job specific vars
config=$1
job_idx=$2  # CONDOR job array index ranges from 0 to n-1
job_cluster=$3  # CONDOR job cluster

export HOME=/home/atatjer
export TMPDIR=/fast/atatjer/tmp/${job_cluster}/${job_idx}
mkdir -p "$TMPDIR"

# Launch torch distributed run on 8 devices
torchrun \
  --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 \
  --standalone --nnodes=1 --nproc_per_node=2  train.py --config=$config --job_idx=$job_idx --job_cluster=$job_cluster
