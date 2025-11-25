#!/bin/bash

source ~/miniforge3/etc/profile.d/conda.sh
conda activate plainLM

# Job specific vars
config=$1
job_idx=$2  # CONDOR job array index ranges from 0 to n-1
job_cluster=$3  # CONDOR job cluster
n_gpus=$4

export HOME=/home/najroldi
export TMPDIR=/fast/najroldi/tmp/${job_cluster}/${job_idx}
mkdir -p "$TMPDIR"

# Launch torch distributed run on 8 devices
torchrun \
  --standalone --nnodes=1 --nproc_per_node=$n_gpus \
  train.py --config=$config --job_idx=$job_idx --job_cluster=$job_cluster
