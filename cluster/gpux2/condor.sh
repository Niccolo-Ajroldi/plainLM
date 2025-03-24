#!/bin/bash

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ssm2

export TMPDIR=/fast/najroldi/tmp

# Job specific vars
config=$1
job_idx=$2  # CONDOR job array index ranges from 0 to n-1
job_cluster=$3  # CONDOR job cluster

# Execute python script
torchrun \
  --redirect 1:0 \
  --standalone --nnodes=1 --nproc_per_node=2 \
  train.py --config=$config \
  --job_idx=$job_idx --job_cluster=$job_cluster
