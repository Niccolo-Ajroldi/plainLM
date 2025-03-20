#!/bin/bash

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ssm2

export TMPDIR=/fast/najroldi/tmp

# Job specific vars
config=$1
job_idx=$2 # CONDOR job arrays range from 0 to n-1
cluster=$3

# Execute python script
torchrun \
  --redirect 1:0,2:0,3:0 \
  --standalone --nnodes=1 --nproc_per_node=4 \
  eval_ckpt_folder.py \
  --config=$config \
  --job_idx=$job_idx \
  --job_cluster=$cluster
