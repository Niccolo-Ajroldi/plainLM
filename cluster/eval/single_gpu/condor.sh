#!/bin/bash

source ~/miniforge3/etc/profile.d/conda.sh
conda activate plainLM

export HOME=/home/najroldi
export TMPDIR=/fast/najroldi/tmp

# Job specific vars
config=$1
job_idx=$2 # CONDOR job arrays range from 0 to n-1
cluster=$3

# Execute python script
python avg_and_eval.py \
  --config=$config \
  --job_idx=$job_idx \
  --job_cluster=$cluster
