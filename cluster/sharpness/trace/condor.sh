#!/bin/bash

source ~/miniforge3/etc/profile.d/conda.sh
conda activate plainLM

# Job
job_idx=$1 # CONDOR job array index ranges from 0 to n-1
job_cluster=$2 # CONDOR job cluster

# Args
paths=${3}
row_idx=${4}
save_path=${5}
save_vectors=${6}
samples=${7}
bsz=${8}
top_n=${9}
max_iter=${10}
tol=${11}

echo "paths=$paths"
echo "row_idx=$row_idx"
echo "save_path=$save_path"
echo "save_vectors=$save_vectors"
echo "samples=$samples"
echo "bsz=$bsz"
echo "top_n=$top_n"
echo "max_iter=$max_iter"
echo "tol=$tol"

export HOME=/home/najroldi
export TMPDIR=/fast/najroldi/tmp/${job_cluster}/${job_idx}
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor_cache"
mkdir -p "$TMPDIR"

python spectra_trace.py \
  --paths=$paths \
  --row_idx=$row_idx \
  --save_path=$save_path \
  --save_vectors=$save_vectors \
  --samples=$samples \
  --bsz=$bsz \
  --top_n=$top_n \
  --max_iter=$max_iter \
  --tol=$tol

