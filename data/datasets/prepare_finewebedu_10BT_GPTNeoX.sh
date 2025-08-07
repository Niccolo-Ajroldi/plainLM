#!/bin/bash

# This script will download and preprocess FineWebEdu-100BT.
# Expect some token loss by batched concat_chunk.

export SOFT_FILELOCK=1
export HF_HOME=/fast/najroldi/hf_fast
export TMPDIR=/fast/najroldi/tmp
export HOME=/fast/najroldi/tmp 

mkdir -p /fast/najroldi/tmp
mkdir -p /fast/najroldi/hf_fast
cd ~/plainLM

PYTHONPATH=. python -m pdb data/datasets/prepare.py \
  --out_path="/fast/najroldi/data/lm/fwedu/fwedu_sample_100BT" \
  --cache_path="/fast/najroldi/tmp" \
  --chunk \
  --tokenizer="EleutherAI/gpt-neox-20b" \
  --seq_length=2048 \
  --split_train_valid=True \
  --n_tokens_valid=10000000
