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

PYTHONPATH=. python data/datasets/prepare.py \
  --out_path="/fast/najroldi/data/lm/fwedu/fwedu_sample_100B_tokenizer_GPTNeoX" \
  --cache_path="/fast/najroldi/tmp" \
  --download --tokenize --chunk \
  --save_tokenized --save_tokenizer \
  --dataset_path="HuggingFaceFW/fineweb-edu" \
  --dataset_split="train" \
  --dataset_name="sample-100BT" \
  --tokenizer="EleutherAI/gpt-neox-20b" \
  --seq_length=2048 \
  --split_train_valid=True \
  --n_tokens_valid=10000000

# # NO CACHE
# PYTHONPATH=. python data/datasets/prepare.py \
#   --out_path="/fast/najroldi/data/lm/fwedu/fwedu_sample_100B_tokenizer_GPTNeoX" \
#   --download --tokenize --chunk \
#   --save_tokenized --save_tokenizer \
#   --dataset_path="HuggingFaceFW/fineweb-edu" \
#   --dataset_split="train" \
#   --dataset_name="sample-100BT" \
#   --tokenizer="EleutherAI/gpt-neox-20b" \
#   --seq_length=2048 \
#   --split_train_valid=True \
#   --n_tokens_valid=10000000
