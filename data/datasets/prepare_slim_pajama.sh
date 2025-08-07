#!/bin/bash

# This script will download and preprocess SlimPajama-627B.
# Expect some token loss by batched concat_chunk.

mkdir -p /fast/najroldi/tmp
cd ~/plainLM

# TRAIN SET
PYTHONPATH=. python data/datasets/prepare.py \
  --out_path="/fast/najroldi/data/lm/slim_pajama/sp_tokenizer_GPTNeoX" \
  --cache_path="/fast/najroldi/tmp" \
  --download --tokenize --chunk \
  --save_raw --save_tokenized --save_tokenizer \
  --dataset_path="cerebras/SlimPajama-627B" \
  --dataset_split="train" \
  --dataset_name="sample-100BT" \
  --tokenizer="EleutherAI/gpt-neox-20b" \
  --seq_length=2048

# VALID SET
PYTHONPATH=. python data/datasets/prepare.py \
  --out_path="/fast/najroldi/data/lm/slim_pajama/sp_tokenizer_GPTNeoX" \
  --cache_path="/fast/najroldi/tmp" \
  --download --tokenize --chunk \
  --save_raw --save_tokenized --save_tokenizer \
  --dataset_path="cerebras/SlimPajama-627B" \
  --dataset_split="validation" \
  --dataset_name="sample-100BT" \
  --tokenizer="EleutherAI/gpt-neox-20b" \
  --seq_length=2048
