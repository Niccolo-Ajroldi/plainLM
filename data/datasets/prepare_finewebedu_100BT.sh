#!/bin/bash

# This script will download and preprocess FineWebEdu-100BT.
# Expect some token loss by batched concat_chunk.


mkdir -p "~/tmp"
cd ~/plainLM
source .venv/bin/activate


PYTHONPATH=. python data/datasets/prepare.py \
  --out_path="~/data/lm/fwedu/fwedu_sample_100B_tokenizer_GPTNeoX" \
  --cache_path="~/tmp" \
  --download --tokenize --chunk \
  --save_tokenized --save_tokenizer \
  --dataset_path="HuggingFaceFW/fineweb-edu" \
  --dataset_split="train" \
  --dataset_name="sample-100BT" \
  --tokenizer="EleutherAI/gpt-neox-20b" \
  --seq_length=2048 \
  --split_train_valid=True \
  --n_tokens_valid=10000000
