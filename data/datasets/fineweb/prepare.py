"""
  This script downloads FineWeb, tokenize it, and groups it in blocks of length (seq_len+1).
  The tokenizer and the chunked dataset are saved.

  Insipred by:
  https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
  https://github.com/JonasGeiping/cramming/blob/main/cramming/data/pretraining_preparation.py

  On the role of EOS:
  https://discuss.huggingface.co/t/how-does-gpt-decide-to-stop-generating-sentences-without-eos-token/41623/2

- cache_dir="~/tmp" -> Disk quota exceeded
- cache_dir=None -> Disk quota exceeded

Currently hitting disk quota or Flock errors when converting from IterableDataset to Dataset.

```
mkdir /tmp/najroldi
export HF_HOME=/tmp/najroldi
export HF_DATASETS_CACHE=/tmp/najroldi
```
"""

import os
from itertools import chain
from functools import partial

from datasets import load_dataset, Dataset
from transformers import GPT2Tokenizer
from timeit import default_timer as timer

timer_start = timer()

# --------------------------------------------------------------------
# Config

# Path wwhere to save dataset
out_path = "/fast/najroldi/data/lm/fineweb/fw_10B_tokens_ctx1024"

# HF dataset name
dataset_name = "HuggingFaceFW/fineweb"

nrows = 17_000_000  # ~11B tokens
# nrows = 31_000_000  # ~20B tokens

seq_len = 1024
max_seq_length = seq_len+1
shuffle_raw_data = True
map_setup = dict(
  batched=True,
  batch_size=1024,
  num_proc=8
)

# --------------------------------------------------------------------
# Load Dataset

print("Loading Dataset")

# Load in streaming mode, creates an IterableDataset
raw_dataset = load_dataset(
  dataset_name,
  split = 'train',  # only split available for Fineweb
  streaming = True,
  columns=["text"],  # it works only for Parquet datasets in streaming mode: https://github.com/huggingface/datasets/issues/4114#issuecomment-1090484293
  ## cache_dir defaults to "~/.cache/huggingface/datasets"
  # cache_dir="/tmp/najroldi"
)

print("From IterableDataset to Dataset")

# From IterableDataset to Dataset
iterable_ds = raw_dataset.take(nrows)
def gen_from_iterable_dataset(iterable_ds):
  yield from iterable_ds
partial_obj = partial(gen_from_iterable_dataset, iterable_ds)
dataset = Dataset.from_generator(
  partial_obj, 
  features=iterable_ds.features,
  
  ## cache_dir defaults to "~/.cache/huggingface/datasets"
  # cache_dir="/fast/najroldi/tmp",  # FileSystem does not appear to support flock
  # cache_dir="/home/najroldi/tmp",  # Disk quota exceeded
  # cache_dir="/tmp",  # No space left on device 
  
  ## Fails as well: Disk quota exceeded
  # keep_in_memory=True,
  # cache_dir=None,
)

# Shuffle so that multiproc has shards of similar size
if shuffle_raw_data:
  dataset = dataset.shuffle(seed=1996)

# --------------------------------------------------------------------
# Tokenize, adding EOS at the end of document.

print("Tokenize")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print(f"Length of tokenizer = {len(tokenizer)}")  # should be 50257 for GPT2

def tokenize_function(examples):
  eos_token = tokenizer.eos_token
  add_eos = lambda seq: (eos_token + seq + eos_token) if seq else seq
  add_eos_batched = lambda seqs: [add_eos(seq) for seq in seqs]
  tokenized_output = tokenizer(
      add_eos_batched(examples["text"]),
      add_special_tokens=False,
      return_special_tokens_mask=False,
      return_attention_mask=False
  )
  return tokenized_output

# Set an high maximum number of tokens that the tokenizer can handle 
# in a single input sequencem to prevent truncation during tokenization.
tokenizer.model_max_length = 1e30 

# Tokenize
tokenized_datasets = dataset.map(
  tokenize_function, 
  remove_columns=['text'],
  **map_setup
)

# Reset to correct value
tokenizer.model_max_length = seq_len

print("Saving Tokenized Data")
tokenized_datasets.save_to_disk(os.path.join(out_path, f"tokenized_data"))

print("Saving Tokenizer")
tokenizer.save_pretrained(os.path.join(out_path, "tokenizer"))

# --------------------------------------------------------------------
# Concat in chunks of max_seq_len.

print("Concatenating in chunks of max_seq_len")

# Main data processing function that will concatenate all texts 
# from tokenized dataset and generate chunks of max_seq_length.
# NOTE: expected token loss by batched concat_chunk, 
# it truncates leftover tokens that don't fill a full max_seq_length chunk.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= max_seq_length:
        total_length = (total_length // max_seq_length) * max_seq_length
    # Split by chunks of max_len.
    result = {
      k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)] 
      for k, t in concatenated_examples.items()
    }
    return result

# Concat in chunks
lm_datasets = tokenized_datasets.map(
  group_texts,
  **map_setup
)

n_tokens = len(lm_datasets) * max_seq_length
print(f"Number of tokens in dataset: {n_tokens:_}")

# --------------------------------------------------------------------

# Cast to tensors
lm_datasets.set_format("torch")

# --------------------------------------------------------------------

# Shuffle and extract train and valid sets
# NOTE: potential single-document contamination here, similar to: 
# https://github.com/KellerJordan/modded-nanogpt/blob/master/data/fineweb.py
# A document might be splitted across train and valid splits.

N_valid = 1_000  # ~100M tokens when seq_len=1024
valid_set = lm_datasets.select(range(N_valid))
train_set = lm_datasets.select(range(N_valid, len(lm_datasets)))

print(f"Number of tokens in train_set: {len(train_set) * max_seq_length:_}")
print(f"Number of tokens in valid_set: {len(valid_set) * max_seq_length:_}")

train_set = train_set.shuffle(seed=96)
valid_set = valid_set.shuffle(seed=96)

# Save
train_set.save_to_disk(os.path.join(out_path, 'train'))
valid_set.save_to_disk(os.path.join(out_path, 'valid'))

print("Shuffled, split done and saved!")

# --------------------------------------------------------------------

timer_end = timer()
elapsed = timer_end - timer_start

print(f"Elapsed: {elapsed} sec")
print(f"Elapsed: {elapsed // 60} min")
print(f"Elapsed: {elapsed // 3600} h")
