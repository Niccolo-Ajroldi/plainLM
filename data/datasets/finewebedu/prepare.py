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

from functools import partial

from datasets import load_dataset, Dataset
from transformers import GPT2Tokenizer
from timeit import default_timer as timer

from utils import concat_chunck

if __name__ == "__main__":
  
  timer_start = timer()

  # --------------------------------------------------------------------
  # Config

  # Path wwhere to save dataset
  out_path = "/fast/najroldi/data/lm/finewebedu/fwedu_sample10B_ctx2048"

  # HF dataset name
  dataset_name = "HuggingFaceFW/fineweb-edu"
  # dataset_name = "HuggingFaceFW/fineweb"

  streaming = False

  # Ignored if streaming=False
  # 10B sample has roughly 97M rows
  # nrows = None
  nrows = 50_000_000

  seq_len = 2048
  max_seq_length = seq_len+1
  shuffle_raw_data = True
  map_setup = dict(
    batched=False,
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
    name='sample-10BT',
    streaming=streaming,
    columns=["text"],  # it works only for Parquet datasets in streaming mode: https://github.com/huggingface/datasets/issues/4114#issuecomment-1090484293

    ## cache_dir defaults to "~/.cache/huggingface/datasets"
    # cache_dir="~/.cache/huggingface/datasets"  # Disk quota exceeded
    # cache_dir="/tmp/najroldi"  # No space left on device 
    # cache_dir="/fast/najroldi/tmp"  # FileSystem does not appear to support flock
  )

  if streaming:
    # If streaming, we need to convert IterableDataset to Dataset
    print("From IterableDataset to Dataset")
    iterable_ds = raw_dataset

    # Optionally subsample the dataset
    if nrows is not None:
      iterable_ds = iterable_ds.take(nrows)
    
    # From IterableDataset to Dataset
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
  else:
    dataset = raw_dataset

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
  # in a single input sequences to prevent truncation during tokenization.
  tokenizer.model_max_length = 1e30 

  # Tokenize
  tokenized_datasets = dataset.map(
    tokenize_function, 
    remove_columns=['text'],
    **map_setup
  )

  # Reset to correct value
  tokenizer.model_max_length = seq_len

  # print("Saving Tokenized Data")
  # tokenized_datasets.save_to_disk(os.path.join(out_path, f"tokenized_data"))

  # print("Saving Tokenizer")
  # tokenizer.save_pretrained(os.path.join(out_path, "tokenizer"))

  # --------------------------------------------------------------------
  # Concat in chunks of max_seq_len.

  print("Concatenating in chunks of max_seq_len")

  # Concat in chunks
  lm_datasets = tokenized_datasets.map(
    partial(concat_chunck, max_seq_length=max_seq_length),
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
  # One document might be splitted across both train and valid splits.

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
