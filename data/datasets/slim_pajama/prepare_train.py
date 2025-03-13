"""
  This script downloads Slim Pajama, tokenize it, and groups it in blocks of length (seq_len+1).
  The tokenizer and the chunked dataset are saved.
  
  slim_pajama, max_seq_length=2048+1
    1K rows => ~ 1M tokens
    10M rows => ~10B tokens
    30M rows => ~30B tokens

  Insipred by:
  https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
  https://github.com/JonasGeiping/cramming/blob/main/cramming/data/pretraining_preparation.py
  
  On the role of EOS:
  https://discuss.huggingface.co/t/how-does-gpt-decide-to-stop-generating-sentences-without-eos-token/41623/2
"""

import os

from itertools import chain
from functools import partial

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# --------------------------------------------------------------------
# Config

# Your path where to save dataset
out_path = "/fast/najroldi/data/lm/slim_pajama/new_sp_15B_tokens"

# HF dataset name
dataset_name = "cerebras/SlimPajama-627B"
split = 'train'

# nrows = 3_000_000 # ~ 3B tokens
nrows = 15_000_000 # ~ 15B tokens
# nrows = 10_000_000 # ~ 10B tokens
# nrows = 1_000 # ~ 1M tokens

seq_len = 2048
max_seq_length = seq_len+1
shuffle_raw_data = True
ordering = "randomized"

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
  split = split,
  streaming = True
)

print("From IterableDataset to Dataset")

# From IterableDataset to Dataset
iterable_ds = raw_dataset.take(nrows)
def gen_from_iterable_dataset(iterable_ds):
  yield from iterable_ds
partial_obj = partial(gen_from_iterable_dataset, iterable_ds)
dataset = Dataset.from_generator(partial_obj, features=iterable_ds.features)

# Shuffle so that multiproc has shards of similar size
if shuffle_raw_data:
  dataset = dataset.shuffle(seed=1996)

# --------------------------------------------------------------------
# Tokenize

print("Tokenize")

# NOTE: 
#   Warming: Special tokens have been added in the vocabulary, 
#   make sure the associated word embeddings are fine-tuned or trained.
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
def tokenize_function(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
  add_eos = lambda seq: (seq + tokenizer.eos_token) if seq else seq
  add_eos_batched = lambda seqs: [add_eos(seq) for seq in seqs]
  return tokenizer(
        add_eos_batched(examples["text"]),  # examples is a dictionary of lists (not a list of dictionaries!)
        return_special_tokens_mask=False, 
        return_attention_mask=False
  )

tokenizer.model_max_length = 1e30 # maximum number of tokens that the model can handle in a single input sequence
tokenized_datasets = dataset.map(
  tokenize_function, 
  remove_columns=['text', 'meta'],
  **map_setup
)
tokenizer.model_max_length = seq_len

## Save tokenized_datasets 
# NOTE: uncomment at your own risk, it's time consuming
tokenized_datasets.save_to_disk(os.path.join(out_path, f"{split}_tokenized"))

print("Saving Tokenizer")

# Save tokenizer
if split == 'train':
  tokenizer.save_pretrained(os.path.join(out_path, "tokenizer"))

# --------------------------------------------------------------------
# Concat in chunks of max_seq_len

print("Concatenating in chunks of max_seq_len")

# Main data processing function that will concatenate all texts 
# from tokenized dataset and generate chunks of max_seq_length.
# NOTE: the current approach leads to data loss at batch boundaries,
#       however concatenation is not possible with this function if batched=False, 
#       sinve concat_chunck relies on processing multiple examples at once to concatenate them.
def group_texts(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
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

# Shuffle
if ordering == "randomized":
  lm_datasets = lm_datasets.shuffle(seed=96)

# Cast to tensors
lm_datasets.set_format("torch")

# Save
lm_datasets.save_to_disk(os.path.join(out_path, split))

print("Shuffled and saved!")
