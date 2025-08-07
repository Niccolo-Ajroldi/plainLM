"""
This script can be used to download, tokenize and chunk an HF dataset.

These three ops can be done independently, so you can run:
- `python prepare.py --donwload` to download the raw dataset.
- `python prepare.py --tokenize` to tokenize the raw dataset.
- `python prepare.py --chunk` to chunk the tokenized dataset.
You can also run all three ops at once by passing all three flags:
- `python prepare.py --download --tokenize --chunk`
If you want to run the tokenization step, it is expected that the raw dataset is available on disk or that `--download` flag is passed.
If you want to run the chunking step, it is expected that the tokenized dataset is available on disk or that `--tokenize` flag is passed.

Download can be done in streaming mode, which might be useful for large datasets.
In this case, the dataset is loaded as an IterableDataset, and then converted to a Dataset.
If you want to use streaming, set the `streaming` flag to True:
- `python prepare.py --donwload --streaming`

By default intermediate datasets are not saved to disk.
You can save the raw dataset by passing `--save_raw` flag.
You can save the tokenized dataset by passing `--save_tokenized` flag.
We recommend saving intermediate datasets to avoid re-downloading or re-tokenizing them.

Example: preprocess FineWebEdu 10BT sample:
```
python prepare.py \
  --out_path=/fast/najroldi/data/lm/fwedu/fwedu_sample_10B_tokenizer_GPT2 \
  --download --tokenize --chunk \
  --dataset_path='HuggingFaceFW/fineweb-edu' \
  --dataset_split=train \
  --dataset_name='sample-10BT' \
  --tokenizer='gpt2' \
  --seq_length=2048 \
  --split_train_valid \
  --n_tokens_valid=10000000 \
  --save_raw --save_tokenized --save_tokenizer
```

The output should look like this:
```
{out_path}/
├── raw_dataset/          # Contains the raw dataset
├── tokenized_dataset/    # Contains the tokenized dataset
├── tokenizer/            # Contains the tokenizer
└── ctx_{seq_length}/
    ├── train/            # Contains the training set
    └── valid/            # Contains the validation set
```

"""

import os
import multiprocessing as mp

# # This will override FileLock globally to use SoftFileLock.
# # Uncomment on filesystems do not support FileLock.
# import filelock
# filelock.FileLock = filelock.SoftFileLock
# os.environ["SOFT_FILELOCK"] = "1"

from absl import app, flags
from functools import partial
from timeit import default_timer as timer
from datasets import Dataset, load_from_disk, load_dataset
from transformers import AutoTokenizer

from data.datasets.data_prep_utils import concat_chunck

# Linux’s default “fork” start method inherits open handles (semaphores/CWD) 
# in the pymp-* temp directory, leading to OSError on cleanup.
# Forcing “spawn” via mp.set_start_method("spawn", force=True) 
# prevents handle inheritance so the temp-dir can be removed safely.
mp.set_start_method('spawn', force=True)

flags.DEFINE_string('out_path', '/fast/najroldi/data/lm/fwedu/test', 'Path where to save the dataset.')
flags.DEFINE_string('cache_path', '~/.cache/huggingface/datasets', 'Cache for download.')

flags.DEFINE_boolean('download', False, 'Download the raw dataset.')
flags.DEFINE_boolean('tokenize', False, 'Tokenize the raw dataset.')
flags.DEFINE_boolean('chunk', False, 'Chunk the tokenized dataset.')

flags.DEFINE_string('dataset_path', 'HuggingFaceFW/fineweb-edu', 'Path of the dataset to download.')
flags.DEFINE_string('dataset_split', 'train', 'Split of the dataset to download.')
flags.DEFINE_string('dataset_name', None, 'Defines the name of the dataset configuration.')
flags.DEFINE_list('dataset_columns', None, 'Columns to keep from the dataset.')

flags.DEFINE_boolean('streaming', False, 'Download the dataset in streaming mode. Ignored if `download` is False.')
flags.DEFINE_integer('nrows', None, 'Number of rows to download. Ignored if `download` is False.')

flags.DEFINE_string('tokenizer', 'gpt2', 'A valid tokenizer for transformers.AutoTokenizer.')
flags.DEFINE_integer('seq_length', None, 'Sequence length for chunking the dataset. Ignored if `chunk` is False.')

flags.DEFINE_boolean('split_train_valid', True, 'Split the dataset into train and valid sets after chunking.')
flags.DEFINE_integer('n_tokens_valid', None, 'Number of tokens in the validation set. Ignored if `chunk` is False.')

flags.DEFINE_boolean('save_raw', False, 'Save the raw dataset to disk. Ignored if `download` is False.')
flags.DEFINE_boolean('save_tokenized', False, 'Save the tokenized dataset to disk. Ignored if `tokenize` is False.')
flags.DEFINE_boolean('save_tokenizer', False, 'Save the tokenizer to disk. Ignored if `tokenize` is False.')

FLAGS = flags.FLAGS


def tokenize_batched(examples, tokenizer):
  bos_token = tokenizer.bos_token
  eos_token = tokenizer.eos_token
  add_eos = lambda seq: (bos_token + seq + eos_token) if seq else seq
  add_eos_batched = lambda seqs: [add_eos(seq) for seq in seqs]
  tokenized_output = tokenizer(
      add_eos_batched(examples["text"]),
      add_special_tokens=False,
      return_special_tokens_mask=False,
      return_attention_mask=False
  )
  return tokenized_output


def main(_):
    raw_ds = None
    tokenized_ds = None
    
    out_path = FLAGS.out_path
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    tokenizer_name = FLAGS.tokenizer.replace('/', '_') if FLAGS.tokenizer is not None else None # sanitize name for paths

    map_setup = dict(
        batched=True,
        batch_size=1024,
        num_proc=8
    )
    
    # --------------------------------------------------------------------
    ## Download.
    if FLAGS.download:

        time_start = timer()
        
        raw_ds = load_dataset(
            FLAGS.dataset_path,
            split=FLAGS.dataset_split,
            name=FLAGS.dataset_name,
            streaming=FLAGS.streaming,
            cache_dir=FLAGS.cache_path,
            **({'columns': FLAGS.dataset_columns} if FLAGS.dataset_columns is not None else {}), # NOTE: it works only for Parquet datasets in streaming mode
          )

        if FLAGS.nrows is not None:
          raw_ds = raw_ds.take(FLAGS.nrows)

        if FLAGS.streaming:
            print("Converting IterableDataset to Dataset.")
            def custom_generator(iterable_ds):
                yield from iterable_ds
            raw_ds = Dataset.from_generator(
                partial(custom_generator, raw_ds), 
                features=raw_ds.features,
            )
    
        if FLAGS.save_raw:
            if os.path.exists(os.path.join(out_path, 'raw_dataset')):
                raise FileExistsError("Raw dataset already exists.")
            print("Saving Raw Dataset")
            raw_ds.save_to_disk(os.path.join(out_path, 'raw_dataset'))

        elapsed = timer() - time_start
        print(f"Downloading time: {elapsed // 60} min")

    # --------------------------------------------------------------------
    ## Tokenize.
    if FLAGS.tokenize:

        time_start = timer()

        if raw_ds is None:
            raw_ds = load_from_disk(os.path.join(out_path, 'raw_dataset'),)
        
        # Shuffle so that multiproc has shards of similar size
        raw_ds = raw_ds.shuffle(seed=1996)

        tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer)
        print(f"Length of tokenizer = {len(tokenizer)}")

        # Set an high maximum number of tokens that the tokenizer can handle 
        # in a single input sequences to prevent truncation during tokenization.
        tokenizer.model_max_length = 1e30

        print("Tokenizing.")
        tokenized_ds = raw_ds.map(
            partial(tokenize_batched, tokenizer=tokenizer), 
            remove_columns=['text'],
            **map_setup
        )

        # Reset to correct value (superflous, but for clarity)
        if FLAGS.seq_length is not None:
            tokenizer.model_max_length = FLAGS.seq_length

        if FLAGS.save_tokenized:
            if os.path.exists(os.path.join(out_path, f"tokenized_{tokenizer_name}", 'tokenized_dataset')):
                raise FileExistsError("Tokenized dataset already exists.")
            print("Saving Tokenized Dataset")
            tokenized_ds.save_to_disk(os.path.join(out_path, f"tokenized_{tokenizer_name}", 'tokenized_dataset'))
        
        if FLAGS.save_tokenizer:
            if os.path.exists(os.path.join(out_path, f"tokenized_{tokenizer_name}", 'tokenizer')):
                raise FileExistsError("Tokenizer already exists.")
            print("Saving Tokenizer")
            tokenizer.save_pretrained(os.path.join(out_path, f"tokenized_{tokenizer_name}", 'tokenizer'))

        elapsed = timer() - time_start
        print(f"Tokenization time: {elapsed // 60} min")

    # --------------------------------------------------------------------
    ## Chunk.
    if FLAGS.chunk:

        time_start = timer()

        if tokenized_ds is None:
            tokenized_ds = load_from_disk(os.path.join(out_path, f"tokenized_{tokenizer_name}", 'tokenized_dataset'))

        tokenized_ds = tokenized_ds.remove_columns(
            [c for c in tokenized_ds.column_names if c != "input_ids"]
        )

        # NOTE: expected token loss by batched concat_chunk, 
        # it truncates leftover tokens that don't fill a full max_seq_length chunk.
        print("Concatenating and chunking in s of max_seq_length")
        max_seq_length = FLAGS.seq_length + 1
        chunked_ds = tokenized_ds.map(
            partial(concat_chunck, max_seq_length=max_seq_length),
            **map_setup
        )
        print(f"Number of tokens in chunked_ds: {len(chunked_ds) * max_seq_length:_}")
    
        # Cast to tensors
        chunked_ds.set_format("torch")

        elapsed = timer() - time_start
        print(f"Chunkization time: {elapsed // 60} min")
        
    # --------------------------------------------------------------------
    ## Split.

    if not FLAGS.split_train_valid:
        chunked_ds.save_to_disk(os.path.join(out_path, f"ctx_{FLAGS.seq_length}", FLAGS.dataset_split))
    
    else:
        # NOTE: potential single-document contamination in train-valid split (similar to modded-gpt).
        # One document might be split across both train and valid splits.
        # We do not shuffle befre chunking, to avoid mulutple documents contamination.

        n_chunks_valid = FLAGS.n_tokens_valid // max_seq_length
        valid_ds = chunked_ds.select(range(n_chunks_valid))
        train_ds = chunked_ds.select(range(n_chunks_valid, len(chunked_ds)))

        print(f"Number of tokens in train_ds: {len(train_ds) * max_seq_length:_}")
        print(f"Number of tokens in valid_ds: {len(valid_ds) * max_seq_length:_}")

        train_ds = train_ds.shuffle(seed=96)
        valid_ds = valid_ds.shuffle(seed=96)

        train_ds_path = os.path.join(out_path, f"tokenized_{tokenizer_name}", f"ctx_{FLAGS.seq_length}", 'train')
        valid_ds_path = os.path.join(out_path, f"tokenized_{tokenizer_name}", f"ctx_{FLAGS.seq_length}", 'valid')

        if os.path.exists(train_ds_path):
            raise FileExistsError("Trainset already exists.")
        if os.path.exists(valid_ds_path):
            raise FileExistsError("Validset already exists.")

        print("Saving trainset")
        train_ds.save_to_disk(train_ds_path)
        print("Saving validset")
        valid_ds.save_to_disk(valid_ds_path)

    print("Succesfull completion.")

if __name__ == "__main__":
    app.run(main)

