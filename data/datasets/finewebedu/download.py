"""
Download 100BT from FineWebEdu.

- cache_dir=None: OSError: [Errno 122] Disk quota exceeded
- cache_dir='/home/najroldi/tmp': OSError: [Errno 122] Disk quota exceeded
- cache_dir='/fast/najroldi/tmp': NotImplementedError: FileSystem does not appear to support flock; use SoftFileLock instead
- cache_dir='/tmp/najroldi': OSError: [Errno 28] No space left on device
  - `import filelock; filelock.FileLock = filelock.SoftFileLock`: stuck forever?
  - patch: https://github.com/huggingface/datasets/issues/6744
"""

import os

import filelock
filelock.FileLock = filelock.SoftFileLock

from datasets import load_dataset

data_dir = "/fast/najroldi/data/lm"
data_dir = os.path.join(data_dir, 'finewebedu-100BT')

cache_dir = '/fast/najroldi/tmp'
cache_dir = os.path.join(cache_dir,'lm') if cache_dir is not None \
    else os.path.expanduser('~/.cache/huggingface/datasets')

ds = load_dataset(
    'HuggingFaceFW/fineweb-edu',
    name='sample-100BT',
    split='train',
    cache_dir=cache_dir)

ds.save_to_disk(data_dir)
