
import torch

from datasets import Dataset, load_from_disk
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from data.datasamplers import StatefulSequentialSampler, StatefulRandomSampler, StatefulDistributedSampler


def get_dataloaders(cfg):
  """Load trainset and perhaps validset. Returns correspondent DataLoaders."""
  
  train_set = load_from_disk(cfg.trainset_path)
  if not isinstance(train_set , Dataset):
    raise ValueError("dataset should be a datasets.Dataset")

  train_sampler = _get_sampler(train_set, cfg)

  if getattr(cfg, 'intra_doc_masking', False) and "docs_lengths" in train_set.column_names:
    def collate_fn(batch):
      return {
        "input_ids": torch.stack([x["input_ids"] for x in batch], dim=0),
        "docs_lengths": [x["docs_lengths"].tolist() for x in batch]
      }
  else:
    def collate_fn(batch):
      return {
        "input_ids": torch.stack([x["input_ids"] for x in batch], dim=0)
      }

  trainloader = DataLoader(
    train_set,
    sampler=train_sampler,
    batch_size=cfg.micro_batch_size,
    num_workers=cfg.num_workers,
    pin_memory=True,
    prefetch_factor=2 if cfg.num_workers > 0 else None,
    persistent_workers=True if cfg.num_workers > 0 else False,
    collate_fn=collate_fn
  )

  if not cfg.validset_path:
    validloader = None
  else:
    valid_set = load_from_disk(cfg.validset_path)
    if not isinstance(valid_set, Dataset):
      raise ValueError("'dataset' should be a datasets.Dataset")

    if getattr(cfg, 'valid_tokens', False):  # subsample validatiion set
      valid_rows = cfg.valid_tokens // (cfg.seq_len + 1)
      valid_set = valid_set.take(valid_rows)

    if dist.is_initialized():
      valid_sampler = DistributedSampler(valid_set, drop_last=True)
    else:
      valid_sampler = SequentialSampler(valid_set)

    validloader = DataLoader(
      valid_set,
      batch_size=cfg.micro_batch_size,
      drop_last=True,  # makes eval with DDP easier
      shuffle=False,
      sampler=valid_sampler,
      num_workers=cfg.num_workers,
      pin_memory=True,
      prefetch_factor=2 if cfg.num_workers > 0 else None,
      persistent_workers=False,
      collate_fn=collate_fn
    )
  
  return trainloader, validloader


def _get_sampler(train_set, cfg):
  """Initlaizes a sampler for a torch.Dataloader.
  Options:
    - random sampler
    - sequential sampler
    - stateful random sampler
    - stateful sequential sampler
  We implement "stateful" sequential samplers for resuming training from a specified step.
  """
  ddp = dist.is_initialized()

  if cfg.sampler == "random":
    if ddp:
      sampler = DistributedSampler(train_set, shuffle=True, seed=cfg.sampler_seed, drop_last=True)
    else:
      sampler = RandomSampler(train_set, generator=torch.Generator().manual_seed(cfg.sampler_seed) if cfg.sampler_seed else None)

  elif cfg.sampler == "sequential":
    if ddp:
      sampler = DistributedSampler(train_set, shuffle=False, drop_last=True)
    else:
      sampler = SequentialSampler(train_set)

  elif cfg.sampler == "stateful_random":
    micro_step_start = cfg.resume_step * cfg.grad_accumulation_steps if cfg.resume else 0
    if ddp:
      # TODO: allow support for drop_last=True!
      sampler = StatefulDistributedSampler(
        train_set, batch_size=cfg.micro_batch_size, seed=cfg.sampler_seed, start_iter=micro_step_start
      )
    else:
      sampler = StatefulRandomSampler(
        train_set, batch_size=cfg.micro_batch_size, shuffle=True, seed=cfg.sampler_seed, start_idx=micro_step_start
      )

  elif cfg.sampler == "stateful_sequential":
    micro_step_start = cfg.resume_step * cfg.grad_accumulation_steps if cfg.resume else 0
    if ddp:
      raise NotImplementedError("StatefulDistributedSampler currently needs a seed.")
    else:
      sampler = StatefulSequentialSampler(train_set, batch_size=cfg.micro_batch_size, start_idx=micro_step_start)

  else:
    raise NotImplementedError(f"Sampler {cfg.sampler} is not implemented.")

  return sampler
