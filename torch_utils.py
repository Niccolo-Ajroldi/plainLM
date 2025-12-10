import os
import random

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group

from utils import print_master


def pytorch_setup(cfg):
  """Init distributed run, set """
  ddp = int(os.environ.get("RANK", -1)) != -1  # check if DDP is enabled

  if ddp:
    init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    master_process = rank == 0
    seed_offset = rank
  else:
    rank = 0
    local_rank = 0
    world_size = 1
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    master_process = True
    seed_offset = 0

  random.seed(cfg.seed + seed_offset)
  np.random.seed(cfg.seed + seed_offset)
  torch.manual_seed(cfg.seed + seed_offset)

  if getattr(cfg, "allow_tf32", False):
    torch.backends.fp32_precision = "tf32"
  else:
    torch.backends.fp32_precision = "ieee"

  # limit CUDA memory
  if hasattr(cfg, "set_memory_fraction"):
    tot_mem_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    red_mem_gb = tot_mem_gb * cfg.set_memory_fraction
    print_master(f"Limit GPU memory from {tot_mem_gb:.2f}GB to: {red_mem_gb:.2f}GB")
    torch.cuda.set_per_process_memory_fraction(cfg.set_memory_fraction, device=device)

  # deterministic run
  if getattr(cfg, "deterministic", False):
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False

  return rank, world_size, device, master_process


def destroy_ddp():
  if torch.distributed.is_initialized():
    torch.cuda.synchronize()  # finish GPU work
    torch.distributed.barrier()  # wait for all ranks
    destroy_process_group()  # cleanly tear down comms
