"""Evaluate all checkpoints in a directory."""

from absl import app, flags
from collections import defaultdict

import os
import re
import math
import torch
import wandb

from datasets import Dataset, load_from_disk
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, SequentialSampler

import utils
from utils import print_master
from torch_utils import pytorch_setup, destroy_ddp
from models import construct_model
from engine import TorchEngine

flags.DEFINE_string('config', 'config/config.yaml', 'Path to config.yaml file.')
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
flags.DEFINE_integer('job_cluster', None, 'Job cluster ID.')
FLAGS = flags.FLAGS


def main(_):

  # Load config from file
  cfg, _ = utils.load_config(FLAGS.config, job_idx=None)

  local_rank, world_size, device, master_process = pytorch_setup(cfg)

  if cfg.use_wandb and master_process:
    utils.init_wandb(cfg)
    utils.log_job_info(FLAGS)

  # Validation dataloader
  valid_set = load_from_disk(cfg.validset_path)
  if not isinstance(valid_set, Dataset):
    raise ValueError("'dataset' should be a datasets.Dataset")
  validloader = DataLoader(
    valid_set,
    batch_size = cfg.micro_batch_size,
    num_workers = cfg.num_workers,
    sampler = DistributedSampler(valid_set, drop_last=True) \
        if dist.is_initialized() else SequentialSampler(valid_set)
  )

  # Model
  model, model_cfg = construct_model(cfg)
  if master_process:
    print(model_cfg)
    print(model)

  # Engine
  engine = TorchEngine(model, cfg, device, local_rank, ckpt=None)

  # List checkpoints in the folder
  ckpt_folder = cfg.eval_ckpt_folder
  if not os.path.isdir(ckpt_folder):
    raise ValueError(f"Invalid checkpoint folder: {ckpt_folder}")
  prefix = 'ckpt_micro_step_'
  checkpoints = [f for f in os.listdir(ckpt_folder) if re.match(rf"^{prefix}\d+\.pth$", f)]
  checkpoints.sort(key=lambda x: int(x[len(prefix):-4]))  # Sort numerically
  print_master(f"Found {len(checkpoints)} checkpoints in {ckpt_folder}")

  # Loop through checkpoints
  print_master(f"=== Eval Loop Started! ===")
  for ckpt_file in checkpoints:
    print_master(f"Checkpoint: {ckpt_file}")

    # Read checkpoint from disk
    ckpt_path = os.path.join(ckpt_folder, ckpt_file)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    micro_step = ckpt['micro_step']
    step = micro_step // cfg.grad_accumulation_steps
  
    # Load checkpoint
    state_dict = {"_orig_mod.module." + k: v for k, v in ckpt['state_dict'].items()}
    engine.model.load_state_dict(state_dict)
    engine.optimizer.load_state_dict(ckpt['optimizer'])
    engine.scheduler.load_state_dict(ckpt['scheduler'])
    if 'scaler' in ckpt:
      engine.scaler.load_state_dict(ckpt['scaler'])

    # Eval
    valid_loss = engine.eval(validloader)
    valid_ppl = math.exp(valid_loss)

    # Log
    if master_process:
      print_master(f'micro_step: {micro_step} | step: {step} | valid/loss: {valid_loss:.3e} | valid/ppl: {valid_ppl:.3e}')
      if cfg.use_wandb:
        wandb.log({'micro_step': micro_step, 'step': step, 'valid/loss': valid_loss, 'valid/ppl': valid_ppl})

  # End of evals
  print_master(f"=== Eval Loop Completed! ===")

  # DDP slaughtering
  destroy_ddp()


if __name__ == "__main__":
  app.run(main)
