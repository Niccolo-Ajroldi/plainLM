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

import avg
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

  # AvgEngine
  avg_engine = avg.AvgEngine(model, cfg, device, local_rank, ckpt=None)

  # List checkpoints in the folder
  ckpt_folder = cfg.eval_ckpt_folder
  if not os.path.isdir(ckpt_folder):
    raise ValueError(f"Invalid checkpoint folder: {ckpt_folder}")
  prefix = 'ckpt_micro_step_'
  checkpoints = [f for f in os.listdir(ckpt_folder) if re.match(rf"^{prefix}\d+\.pth$", f)]
  checkpoints.sort(key=lambda x: int(x[len(prefix):-4]))  # Sort numerically
  print_master(f"Found {len(checkpoints)} checkpoints in {ckpt_folder}")
  print_master(f"=== Eval Loop Started! ===")

  # Start eval when we have at least one model in the avg buffer
  s = avg_engine.avg_start_step
  e = avg_engine.avg_every_steps
  eval_start_step = ((s // e) + 1) * e

  for step in range(cfg.step_budget):

    micro_step = step * cfg.grad_accumulation_steps
    ckpt_file = f"{prefix}{micro_step}.pth"
    if ckpt_file in checkpoints:
      print_master(f"Found Checkpoint: {ckpt_file}")
      ckpt_path = os.path.join(ckpt_folder, ckpt_file)
      avg_engine.maybe_update_buffer(ckpt_path, step)

    # Eval
    valid_loss = None
    if cfg.eval and step % cfg.eval_every_steps == 0 and step >= eval_start_step:
      print_master("Evaluating on validation set")
      avg_engine.prepare_for_eval()
      valid_loss = avg_engine.eval(validloader)
      valid_ppl = math.exp(valid_loss)
      print_master(f'step: {step} | valid/loss: {valid_loss:.3e} | valid/ppl: {valid_ppl:.3e}')
      if master_process and cfg.use_wandb:
        wandb.log({'step': step, 'valid/loss': valid_loss, 'valid/ppl': valid_ppl})

  # End of evals
  print_master(f"=== Eval Loop Completed! ===")

  # DDP slaughtering
  destroy_ddp()


if __name__ == "__main__":
  app.run(main)
