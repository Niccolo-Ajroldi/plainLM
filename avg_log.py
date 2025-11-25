"""
Loop over checkpoints in a folder, averaging and evaluating.

Notice that:
- there is no training set.
- micro_batch_size is used for validation, not for training.
- grad accumulation does not need to match the one from the training script (not needed!).
- optimizer and scheduler are NOT used and hence not initialized.
"""

from absl import app, flags

import os
import re
import math
import wandb
import torch

from datasets import load_from_disk
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, SequentialSampler

import utils
from utils import print_master
from torch_utils import pytorch_setup, destroy_ddp
from checkpoint_utils import save_checkpoint
from models import construct_model
from avg import AVG_REGISTRY

flags.DEFINE_string('config', 'config/config.yaml', 'Path to config.yaml file.')
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
flags.DEFINE_integer('job_cluster', None, 'Job cluster ID.')
FLAGS = flags.FLAGS


def main(_):

  # Load config from file
  cfg, _ = utils.load_config(FLAGS.config, job_idx=FLAGS.job_idx)

  local_rank, world_size, device, master_process = pytorch_setup(cfg)

  if master_process:
    utils.maybe_make_dir(cfg, FLAGS.job_idx)

  if cfg.use_wandb and master_process:
    utils.init_wandb(cfg)
    utils.log_job_info(FLAGS)

  # Validation dataloader
  valid_set = load_from_disk(cfg.validset_path)
  if cfg.intra_doc_masking and "docs_lengths" in valid_set.column_names:
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
  validloader = DataLoader(
    valid_set,
    batch_size=cfg.micro_batch_size,
    drop_last=True,  # makes eval with DDP easier
    shuffle=False,
    sampler = DistributedSampler(valid_set, drop_last=True) if dist.is_initialized() else SequentialSampler(valid_set),
    num_workers=cfg.num_workers,
    pin_memory=True,
    prefetch_factor=2 if cfg.num_workers > 0 else None,
    persistent_workers=False,
    collate_fn=collate_fn
  )

  # Model
  model, _ = construct_model(cfg)

  # List checkpoints in the folder
  ckpt_folder = cfg.eval_ckpt_folder
  if not os.path.isdir(ckpt_folder):
    raise ValueError(f"Invalid checkpoint folder: {ckpt_folder}")
  prefix = "ckpt_step_"
  checkpoints = [f for f in os.listdir(ckpt_folder) if re.match(rf"^{prefix}\d+\.pth$", f)]
  avail_steps = sorted(int(f[len(prefix):-4]) for f in checkpoints)
  min_step = avail_steps[0]
  print_master(f"Found {len(checkpoints)} checkpoints in {ckpt_folder}")
  print_master(f"Min step: {min_step}")

  # AvgEngine
  avg_engine = AVG_REGISTRY[cfg.avg_scheme](
      model, cfg, device, local_rank,
      avail_steps, ckpt_folder, prefix,
  )

  print_master(f"=== Loop Started! ===")

  for step in range(min_step, cfg.steps_budget+1):  # +1 to allow eval at the end

    # Eval
    valid_loss = None
    if cfg.eval and step % cfg.eval_every_steps == 0:
      
      # Compute average
      print_master(f"Compute avg [step={step}]")
      is_avg = avg_engine.maybe_update_buffer(step)

      # If no average available yet, continue
      if not is_avg:
        continue

      print_master(f"Preparing for evaluation [step={step}]")
      avg_engine.prepare_for_eval()
      print_master("Evaluating on validation set")
      valid_loss = avg_engine.eval(validloader)
      valid_ppl = math.exp(valid_loss)
      print_master(f'step: {step} | valid/loss: {valid_loss:.3e} | valid/ppl: {valid_ppl:.3e}')
      if master_process and cfg.use_wandb:
        wandb.log({'step': step, 'valid/loss': valid_loss, 'valid/ppl': valid_ppl})

      # Save AVG Checkpoint (avg already loaded in model because of `prepare_for_eval` call)
      if master_process and cfg.save_intermediate_checkpoints and step % cfg.save_every_steps == 0:
        exp_dir = utils.get_exp_dir_path(cfg, FLAGS.job_idx)
        save_path = os.path.join(exp_dir, f'ckpt_step_{step}.pth')
        print(f"Saving checkpoint to {save_path}")
        torch.save(avg_engine.model.state_dict(), save_path)

  # End of evals
  print_master(f"=== Eval Loop Completed! ===")
  if master_process and cfg.save_last_checkpoint:
    print_master("Preparing for evaluation")
    avg_engine.prepare_for_eval() # explicitely load avg into model
    exp_dir = utils.get_exp_dir_path(cfg, FLAGS.job_idx)
    save_path = os.path.join(exp_dir, f'ckpt_step_{step}.pth')
    print(f"Saving checkpoint to {save_path}")
    torch.save(avg_engine.model.state_dict(), save_path)

  # DDP slaughtering
  destroy_ddp()


if __name__ == "__main__":
  app.run(main)
