"""Pretrain a Transformer on language modeling."""

from absl import app, flags
from collections import defaultdict
import torch

import utils
from utils import print_master
from torch_utils import pytorch_setup, destroy_ddp
from data import get_dataloaders
from checkpoint_utils import save_checkpoint, get_checkpoint_path
from engine.checkpointer import get_full_model_state_dict, get_full_optimizer_state_dict
from models import construct_model
from engine import TorchEngine

flags.DEFINE_string('config', 'config/config.yaml', 'Path to config.yaml file.')
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
flags.DEFINE_integer('job_cluster', None, 'Job cluster ID.')
FLAGS = flags.FLAGS


def main(_):

  CFG_PATH, JOB_IDX = FLAGS.config, FLAGS.job_idx
  cfg, _ = utils.load_config(CFG_PATH, JOB_IDX)

  local_rank, world_size, device, master_process = pytorch_setup(cfg)

  if master_process:
    utils.maybe_make_dir(cfg, JOB_IDX)

  if cfg.use_wandb and master_process:
    utils.init_wandb(cfg)
    utils.log_job_info(FLAGS)

  # Checkpoint path
  ckpt_path = get_checkpoint_path(cfg)

  # Dataset
  trainloader, validloader = get_dataloaders(cfg)

  # Model TODO: move inside engine?
  with torch.device("meta"):
    model, _ = construct_model(cfg)

  # Engine
  engine = TorchEngine(model, cfg, device, ckpt_path)

  # If we are just cooling down, we set budget = resume + cooldown
  steps_budget = cfg.steps_budget if cfg.scheduler != "linear_cooldown" else cfg.resume_step + engine.scheduler.cooldown_steps
  micro_step_budget = steps_budget * cfg.grad_accumulation_steps
  if micro_step_budget > len(trainloader):
    raise ValueError("trainloader too short!")

  # Start the dataloader from the correct micro-batch
  step_start = engine._step if cfg.resume is not None else 0 # bleah!
  micro_step_start = step_start * cfg.grad_accumulation_steps
  print_master(f"=== Start Training from step: {step_start}/{steps_budget}, micro_step: {micro_step_start}/{micro_step_budget} ===")

  # Bookkeeping
  metrics = defaultdict(list)
  train_loss_array = []

  # Training
  for micro_step, micro_batch in enumerate(trainloader, micro_step_start+1):
    step = micro_step // cfg.grad_accumulation_steps
    is_step = micro_step % cfg.grad_accumulation_steps == 0
    if step > steps_budget and is_step:
      break

    # Train
    train_loss = engine.step(micro_batch)
    train_loss_array.append(train_loss)

    # Eval
    valid_loss = None
    if cfg.eval and step % cfg.eval_every_steps == 0 and is_step:
      print_master("Evaluating on validation set")
      valid_loss = engine.eval(validloader)

    # Log
    if master_process and step % cfg.log_every_steps == 0 and is_step:
      utils.log(cfg, metrics, micro_step, train_loss, train_loss_array, valid_loss, engine.optimizer, world_size)
      train_loss_array = []

    # Checkpoint TODO: replace w/ engine.save() + json dump
    if cfg.save_intermediate_checkpoints and step % cfg.save_every_steps == 0 and is_step:
      save(step, engine, cfg, metrics, JOB_IDX, master_process)

  # End of training: log and save checkpoint
  print_master(f"=== Training Completed! ===")
  if cfg.save_last_checkpoint:
    save(step, engine, cfg, metrics, JOB_IDX, master_process)

  # DDP slaughtering
  destroy_ddp()


def save(step, engine, cfg, metrics, JOB_IDX, master_process):
  model_state_dict = get_full_model_state_dict(engine.model, cfg.fsdp2)
  optim_state_dict = get_full_optimizer_state_dict(engine.optimizer, cfg.fsdp2)
  if master_process:
    save_checkpoint(step, model_state_dict, optim_state_dict, engine, cfg, metrics, JOB_IDX)


if __name__ == "__main__":
  app.run(main)
