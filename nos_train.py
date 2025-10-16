"""
Pretrain a Transformer on language modeling.

We refactor the training loop into a function `train` that takes learning rate and
pipeline directory as arguments, so that it can be called from NOS.

TODOs:
- [] no resume logic
- [] args: opt, lr, pipeline_directory
- [] base config path
- [] ruff
"""

from collections import defaultdict

import torch

import utils
from checkpoint_utils import maybe_load_checkpoint, save_checkpoint
from data import get_dataloaders
from engine import TorchEngine
from models import construct_model, get_param_groups
from optim import initialize_scheduler
from torch_utils import destroy_ddp, pytorch_setup
from utils import print_master

# Default config path for a 70M transformer model, TODO: pass it as arg?
CFG_PATH = "config/opt_search/70M_10BT.yaml"


def train(
  optimizer_cls: torch.optim.Optimizer,
  lr: float,
  pipeline_directory: str,
) -> float:
  """
  Train a transformer on Causal Language Modeling.

  Args:
    otpimizer_cls: optimizer class to use for this training run.
    lr: learning rate to use for this training run.
    pipeline_directory: directory to save checkpoints and logs.
  Returns:
    valid_loss: validation loss after training.
  """

  # Load a default config as a namedtuple
  cfg, _ = utils.load_config(CFG_PATH)

  # Replace some arguments with user-specified ones
  cfg = cfg._replace(lr=lr)
  cfg = cfg._replace(out_dir=pipeline_directory)
  cfg = cfg._replace(wandb_dir=pipeline_directory)

  local_rank, world_size, device, master_process = pytorch_setup(cfg)

  if master_process:
    utils.maybe_make_dir(cfg)

  if cfg.use_wandb and master_process:
    utils.init_wandb(cfg)

  # Load checkpoint
  ckpt = maybe_load_checkpoint(cfg)

  # Dataset
  trainloader, validloader = get_dataloaders(cfg)

  # Model
  model, _ = construct_model(cfg)

  # Engine
  engine = TorchEngine(model, cfg, device, local_rank, ckpt)

  # Optimizer is usually defined by engine, we define it here for ease of use with NOS
  param_groups = get_param_groups(model, cfg)
  engine.optimizer = optimizer_cls(
    param_groups, 
    lr=cfg.lr, 
    weight_decay=cfg.weight_decay, 
    betas=(cfg.beta1, cfg.beta2),
    eps=getattr(cfg, "eps", 1e-8)
  )
  engine.scheduler = initialize_scheduler(engine.optimizer, cfg)

  # If we are just cooling down, we set budget = resume + cooldown
  steps_budget = cfg.steps_budget if cfg.scheduler != "linear_cooldown" else cfg.resume_step + engine.scheduler.cooldown_steps
  micro_step_budget = steps_budget * cfg.grad_accumulation_steps
  if micro_step_budget > len(trainloader):
    raise ValueError("trainloader too short!")

  # Start the dataloader from the correct micro-batch
  step_start = cfg.resume_step if cfg.resume else 0
  micro_step_start = step_start * cfg.grad_accumulation_steps
  print_master(f"Start Training from micro_step: {micro_step_start}/{micro_step_budget}")

  # Bookkeeping
  metrics = defaultdict(list)

  # Training
  for micro_step, micro_batch in enumerate(trainloader, micro_step_start + 1):
    step = micro_step // cfg.grad_accumulation_steps
    is_step = micro_step % cfg.grad_accumulation_steps == 0
    if step > steps_budget and is_step:
      break

    # Train
    train_loss = engine.step(micro_batch)

    # Log
    if master_process and step % cfg.log_every_steps == 0 and is_step:
      metrics["step"].append(step)
      metrics["micro_step"].append(micro_step)
      metrics["tokens"].append(step * cfg.seq_len * world_size)
      metrics["lr"].append(engine.optimizer.param_groups[0]["lr"])
      metrics["train/loss"].append(train_loss.item())
      utils.log(cfg, metrics)

    # Checkpoint
    if master_process and cfg.save_intermediate_checkpoints and step % cfg.save_every_steps == 0 and is_step:
      save_checkpoint(step, model, engine, cfg, metrics)

  # End of training: log and save checkpoint
  print_master("Training Completed!")
  if master_process and cfg.save_last_checkpoint:
    save_checkpoint(step, model, engine, cfg, metrics)

  # Eval
  valid_loss = None
  if cfg.eval:
    print_master("Evaluating on validation set")
    valid_loss = engine.eval(validloader)
    if master_process:
      metrics["valid/loss"] = valid_loss
      utils.log(cfg, metrics)

  # DDP slaughtering
  destroy_ddp()

  return valid_loss
