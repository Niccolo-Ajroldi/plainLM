"""Pretrain a Transformer on language modeling."""

from collections import defaultdict

from absl import app, flags

import utils
from checkpoint_utils import save_checkpoint
from data import get_dataloaders
from engine import TorchEngine
from models import construct_model
from torch_utils import destroy_ddp, pytorch_setup
from utils import print_master

flags.DEFINE_string("config", "config/config.yaml", "Path to config.yaml file.")
flags.DEFINE_integer("job_idx", None, "Job idx for job-array sweeps. From 0 to n-1.")
flags.DEFINE_integer("job_cluster", None, "Job cluster ID.")
FLAGS = flags.FLAGS


def main(_):
  CFG_PATH, JOB_IDX = FLAGS.config, FLAGS.job_idx
  cfg, _ = utils.load_config(CFG_PATH, JOB_IDX)

  rank, world_size, device, master_process = pytorch_setup(cfg)

  if master_process:
    utils.maybe_make_dir(cfg, JOB_IDX)

  if cfg.use_wandb and master_process:
    utils.init_wandb(cfg)
    utils.log_job_info(FLAGS)

  # Dataset
  trainloader, validloader = get_dataloaders(cfg)

  # Model
  model, _ = construct_model(cfg)

  # Engine
  engine = TorchEngine(model, cfg, device)

  # If we are just cooling down, we set budget = resume + cooldown
  steps_budget = cfg.steps_budget if cfg.scheduler != "linear_cooldown" else cfg.resume_step + engine.scheduler.cooldown_steps
  micro_step_budget = steps_budget * cfg.grad_accumulation_steps
  if micro_step_budget > len(trainloader):
    raise ValueError("trainloader too short!")

  # Start the dataloader from the correct micro-batch
  step_start = cfg.resume_step if cfg.resume else 0
  micro_step_start = step_start * cfg.grad_accumulation_steps
  print_master(
    f"=== Start Training from step: {step_start}/{steps_budget}, micro_step: {micro_step_start}/{micro_step_budget} ===",
  )

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
    
    # Eval
    valid_loss = None
    if cfg.eval and step % cfg.eval_every_steps == 0 and is_step:
      print_master("Evaluating on validation set")
      valid_loss = engine.eval(validloader)
    metrics["valid/loss"].append(valid_loss)

    # Log
    if master_process and step % cfg.log_every_steps == 0 and is_step:
      metrics["step"].append(step)
      metrics["micro_step"].append(micro_step)
      metrics["tokens"].append(step * cfg.seq_len * cfg.micro_batch_size * world_size)
      metrics["train/loss"].append(train_loss.item())
      for n, optim in engine.optimizers.items():
        metrics[f"{n}_lr"].append(optim.param_groups[0]["lr"])
      utils.log(cfg, metrics)

    # Checkpoint
    if (
      cfg.save_intermediate_checkpoints
      and step % cfg.save_every_steps == 0
      and is_step
    ):
      save_checkpoint(step, model, engine, cfg, metrics, rank, JOB_IDX)

  # End of training: log and save checkpoint
  print_master("=== Training Completed! ===")
  if cfg.save_last_checkpoint:
    save_checkpoint(step, model, engine, cfg, metrics, rank, JOB_IDX)

  # DDP slaughtering
  destroy_ddp()


if __name__ == "__main__":
  app.run(main)
