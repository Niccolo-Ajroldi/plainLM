
import os
import yaml
import math
import shutil
import wandb
import torch

from itertools import product
from collections import namedtuple


def load_config(path, job_idx=None):
  """
  Parse a yaml file and return the correspondent config as a namedtuple.
  If the config files has multiple entries, returns the one corresponding to job_idx.
  """
  
  with open(path, 'r') as file:
    config_dict = yaml.safe_load(file)
  Config = namedtuple('Config', config_dict.keys())

  if job_idx is None:
    cfg = config_dict
    sweep_size = 1

  else:
    keys = list(config_dict.keys())
    values = [val if isinstance(val, list) else [val] for val in config_dict.values()]
    combinations = list(product(*values))

    sweep_size = len(combinations)
    if job_idx >= sweep_size:
      raise ValueError("job_idx exceeds the total number of hyperparam combinations.")

    combination = combinations[job_idx]
    cfg = {keys[i]: combination[i] for i in range(len(keys))}
  
  return Config(**cfg), sweep_size


def init_wandb(cfg):
  """Initalizes a wandb run"""
  os.environ["WANDB__SERVICE_WAIT"] = "600"
  os.environ["WANDB_SILENT"] = "true"

  if getattr(cfg, 'check_existing_wandb_run', False):
    if _matching_wandb_run_exists(cfg):
      raise FileExistsError("A run with the same config exists. Aborting.")

  wandb.init(
    project=cfg.wandb_project, 
    name=cfg.wandb_run_name, 
    dir=cfg.wandb_dir,
    config=cfg._asdict()
  )


def log_job_info(FLAGS):
  """Logs info about cluster job."""
  if FLAGS.job_cluster is not None and FLAGS.job_idx is not None:
    print(f'JOB_CLUSER = {FLAGS.job_cluster}')
    print(f'JOB_INDEX = {FLAGS.job_idx}')
    print(f'JOB_ID = {FLAGS.job_cluster}.{FLAGS.job_idx}')
    wandb.log({'JOB_CLUSTER': FLAGS.job_cluster})
    wandb.log({'JOB_INDEX': FLAGS.job_idx})
    wandb.log({'JOB_ID': f"{FLAGS.job_cluster}.{FLAGS.job_idx}",})


def _matching_wandb_run_exists(cfg):
    """Check for runs on wandb with the same config. Return True if such run exists."""

    api = wandb.Api()
    runs = api.runs(f"ajnico/{cfg.wandb_project}")
    
    # Extract important flags from configs
    to_match_config = {k: getattr(cfg, k) for k in {
      "trainset_path", "vocab_size", "seq_len", "eval", "validset_path", "eval_every_steps",
      "model", "d_model", "expand", "n_layers", "n_heads", "mlp_class", "tie_embeddings",
      "steps_budget", "micro_batch_size", "grad_accumulation_steps", "dtype", 
      "optim", "lr", "weight_decay", "beta1", "beta2", "grad_clip", "eps",
      "scheduler", "warmup_steps", "cooldown_steps", "lr_start", "lr_end", "lr_end_pct",
      "sampler_seed", "seed",
    }}
    print(f"Checking for wand runs with the same config....")
    # print(f"Matching config: {to_match_config}\n\n")

    # Separate non-float and float keys
    non_float_config = {k: v for k, v in to_match_config.items() if not isinstance(v, float)}
    float_config = {k: v for k, v in to_match_config.items() if isinstance(v, float)}

    # Step 1: Fetch and filter runs using non-float keys
    runs = api.runs(
      f"ajnico/{cfg.wandb_project}",
      filters={"$and": [{"config.{}".format(k): v} for k, v in non_float_config.items()]}
    )
    if not runs:  # If no matches are found in the first stage
      return False

    # Step 2: Refine matches using float attributes only
    for run in runs:
      if all(
        math.isclose(v, run.config.get(k), rel_tol=1e-5, abs_tol=1e-5)
        for k, v in float_config.items()
      ):
        print(f"Found matching wandb run with ID: {run.id}")
        return True

    return False


def maybe_make_dir(cfg, job_idx=None):
  """Creates an experiment directory if checkpointing is enabled"""
  if not cfg.save_intermediate_checkpoints and not cfg.save_last_checkpoint:
    return
  if cfg.resume and cfg.resume_exp_name is None:  # if resuming from the same exp
    return

  exp_dir = os.path.join(cfg.out_dir, cfg.exp_name)
  if job_idx is not None:  # subfolder for each job in the sweep
    exp_dir = os.path.join(exp_dir, f"job_idx_{job_idx}")

  if os.path.exists(exp_dir):
    if not cfg.over_write:
      raise ValueError(f"Found existing exp_dir at {exp_dir}.")
    print(f"Removing experiment dir: {exp_dir}")
    shutil.rmtree(exp_dir)

  print(f"Creating experiment directory: {exp_dir}")
  os.makedirs(exp_dir, exist_ok=True)
  with open(os.path.join(exp_dir, 'config.yaml'), 'w') as file:
    yaml.dump(cfg._asdict(), file, default_flow_style=False)


def log(cfg, metrics, micro_step, train_losses, valid_loss, optimizer, world_size):
  "Computes new metrcs and appends them to metrics. Logs on wandb. Prints log."
  # NOTE: train_losses is an array of losses, if DDP, this is from master_process only
  # NOTE: valid_loss is a float, already reduced across GPUs

  if isinstance(train_losses, list):
    train_loss = torch.stack(train_losses).mean().item() # avg loss

  new_metrics = {
    "micro_step": micro_step,
    "step": int(micro_step / cfg.grad_accumulation_steps),
    "tokens": micro_step * cfg.micro_batch_size * cfg.seq_len * world_size,
    "lr": optimizer.param_groups[0].get("lr", float("NaN")),
    "train/loss": train_loss,
    "train/ppl": math.exp(train_loss),
  }
  if valid_loss is not None:
    new_metrics["valid/loss"] = valid_loss
    new_metrics["valid/ppl"] = math.exp(valid_loss)

  for k,v in new_metrics.items():
    metrics[k].append(v)

  if cfg.print_progress:
    msg = ' | '.join(
      f"{key}: {value:.3e}" if isinstance(value, float) else f"{key}: {value}"
      for key, value in new_metrics.items()
    )
    print(msg)
  
  if cfg.use_wandb:
    wandb.log(new_metrics)


def print_master(msg):
  """Prints only in master process if using multiple GPUs."""
  rank = os.environ.get('RANK', -1)
  ddp = int(rank) != -1
  master_process = (not ddp) or (int(rank) == 0)
  
  if master_process:
    print(msg)
