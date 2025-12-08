import json
import os
import re

import torch

import utils


def _latest_ckpt_dir(ckpt_dir: str, prefix: str = "ckpt_step_") -> str | None:
    """Return latest directory matching f"{prefix}{step}/"."""
    if not os.path.isdir(ckpt_dir):
        return None

    # pattern:  ^<prefix>(digits)$
    pat = rf"^{re.escape(prefix)}(\d+)$"

    dirs = [
        d for d in os.listdir(ckpt_dir)
        if os.path.isdir(os.path.join(ckpt_dir, d)) and re.match(pat, d)
    ]
    if not dirs:
        return None

    dirs.sort(key=lambda d: int(re.match(pat, d).group(1)))
    return os.path.join(ckpt_dir, dirs[-1])


def save_checkpoint(step, model, engine, cfg, metrics, rank, job_idx=None):
  
  exp_dir = utils.get_exp_dir_path(cfg, job_idx)
  save_dir = os.path.join(exp_dir, f"ckpt_step_{step}")
  os.makedirs(save_dir, exist_ok=True)
  print(f"Saving checkpoint to {save_dir}")

  if rank == 0:
    # step index
    with open(os.path.join(save_dir, "step.txt"), "w") as f:
        f.write(str(step))

    # metrics
    metrics_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_path, "w") as f:
      json.dump(dict(metrics), f)

    # model
    torch.save(
      model.state_dict(), 
      os.path.join(save_dir, f"model_state.pth")
    )

    # schedulers
    for n, scheduler in engine.schedulers.items():
      torch.save(
        scheduler.state_dict() if scheduler else {},
        os.path.join(save_dir, f'scheduler_{n}_state.pth')
      )
  
  # optimizers
  for n, optimizer in engine.optimizers.items():
    if n == 'zero1adamw':   # sharded, each rank saves
      torch.save(
        optimizer.state_dict(), 
        os.path.join(save_dir, f'optimizer_{n}_rank{rank}_state.pth')
      )
    elif rank == 0:         # replicated, only rank 0 saves
      torch.save(
        optimizer.state_dict(), 
        os.path.join(save_dir, f'optimizer_{n}_state.pth')
      )


def load_checkpoint(cfg):
  """
    returns:
      {
      "step": 1200,
      "model_state": <state_dict>,
      "scheduler_adamw_state": <state_dict>,
      "optimizer_zero1adamw_state": <state_dict>,
      "optimizer_zero1adamw_rank0_state": <state_dict>,
      "optimizer_zero1adamw_rank1_state": <state_dict>,
      ...
  """
  if not cfg.resume:
      raise ValueError("No checkpoint to resume from.")

  # resume from a specified exp or from the same exp
  # notice that we can resume from `resume_exp_name`, but save to a different `exp_name`
  resume_exp = cfg.resume_exp_name or cfg.exp_name
  base = os.path.join(cfg.out_dir, resume_exp)

  if cfg.resume_step is not None:
      ckpt_dir = os.path.join(base, f"ckpt_step_{cfg.resume_step}")
  else:
      ckpt_dir = _latest_ckpt_dir(base, prefix="ckpt_step_")

  if ckpt_dir is None or not os.path.isdir(ckpt_dir):
      raise ValueError(f"No checkpoint directory found in: {base}")

  print(f"Resuming from {ckpt_dir}")
  out = {}

  # step
  with open(os.path.join(ckpt_dir, "step.txt"), "r") as f:
      out['step'] = int(f.read().strip())

  # load model, optimizers, schedulers
  for fn in os.listdir(ckpt_dir):
      if not fn.endswith(".pth"):
          continue

      path = os.path.join(ckpt_dir, fn)

      key = fn[:-4] # drop .pth 
      out[key] = torch.load(path, map_location="cpu")

  return out
