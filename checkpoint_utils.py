
import os
import re
import json
import torch
import utils


def _latest_checkpoint(ckpt_dir: str, prefix: str = 'checkpoint_') -> str | None:
  """Retrieve the latest checkpoint path in a directory."""
  if not os.path.isdir(ckpt_dir):
    return None

  # List all files matching the prefix pattern
  checkpoints = [f for f in os.listdir(ckpt_dir) if re.match(rf"^{prefix}\d+\.pth$", f)]
  checkpoints.sort(key=lambda x: int(x[len(prefix):-4])) # Sort numerically

  return os.path.join(ckpt_dir, checkpoints[-1]) if checkpoints else None


def save_checkpoint(step, model, engine, cfg, metrics, job_idx=None):

  optimizer = engine.optimizer
  scheduler = engine.scheduler
  scaler = engine.scaler

  save_optim = getattr(cfg, 'save_optim', True)
  save_scheduler = getattr(cfg, 'save_scheduler', True)
  save_scaler = getattr(cfg, 'save_scaler', True)

  state = {
    "step": step,
    "state_dict": model.state_dict(),
    "optimizer": optimizer.state_dict() if save_optim else None,
    "scheduler": scheduler.state_dict() if scheduler and save_scheduler else {},
    "scaler": scaler.state_dict() if save_scaler else None,
  }

  exp_dir = utils.get_exp_dir_path(cfg, job_idx)

  # Save ckpt
  save_path = os.path.join(exp_dir, f'ckpt_step_{step}.pth')
  print(f"Saving checkpoint to {save_path}")
  torch.save(state, save_path)

  # Save metrics
  metrics_path = os.path.join(exp_dir, f'metrics.json')
  with open(metrics_path, 'w') as f:
    json.dump(dict(metrics), f)


def maybe_load_checkpoint(cfg, device):
  
  ckpt = None
  
  if cfg.resume:
    
    # resume from a specified exp or from the same exp
    # notice that we can resume from `resume_exp_name`, but save to a different `exp_name`
    resume_exp_name = cfg.resume_exp_name if cfg.resume_exp_name is not None else cfg.exp_name
    ckpt_dir = os.path.join(cfg.out_dir, resume_exp_name) # if `resume_exp_name` is already an absolute path, it will just return `resume_exp_name`
    
    # resume from a specified checkpoint or from the latest
    if cfg.resume_step is not None:
      ckpt_path = os.path.join(ckpt_dir, f'ckpt_step_{cfg.resume_step}.pth')
    else:
      ckpt_path = _latest_checkpoint(ckpt_dir, prefix='ckpt_')
    
    # load checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device)

  return ckpt


def match_state_dict_keys(state_dict: dict, state_dict_orig: dict) -> dict:
  """Modifies the keys of 'state_dict' to match the keys of 'state_dict_orig'.

  Takes care of stat_dict discrepancies caused by DDP or torch.compile,
  drop any prefixes from the state_dict, then add the correct prefixes in the correct order.

  Args:
      state_dict (dict): dict to modify
      state_dict_orig (dict): dict to match
  """

  state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
  state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
  
  orig_key = next(iter(state_dict_orig.keys()))  # first key of orig state_dict
  
  if orig_key.startswith('_orig_mod.module.'):
    state_dict = {"_orig_mod.module." + k: v for k, v in state_dict.items()}
  elif orig_key.startswith('_orig_mod.'):
    state_dict = {"_orig_mod." + k: v for k, v in state_dict.items()}
  elif orig_key.startswith('module._orig_mod.'):
    state_dict = {"module._orig_mod." + k: v for k, v in state_dict.items()}
  elif orig_key.startswith('module.'):
    state_dict = {"module." + k: v for k, v in state_dict.items()}

  return state_dict
  
  
