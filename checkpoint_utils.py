
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

def save_checkpoint(step, model, engine, cfg, metrics):
  optimizer = engine.optimizer
  scheduler = engine.scheduler
  scaler = engine.scaler

  save_optim_every_steps = getattr(cfg, 'save_optim_every_steps', cfg.save_every_steps)
  save_step_update_every_steps = getattr(cfg, 'save_step_update_every_steps', cfg.save_every_steps) # best if flag for each

  is_time_to_save_optim = (step % save_optim_every_steps == 0)
  save_optim = is_time_to_save_optim
  save_scheduler = is_time_to_save_optim
  save_scaler = is_time_to_save_optim
  save_step_update = (step % save_step_update_every_steps == 0)
  
  state = {
    "step": step,
    "state_dict": model.state_dict(),
    "optimizer": optimizer.state_dict() if save_optim else None,
    "scheduler": scheduler.state_dict() if scheduler and save_scheduler else {},
    "scaler": scaler.state_dict() if save_scaler else None,
    "step_update": optimizer.state.get('step_update', None) if save_step_update else None,
  }

  exp_dir = utils.get_exp_dir_path(cfg)

  # Save ckpt
  save_path = os.path.join(exp_dir, f'ckpt_step_{step}.pth')
  print(f"Saving checkpoint to {save_path}")
  torch.save(state, save_path)

  # Save metrics
  metrics_path = os.path.join(exp_dir, f'metrics.json')
  with open(metrics_path, 'w') as f:
    json.dump(dict(metrics), f)


def maybe_load_checkpoint(cfg, device):
  """Each job_idx will restore where it left of."""
  ckpt = None
  
  if cfg.resume:
    # Paths are saved with utils.get_exp_dir_path(cfg), with global flags we can call that again from here
    # but this will not control the job_idx name.
    # If cfg.resume_exp_name is given, we will resume from that experiment name
    # else, this will resume the exact sweep with only one config line!
    # commented out because maybe it is not straightforward logic with current config design
    # if cfg.resume_exp_name:
    #   ckpt_dir = os.path.join(cfg.out_dir, cfg.resume_exp_name)
    # else: # verbatim as it was saved
    #   ckpt_dir = utils.get_exp_dir_path(cfg)
      
    # notice that we can resume from `resume_exp_name`, but save to a different `exp_name`
    resume_exp_name = cfg.resume_exp_name if cfg.resume_exp_name is not None else cfg.exp_name
    ckpt_dir = os.path.join(cfg.out_dir, resume_exp_name) # if `resume_exp_name` is already an absolute path, it will just return `resume_exp_name`
    
    # resume from a specified checkpoint or from the latest
    if cfg.resume_step is not None:
      ckpt_path = os.path.join(ckpt_dir, f'ckpt_step_{cfg.resume_step}.pth')
    else:
      ckpt_path = _latest_checkpoint(ckpt_dir, prefix='ckpt_step_')
    
    # load checkpoint on cpu to later avoid OOM when intializing the model on device
    # (an alternative design would be to initialize the model on `meta` device instead)
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')

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
  
  
