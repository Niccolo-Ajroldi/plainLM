
import os
import re
import torch

# from flax.training.checkpoints import latest_checkpoint


def _latest_checkpoint(ckpt_dir: str, prefix: str = 'checkpoint_') -> str | None:
    """Retrieve the latest checkpoint path in a directory."""
    if not os.path.isdir(ckpt_dir):
        return None

    # List all files matching the prefix pattern
    checkpoints = [f for f in os.listdir(ckpt_dir) if re.match(rf"^{prefix}\d+$", f)]
    checkpoints.sort(key=lambda x: int(x[len(prefix):]))  # Sort numerically

    return os.path.join(ckpt_dir, checkpoints[-1]) if checkpoints else None


def save_checkpoint(micro_step, model, engine, cfg, job_idx=None):

  optimizer = engine.optimizer
  scheduler = engine.scheduler
  scaler = engine.scaler
  
  state = {
    "micro_step": micro_step,
    "state_dict": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict() if scheduler else {},
    "scaler": scaler.state_dict()
  }

  exp_dir = os.path.join(cfg.out_dir, cfg.exp_name)
  if job_idx is not None:  # subfolder for each job in the sweep
    exp_dir = os.path.join(exp_dir, f"job_idx_{job_idx}")
    
  save_path = os.path.join(exp_dir, f'ckpt_micro_step_{micro_step}.pth')
  print(f"Saving checkpoint to {save_path}")
  torch.save(state, save_path)
  print(f"Successfully saved checkpoint!")


def maybe_load_checkpoint(cfg, device):
  
  ckpt = None
  micro_step_start = 0
  
  if cfg.resume:
    
    # resume from a specified exp or from the same exp
    exp_name = cfg.resume_exp_name if cfg.resume_exp_name is not None else cfg.exp_name
    ckpt_dir = os.path.join(cfg.out_dir, exp_name)
    print(f"Resuming from {ckpt_dir}")
    
    # resume from a specified checkpoint or from the latest
    if cfg.resume_micro_step is not None:
      ckpt_path = os.path.join(ckpt_dir, f'ckpt_micro_step_{cfg.resume_micro_step}.pth')
    else:
      ckpt_path = _latest_checkpoint(ckpt_dir, prefix='ckpt_')
    
    # load checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    micro_step_start = ckpt['micro_step']
  
  return ckpt, micro_step_start
