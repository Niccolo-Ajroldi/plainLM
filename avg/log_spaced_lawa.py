"""
Engine class for averaging weights with LogSpacedLAWA.

We store a DDP model on CUDA.

Hyperparams:
- `\\nu`
- `L`

At time `t`:
- define grid `g = {t - \\nu^{n}}_{n=1}^{L}`
- intersect grid with `avail_steps`
- loop over grid, load checkpoints and average them, store them on cpu
"""

import os
import torch
from typing import List

from checkpoint_utils import match_state_dict_keys
from avg.avg import AvgEngine


class LogSpacedLAWA(AvgEngine):
  """Log spaced LAWA."""

  def __init__(
        self, model, cfg, device, local_rank, 
        avail_steps: List, ckpt_folder: str, prefix: str,
    ):
    # Init parent class. Stores model to CUDA.
    super().__init__(model, cfg, device, local_rank)

    # LAWA-specific hyperparameters
    if not hasattr(cfg, 'lawa_queue_len'):
      raise KeyError('Missing LAWA queue length.')
    if cfg.lawa_queue_len < 0:
      raise ValueError('Invalid LAWA queue length value.')
    self.maxlen = cfg.lawa_queue_len
    
    # Infos about the checkpoints
    self.avail_steps = set(avail_steps)
    self.ckpt_folder = ckpt_folder
    self.prefix = prefix
    self.save_freq = avail_steps[1] - avail_steps[0]

    # Empty state dict to hold the average
    self.avg_state = None

  @torch.no_grad()
  def maybe_update_buffer(self, step):
    """Compute average at step t."""
    
    nu = self.avg_every_steps
    l = self.maxlen

    # Define averaging grid
    grid = [step] + [step - nu**i for i in range(1, l)]
    print(f"Grid: {grid}")

    # Round to closest multiple of save_freq
    grid = [g // self.save_freq * self.save_freq for g in grid]
    
    # Intersect with available checkpoints (and deduplicates after rounding)
    grid = sorted(list(set(grid) & self.avail_steps))
    print(f"Intersect grid: {grid}")

    if len(grid) == 0:
      return False

    # Empty state dict to hold the average
    self.avg_state = None

    # Load and average checkpoints in the grid
    for step in grid:

      # Load checkpoint on CPU
      ckpt_file = f"{self.prefix}{step}.pth"
      ckpt_path = os.path.join(self.ckpt_folder, ckpt_file)
      ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
      ckpt_state_dict = ckpt['state_dict']

      # Match state_dict keys
      ckpt_state_dict = match_state_dict_keys(ckpt_state_dict, self.model.state_dict())

      # Accumulate tensors in `avg_state`
      if self.avg_state is None:
        self.avg_state = {k: v.clone() for k, v in ckpt_state_dict.items()}
      else:
        for k in self.avg_state:
          self.avg_state[k] += ckpt_state_dict[k]

    # Final average
    for k in self.avg_state:
      self.avg_state[k] /= len(grid)

    # If we have at least one point, we can eval
    return True
  


  @torch.no_grad()
  def prepare_for_eval(self):
    """Load average state_dict (cpu) into the self.model (cuda)."""
    if self.avg_state is None:
      raise ValueError(f"Evaluating without an averaged state!")
    print(f"Load avg into model")
    self.model.load_state_dict(self.avg_state)
    
