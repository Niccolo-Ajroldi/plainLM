"""
Engine class for averaging weights with EWA.

We keep the EMA in CUDA memory.
EMA is stored directly in model.parameters().
Hence EMA is initialized at model initialization.
"""

import torch

from checkpoint_utils import match_state_dict_keys
from avg.avg import AvgEngine


class EWAOffline(AvgEngine):
  """A module to average weigths."""

  def __init__(self, model, cfg, device, local_rank):
    # Init parent class. Here model stores the EMA!
    super().__init__(model, cfg, device, local_rank)

    # EMA-specific hyperparameters
    if not hasattr(cfg, 'ewa_beta'):
      raise KeyError('Missing EWA beta.')
    if cfg.ewa_beta < 0 or cfg.ewa_beta > 1:
      raise ValueError('Invalid EWA beta value.')
    self.beta = cfg.ewa_beta


  @torch.no_grad()
  def maybe_update_buffer(self, ckpt_path, step):
    """Update EMA from a ckpt_path."""
    if step >= self.avg_start_step and step % self.avg_every_steps == 0:

      print(f"Update EMA")

      # Load checkpoint in CUDA
      ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
      ckpt_state_dict = ckpt['state_dict']

      # match state_dict keys
      ckpt_state_dict = match_state_dict_keys(ckpt_state_dict, self.model.state_dict())

      # Update EMA
      beta = self.beta
      for n, p in self.model.named_parameters():
        p.mul_(beta).add_(ckpt_state_dict[n], alpha=1-beta)

  @torch.no_grad()
  def prepare_for_eval(self):
    """Prepare for evaluation, by loading the avg into self.model."""
    pass

