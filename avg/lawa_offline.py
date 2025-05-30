"""
Engine class for averaging weights with LAWA.

We keep the EMA in CPU memory.
"""

from collections import deque

import torch

from checkpoint_utils import match_state_dict_keys
from avg.avg import AvgEngine


class LAWAOffline(AvgEngine):
  """A module to average weigths."""

  def __init__(self, model, cfg, device, local_rank):
    # Init parent class. Here model is updated only in prep for eval.
    super().__init__(model, cfg, device, local_rank)

    # LAWA-specific hyperparameters
    if not hasattr(cfg, 'lawa_queue_len'):
      raise KeyError('Missing LAWA queue length.')
    if cfg.lawa_queue_len < 0:
      raise ValueError('Invalid LAWA queue length value.')
    self.maxlen = cfg.lawa_queue_len

    # LAWA buffers: queue and running avg, both on CPU
    self.queue = deque(maxlen=self.maxlen)
    self.avg = None


  @torch.no_grad()
  def maybe_update_buffer(self, ckpt_path, step):
    """Update LAWA from a ckpt_path."""
    if step >= self.avg_start_step and step % self.avg_every_steps == 0:

      print(f"Updating LAWA")

      # Load checkpoint on CPU
      ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
      ckpt_state_dict = ckpt['state_dict']

      # match state_dict keys
      ckpt_state_dict = match_state_dict_keys(ckpt_state_dict, self.model.state_dict())

      # Write state_dict in a list of params
      new_params = [ckpt_state_dict[n] for n,_ in self.model.named_parameters()]

      k = self.maxlen
      if self.avg is None: 
        self.avg = [p.clone().div(k) for p in new_params]
      else: 
        # Subtract oldest element from running avg
        if len(self.queue) == self.maxlen:
          old = self.queue[0]
          for p_avg, p_old in zip(self.avg, old):
            p_avg.sub_(p_old.div(k))

        # Update running avg with new params
        for avg, new_p in zip(self.avg, new_params):
          avg.add_(new_p.div(k))

      # append right automatically popleft() when deque has a maxlen set and is full
      self.queue.append(new_params)


  @torch.no_grad()
  def prepare_for_eval(self):
    """Load running average (cpu) into the self.model (cuda)."""
    if self.avg is None:
      raise ValueError(f"Evaluating without a model!")
    print(f"Load avg into model")
    for p, p_avg in zip(self.model.parameters(), self.avg):
      p.copy_(p_avg.to(p.device))          
      # p.data.copy_(p_avg.data)  # also moves to cuda
