"""
Engine class for averaging weights with LEWA.

We keep the LEWA in CPU memory.
"""

from collections import deque

import torch

from checkpoint_utils import match_state_dict_keys
from avg.avg import AvgEngine


class LEWAOffline(AvgEngine):
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

    # EWA-specific hyperparameters
    if not hasattr(cfg, 'ewa_beta'):
      raise KeyError('Missing EWA beta.')
    if cfg.ewa_beta < 0 or cfg.ewa_beta > 1:
      raise ValueError('Invalid EWA beta value.')
    self.beta = cfg.ewa_beta

    # LAWA buffer: queue on CPU
    self.queue = deque(maxlen=self.maxlen)

    # Order in which to loop over ckpts
    self.reversed_ewa = getattr(cfg, 'reversed_ewa', False)


  @torch.no_grad()
  def maybe_update_buffer(self, ckpt_path, step):
    """Update LAWA from a ckpt_path."""
    if step >= self.avg_start_step and step % self.avg_every_steps == 0:

      print(f"Storing into Queue")

      # Load checkpoint on CPU
      ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
      ckpt_state_dict = ckpt['state_dict']

      # match state_dict keys
      ckpt_state_dict = match_state_dict_keys(ckpt_state_dict, self.model.state_dict())

      # Write state_dict in a list of params, ordered by model.named_parameters()
      new_params = [ckpt_state_dict[n] for n,_ in self.model.named_parameters()]

      # append right automatically popleft() when deque has a maxlen set and is full
      self.queue.append(new_params)


  @torch.no_grad()
  def prepare_for_eval(self):
    """Compute EMA (cpu) and load into self.model (cuda)."""
    if len(self.queue) == 0:
      raise ValueError(f"Evaluating without an average!")

    print("Computing LEWA")

    if getattr(self, 'reversed_ewa', False):
      queue_iter = reversed(self.queue) # reversed EMA (newest -> oldest)
    else:
      queue_iter = iter(self.queue) # forward EMA (oldest -> newest)

    ewa = [p.clone() for p in next(queue_iter)] # init from first element in chosen order
    beta = self.beta
    for params in queue_iter:
        for e, p in zip(ewa, params):
            e.mul_(beta).add_(p, alpha=1 - beta)
    
    print(f"Load avg into model")
    for p, e in zip(self.model.parameters(), ewa):
        p.copy_(e.to(p.device))

