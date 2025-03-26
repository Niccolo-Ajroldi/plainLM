
from collections import deque

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

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
      state_dict = ckpt['state_dict']
      
      # # import pdb
      # # pdb.set_trace()

      if isinstance(self.model, DDP):
        state_dict = {"module." + k: v for k, v in state_dict.items()}

      # if self.torch_compile:
      #   state_dict = {"_orig_mod." + k: v for k, v in state_dict.items()}

      # print(state_dict.keys())
      # print(self.model.state_dict().keys())
      
      # Write state_dict in a list of params
      new_params = [state_dict[n] for n,_ in self.model.named_parameters()]

      # Subtract oldest element from running avg
      if len(self.queue) == self.maxlen:
        outdated = self.queue[0]
        for p_avg, p_outdated in zip(self.avg, outdated):
          p_avg.sub_(p_outdated.div(self.maxlen))

      # Update running avg with new params
      if self.avg is None:
        self.avg = [p.clone() for p in new_params]
      else:
        for avg, new_p in zip(self.avg, new_params):
          avg.add_(new_p.div(self.maxlen))

      # append pushes the new element into the queue and pops the oldest
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
