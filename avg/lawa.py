"""
Offline LAWA Queue

Hyperparameters:
  - lawa_burnin_pct
  - lawa_every_pct
  - lawa_queue_len

"""

import math
from typing import Dict, Iterator, List, Tuple

from collections import deque
from itertools import islice
from torch import distributed as dist


USE_PYTORCH_DDP = dist._is_initalized()


class LAWA():
  def __init__(self, cfg, step_budget) -> None:
    self.maxlen = int(cfg.lawa_queue_len)
    self.queue = deque(maxlen=self.maxlen)
    self.running_avg = None

    self.start_step = math.ceil(step_budget * cfg.lawa_burnin_pct)

    has_pct = getattr(cfg, "lawa_every_pct", None) is not None
    has_step = getattr(cfg, "lawa_every_steps", None) is not None
    if not has_pct and not has_step:
      raise ValueError("Missing hyperparameter: lawa_every_steps or lawa_every_pct")
    if has_step and has_pct:
      raise ValueError("Both lawa_every_steps and lawa_every_pct are defined")

    if has_step:
      self.every_step = int(cfg.lawa_every_steps)
    else:
      self.every_step = math.ceil(step_budget * cfg.lawa_every_pct)
    print('=== Running LAWA with self.every_step = %d ===', self.every_step)

  def append(self, params):
    new_params = [p.detach().cpu() for p in params]

    # Remove oldest element from the running avg
    if self.full():
      removed_params = self.queue[0]
      for avg, removed_p in zip(self.running_avg, removed_params):
        avg.sub_(removed_p.div(self.maxlen))

    # Update running average with the new parameters
    if self.running_avg is None:
      self.running_avg = [p.clone() for p in new_params]
    else:
      for avg, new_p in zip(self.running_avg, new_params):
        avg.add_(new_p.div(self.maxlen))

    # append pushes the new element into the queue and pops the oldest
    self.queue.append(new_params)

  def full(self):
    return len(self.queue) == self.maxlen

  def avg(self):
    """Returns the average tensor, which lays on CPU."""
    k = float(self.maxlen)

    # Initialize avg with first element of the queue
    q_avg = [p.clone().div_(k) for p in self.queue[0]] # self.queue[0] is already on cpu!

    # Loop over queue and update avg
    for chkpts in islice(self.queue, 1, None):
      for p_avg,p in zip(q_avg, chkpts):
        p_avg.add_(p/k)

    return q_avg
  
  def state_dict(self):
    return {key: value for key, value in self.__dict__.items()}

  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)

