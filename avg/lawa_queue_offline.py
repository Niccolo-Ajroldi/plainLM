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

from absl import logging

from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup

USE_PYTORCH_DDP = pytorch_setup()[0]


class LAWA():
  def __init__(self, hyperparameters, workload) -> None:
    self.maxlen = int(hyperparameters.lawa_queue_len)
    self.queue = deque(maxlen=self.maxlen)
    self.running_avg = None

    self.start_step = math.ceil(workload.step_hint * hyperparameters.lawa_burnin_pct)

    has_pct = getattr(hyperparameters, "lawa_every_pct", None) is not None
    has_step = getattr(hyperparameters, "lawa_every_steps", None) is not None
    if not has_pct and not has_step:
      raise ValueError("Missing hyperparameter: lawa_every_steps or lawa_every_pct")
    if has_step and has_pct:
      raise ValueError("Both lawa_every_steps and lawa_every_pct are defined")

    if has_step:
      self.every_step = int(hyperparameters.lawa_every_steps)
    else:
      self.every_step = math.ceil(workload.step_hint * hyperparameters.lawa_every_pct)
    logging.info('=== Running LAWA with self.every_step = %d ===', self.every_step)

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


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates a NAdamW optimizer and a learning rate schedule."""
  del model_params
  del model_state
  del rng

  optimizer_state = {}
  optimizer_state['lawa'] = LAWA(hyperparameters, workload)

  return optimizer_state


def update_params(workload: spec.Workload,
                  current_param_container: spec.ParameterContainer,
                  current_params_types: spec.ParameterTypeTree,
                  model_state: spec.ModelAuxiliaryState,
                  hyperparameters: spec.Hyperparameters,
                  batch: Dict[str, spec.Tensor],
                  loss_type: spec.LossType,
                  optimizer_state: spec.OptimizerState,
                  eval_results: List[Tuple[int, float]],
                  global_step: int,
                  rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del workload
  del current_params_types
  del hyperparameters
  del batch
  del loss_type
  del eval_results
  del rng

  lawa = optimizer_state['lawa']
  current_model = current_param_container

  # Update LAWA
  if global_step >= lawa.start_step and global_step % lawa.every_step == 0:
    lawa.append(current_model.parameters())
  
  # logging.info(f"LAWA --- Queue length = {len(lawa.queue)}")

  return (optimizer_state, current_param_container, model_state)


def prepare_for_eval(workload: spec.Workload,
                     current_param_container: spec.ParameterContainer,
                     current_params_types: spec.ParameterTypeTree,
                     model_state: spec.ModelAuxiliaryState,
                     hyperparameters: spec.Hyperparameters,
                     loss_type: spec.LossType,
                     optimizer_state: spec.OptimizerState,
                     eval_results: List[Tuple[int, float]],
                     global_step: int,
                     rng: spec.RandomState) -> spec.UpdateReturn:
  del workload
  del current_params_types
  del hyperparameters
  del loss_type
  del eval_results
  del rng

  lawa = optimizer_state['lawa']
  current_model = current_param_container

  if global_step < lawa.start_step or not lawa.full():
    return (optimizer_state, current_model, model_state)

  # logging.info(f"LAWA --- Loading avg into model")

  # Load avg into model
  if lawa.full():  # redundant
    avg = lawa.avg()  # compute avg on CPU
    for p, p_avg in zip(current_model.parameters(), avg):
      p.data.copy_(p_avg.data)  # move avg to GPU

  return (optimizer_state, current_model, model_state)

