"""Custom implementation of LR schedules."""

import math
from abc import ABC, abstractmethod


class CustomLRSchedule(ABC):
  """An abstract parent class for custom LR Schedules."""
  def __init__(self, optimizer):
    self.optimizer = optimizer

  def set_optim_lr(self, lr):
    """Set a learning rate for all parameter groups."""
    for group in self.optimizer.param_groups:
      group["lr"] = lr

  def state_dict(self):
    return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)
  
  @abstractmethod
  def step(self):
    pass


class WarmupCosine(CustomLRSchedule):
  """Linear warmup followed by Cosine Decay."""
  def __init__(self, optimizer, lr_start, lr_max, lr_end, warmup_steps, T):
    super().__init__(optimizer)
    self.lr_start = lr_start
    self.lr_max = lr_max
    self.lr_end = lr_end
    self.warmup_steps = warmup_steps
    self.T = T
    self.iter = 0
    self.set_optim_lr(lr_start)

  def get_lr(self, t):
    """Computes and returns lr(t), where t is the current step."""
    if t <= self.warmup_steps:
      return self.lr_start + (self.lr_max-self.lr_start)/self.warmup_steps * t
    elif t <= self.T:
      progress = (t-self.warmup_steps) / (self.T-self.warmup_steps)
      return self.lr_end + 0.5 * (self.lr_max-self.lr_end) * (1 + math.cos(math.pi * progress))
    return self.lr_end

  def step(self):
    self.iter += 1
    lr = self.get_lr(self.iter)
    self.set_optim_lr(lr)


class WarmupLinearDecay(CustomLRSchedule):
  """Linear warmup followed by linear decay."""
  def __init__(self, optimizer, lr_start, lr_max, lr_end, warmup_steps, T):
    super().__init__(optimizer)
    self.lr_start = lr_start
    self.lr_max = lr_max
    self.lr_end = lr_end
    self.warmup_steps = warmup_steps
    self.T = T
    self.iter = 0
    self.set_optim_lr(lr_start)

  def get_lr(self, t):
    if t <= self.warmup_steps:
      return self.lr_start + (self.lr_max - self.lr_start) / self.warmup_steps * t
    elif t <= self.T:
      return self.lr_max + (self.lr_end-self.lr_max) / (self.T-self.warmup_steps) * (t-self.warmup_steps)
    return self.lr_end

  def step(self):
    self.iter += 1
    self.set_optim_lr(self.get_lr(self.iter))


class WSD(CustomLRSchedule):
  """Trapezoidal schedule / WSD: (linear) Warmup + Stable + (linear) Decay."""
  def __init__(self, optimizer, lr_start, lr_max, lr_end, warmup_steps, cooldown_start_step, cooldown_steps):
    super().__init__(optimizer)
    self.lr_start = lr_start
    self.lr_max = lr_max
    self.lr_end = lr_end
    self.warmup_steps = warmup_steps
    self.cooldown_start_step = cooldown_start_step
    self.cooldown_steps = cooldown_steps
    self.iter = 0
    self.set_optim_lr(lr_start)

  def get_lr(self, t):
    """Computes and returns lr(t), where t is the current step."""
    if t <= self.warmup_steps:
      return self.lr_start + (self.lr_max-self.lr_start)/self.warmup_steps * t
    elif t <= self.cooldown_start_step:
      return self.lr_max
    return self.lr_max + (self.lr_end-self.lr_max)/self.cooldown_steps * (t-self.cooldown_start_step)

  def step(self):
    self.iter += 1
    lr = self.get_lr(self.iter)
    self.set_optim_lr(lr)


class WarmupConstant(CustomLRSchedule):
  """Linear Warmup + Constant LR."""
  def __init__(self, optimizer, lr_start, lr_max, warmup_steps):
    super().__init__(optimizer)
    self.lr_start = lr_start
    self.lr_max = lr_max
    self.warmup_steps = warmup_steps
    self.iter = 0
    self.set_optim_lr(lr_start)

  def get_lr(self, t):
    """Computes and returns lr(t), where t is the current step."""
    if t <= self.warmup_steps:
      return self.lr_start + (self.lr_max-self.lr_start)/self.warmup_steps * t
    return self.lr_max

  def step(self):
    self.iter += 1
    lr = self.get_lr(self.iter)
    self.set_optim_lr(lr)


class LinearCooldown(CustomLRSchedule):
  """Linear Cooldown."""
  def __init__(self, optimizer, lr_max, lr_end, cooldown_start_step, cooldown_steps):
    super().__init__(optimizer)
    self.lr_max = lr_max
    self.lr_end = lr_end
    self.cooldown_start_step = cooldown_start_step
    self.cooldown_steps = cooldown_steps
    self.iter = 0

  def get_lr(self, t):
    """Computes and returns lr(t), where t is the current step."""
    if t <= self.cooldown_start_step:
      return self.lr_max
    return self.lr_max + (self.lr_end-self.lr_max)/self.cooldown_steps * (t-self.cooldown_start_step)

  def step(self):
    self.iter += 1
    lr = self.get_lr(self.iter)
    self.set_optim_lr(lr)

  def load_state_dict(self, state_dict):
    # We load only the iter parameter from the saved state dict.
    self.iter = state_dict.get("iter", 0)


class WarmupInverseSqrt(CustomLRSchedule):
  """Linear warmup followed by Inverse Square Root Decay."""
  def __init__(self, optimizer, lr_start, lr_max, warmup_steps):
    super().__init__(optimizer)
    self.lr_start = lr_start
    self.lr_max = lr_max
    self.warmup_steps = warmup_steps
    self.iter = 0
    self.set_optim_lr(lr_start)

  def get_lr(self, t):
    """Computes and returns lr(t), where t is the current step."""
    if t <= self.warmup_steps:
      return self.lr_start + (self.lr_max-self.lr_start)/self.warmup_steps * t
    return self.lr_max / ((t - self.warmup_steps) ** 0.5)
    ## fairseq style would be:
    # return self.lr_max * (self.warmup_steps ** 0.5) / (t ** 0.5) 

  def step(self):
    self.iter += 1
    lr = self.get_lr(self.iter)
    self.set_optim_lr(lr)


class OneMinSqrtCooldown(CustomLRSchedule):
  """OneMinSqrt Cooldown https://arxiv.org/pdf/2405.18392, page 4."""
  def __init__(self, optimizer, lr_max, lr_end, cooldown_start_step, cooldown_steps):
    super().__init__(optimizer)
    self.lr_max = lr_max
    self.lr_end = lr_end
    self.cooldown_start_step = cooldown_start_step
    self.cooldown_steps = cooldown_steps
    self.iter = 0

  def get_lr(self, t):
    """Computes and returns lr(t), where t is the current step."""
    if t <= self.cooldown_start_step:
      return self.lr_max
    return self.lr_max - self.lr_max * ((t - self.cooldown_start_step) / (self.cooldown_steps)) ** 0.5

  def step(self):
    self.iter += 1
    lr = self.get_lr(self.iter)
    self.set_optim_lr(lr)

  def load_state_dict(self, state_dict):
    # We load only the iter parameter from the saved state dict.
    self.iter = state_dict.get("iter", 0)

class OneMinSqrtFromInvSqrtCooldown(CustomLRSchedule):
  """OneMinSqrt Cooldown from a Inverse Sqrt schedule."""
  def __init__(self, optimizer, lr_start, lr_max, lr_end, warmup_steps, cooldown_start_step, cooldown_steps):
    super().__init__(optimizer)
    self.lr_start = lr_start
    self.lr_max = lr_max
    self.lr_end = lr_end
    self.warmup_steps = warmup_steps
    self.cooldown_start_step = cooldown_start_step
    self.cooldown_steps = cooldown_steps
    self.iter = 0
    self.lr_before_decay = self.get_lr_inv_sqrt(cooldown_start_step)
    self.set_optim_lr(self.lr_before_decay)

  def get_lr_inv_sqrt(self, t):
    """Get LR from the baseline InvSqrt schedule."""
    if t <= self.warmup_steps:
      return self.lr_start + (self.lr_max-self.lr_start)/self.warmup_steps * t
    return self.lr_max / ((t - self.warmup_steps) ** 0.5)

  def get_lr(self, t):
    """Computes and returns lr(t), where t is the current step."""
    if t <= self.cooldown_start_step:
      return self.lr_before_decay
    return self.lr_before_decay - self.lr_before_decay * ((t - self.cooldown_start_step) / (self.cooldown_steps)) ** 0.5

  def step(self):
    self.iter += 1
    lr = self.get_lr(self.iter)
    self.set_optim_lr(lr)

  def load_state_dict(self, state_dict):
    # We load only the iter parameter from the saved state dict.
    self.iter = state_dict.get("iter", 0)
