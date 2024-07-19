"""Custom implementation of LR schedules."""

import math


class WarmupCosine(object):
  """Linear warmup followed by cosine decay"""
  def __init__(self, optimizer, lr_start, lr_max, lr_end, warmup_steps, T):
    self.optimizer = optimizer
    self.lr_start = lr_start
    self.lr_max = lr_max
    self.lr_end = lr_end
    self.warmup_steps = warmup_steps
    self.T = T
    self.iter = 0
    for group in self.optimizer.param_groups:
      group["lr"] = lr_start

  def schedule(self, t):
    """returns lr(t), where t is the current step"""
    if t <= self.warmup_steps:
      return self.lr_start + (self.lr_max-self.lr_start)/self.warmup_steps * t
    elif t <= self.T:
      progress = (t-self.warmup_steps) / (self.T-self.warmup_steps)
      return self.lr_end + 0.5 * (self.lr_max-self.lr_end) * (1 + math.cos(math.pi * progress))
    return self.lr_end

  def step(self):
    self.iter += 1
    lr = self.schedule(self.iter)
    for group in self.optimizer.param_groups:
      group["lr"] = lr

  def state_dict(self):
    return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)


class WSD(object):
  """Trapezoidal schedule / WSD: (linear) Warmup, Stable, (linear) Decay"""
  def __init__(self, optimizer, lr_start, lr_max, lr_end, warmup_steps, cooldown_start_step, cooldown_steps):
    self.optimizer = optimizer
    self.lr_start = lr_start
    self.lr_max = lr_max
    self.lr_end = lr_end
    self.warmup_steps = warmup_steps
    self.cooldown_start_step = cooldown_start_step
    self.cooldown_steps = cooldown_steps
    self.iter = 0
    
    for group in self.optimizer.param_groups:
      group["lr"] = lr_start

  def schedule(self, t):
    """returns lr(t), where t is the current step"""
    if t <= self.warmup_steps:
      return self.lr_start + (self.lr_max-self.lr_start)/self.warmup_steps * t
    elif t <= self.cooldown_start_step:
      return self.lr_max
    return self.lr_max + (self.lr_end-self.lr_max)/self.cooldown_steps * (t-self.cooldown_start_step)

  def step(self):
    """computes new lr and sets it in self.optimizer"""
    self.iter += 1
    lr = self.schedule(self.iter)
    for group in self.optimizer.param_groups:
      group["lr"] = lr

  def state_dict(self):
    return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)


class WarmupConstant(object):
  """Linear Warmup + Constant LR"""
  def __init__(self, optimizer, lr_start, lr_max, warmup_steps):
    self.optimizer = optimizer
    self.lr_start = lr_start
    self.lr_max = lr_max
    self.warmup_steps = warmup_steps
    self.iter = 0
    for group in self.optimizer.param_groups:
      group["lr"] = lr_start

  def schedule(self, t):
    """returns lr(t), where t is the current step"""
    if t <= self.warmup_steps:
      return self.lr_start + (self.lr_max-self.lr_start)/self.warmup_steps * t
    return self.lr_max

  def step(self):
    """computes new lr and sets it in self.optimizer"""
    self.iter += 1
    lr = self.schedule(self.iter)
    for group in self.optimizer.param_groups:
      group["lr"] = lr

  def state_dict(self):
    return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)

