"""Intialize optimizer and scheduler."""

import torch

from .lr_schedule import WSD, LinearCooldown, WarmupConstant, WarmupCosine
from models import get_param_groups


def intialize_optimizer(model, cfg):
  """
  Intialize an optimizer.
  NOTE: we pass weight_decay to optim, but it gets overwritten by the weight_decay in param_groups!
  """
  optimizers = {}

  if cfg.optim == "adamw":
    param_groups = get_param_groups(model, cfg.weight_decay)
    optimizers[cfg.optim] = torch.optim.AdamW(
      param_groups,
      lr=cfg.lr,
      betas=[cfg.beta1, cfg.beta2],
      weight_decay=cfg.weight_decay,
      fused=getattr(cfg, "fused_optim", True),
      eps=getattr(cfg, "eps", 1e-8),
    )
    

  elif cfg.optim == "sgd":
    param_groups = get_param_groups(model, cfg.weight_decay)
    optimizers[cfg.optim] = torch.optim.SGD(
      param_groups,
      lr=cfg.lr,
      momentum=cfg.beta1,
      dampening=cfg.dampening,
      weight_decay=cfg.weight_decay,
    )

  elif cfg.optim == "zero1adamw":
    from optim.zero1adamw import ZeRO1AdamW
    param_groups = get_param_groups(model, cfg.weight_decay)
    optimizers[cfg.optim] = ZeRO1AdamW(
      param_groups,
      lr=cfg.lr,
      betas=[cfg.beta1, cfg.beta2],
      weight_decay=cfg.weight_decay,
      adamc_wd=getattr(cfg, "adamc_wd", False),
      eps=getattr(cfg, "eps", 1e-8),
    )

  elif cfg.optim == "muonVanilla":
    from optim.muon import MuonVanilla
    from optim.muon import split_params_muon_adam
    muon_params, adam_params = split_params_muon_adam(model)  
    optimizers['muon'] = MuonVanilla(
        muon_params,
        lr=cfg.muon_lr,
        weight_decay=cfg.muon_weight_decay,
        beta=cfg.muon_beta,
        nesterov=cfg.muon_nesterov,
        ns_steps=cfg.muon_ns_steps,
        ns_eps=cfg.muon_ns_eps
    )
    optimizers['adamw'] = torch.optim.AdamW(
        adam_params,
        lr=cfg.adamw_lr,
        weight_decay=cfg.adamw_weight_decay,
        betas=(cfg.adamw_beta1, cfg.adamw_beta2),
        eps=cfg.adamw_eps,
        fused=cfg.adamw_fused,
    )

  elif cfg.optim == "muonDP":
    from optim.muon import MuonDP
    from optim.muon import split_params_muon_adam
    muon_params, adam_params = split_params_muon_adam(model)  
    optimizers['muon'] = MuonDP(
        muon_params,
        lr=cfg.muon_lr,
        weight_decay=cfg.muon_weight_decay,
        beta=cfg.muon_beta,
        nesterov=cfg.muon_nesterov,
        ns_steps=cfg.muon_ns_steps,
        ns_eps=cfg.muon_ns_eps
    )
    optimizers['adamw'] = torch.optim.AdamW(
        adam_params,
        lr=cfg.adamw_lr,
        weight_decay=cfg.adamw_weight_decay,
        betas=(cfg.adamw_beta1, cfg.adamw_beta2),
        eps=cfg.adamw_eps,
        fused=cfg.adamw_fused,
    )

  else:
    raise NotImplementedError(f"Not implemented optim: {cfg.optim}.")

  return optimizers


def initialize_scheduler(optimizer, cfg):
  if cfg.scheduler is None:
    return None

  ## Number of warmup steps
  # either specified directly (int) or as a fraction of steps_budget (float)
  if getattr(cfg, "warmup_steps", None) is not None:
    warmup_steps = (
      cfg.warmup_steps
      if isinstance(cfg.warmup_steps, int)
      else int(cfg.warmup_steps * cfg.steps_budget)
    )

  ## Number of cooldown steps
  # either specified directly (int) or as a fraction of steps_budget (float)
  if getattr(cfg, "cooldown_steps", None) is not None:
    cooldown_steps = (
      cfg.cooldown_steps
      if isinstance(cfg.cooldown_steps, int)
      else int(cfg.cooldown_steps * cfg.steps_budget)
    )

  ## Final LR of the schedule
  # either specified directly via `lr_end` or as a fraction of top lr via `lr_end_pct`
  if getattr(cfg, "lr_end", None) is not None or getattr(cfg, "lr_end_pct", None) is not None:
    lr_end = cfg.lr_end if (cfg.lr_end is not None) else (cfg.lr_end_pct * cfg.lr)

  if cfg.scheduler == "warmup_cosine":
    scheduler = WarmupCosine(
      optimizer,
      lr_start=cfg.lr_start,
      lr_max=cfg.lr,
      lr_end=lr_end,
      warmup_steps=warmup_steps,
      T=cfg.steps_budget,
    )

  elif cfg.scheduler == "wsd":
    cooldown_start_step = cfg.steps_budget - cooldown_steps
    scheduler = WSD(
      optimizer,
      lr_start=cfg.lr_start,
      lr_max=cfg.lr,
      lr_end=lr_end,
      warmup_steps=warmup_steps,
      cooldown_start_step=cooldown_start_step,
      cooldown_steps=cooldown_steps,
    )

  elif cfg.scheduler == "warmup_constant":
    scheduler = WarmupConstant(
      optimizer,
      lr_start=cfg.lr_start,
      lr_max=cfg.lr,
      warmup_steps=warmup_steps,
    )

  elif cfg.scheduler == "linear_cooldown":
    cooldown_start_step = cfg.resume_step
    scheduler = LinearCooldown(
      optimizer,
      lr_max=cfg.lr,
      lr_end=lr_end,
      cooldown_start_step=cooldown_start_step,
      cooldown_steps=cooldown_steps,
    )

  else:
    raise NotImplementedError(f"Not implemented scheduler: {cfg.scheduler}.")

  return scheduler
