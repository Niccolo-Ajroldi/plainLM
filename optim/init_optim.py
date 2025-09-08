"""Intialize optimizer and scheduler."""

from .lr_schedule import WarmupCosine, WarmupLinearDecay, WSD, WarmupConstant, LinearCooldown

def intialize_optimizer(model, cfg):
  """
  Intialize an optimizer.
  NOTE: we pass weight_decay to optim, but it gets overwritten by the weight_decay in param_groups!
  """
  
  if cfg.optim == 'adamw':
    from torch.optim import AdamW
    param_groups = get_param_groups_default(model, cfg)
    optimizer = AdamW(
      param_groups,
      lr=cfg.lr,
      betas=[cfg.beta1, cfg.beta2],
      weight_decay=cfg.weight_decay,
      fused=cfg.fused_optim,
      eps=getattr(cfg, 'eps', 1e-8)
    )

  elif cfg.optim == 'custom_adam':
    from .custom_adam import CustomAdam
    param_groups = get_param_groups_default(model, cfg)
    optimizer = CustomAdam(
      param_groups,
      lr=cfg.lr,
      betas=[cfg.beta1, cfg.beta2],
      weight_decay=cfg.weight_decay,
      corrected_weight_decay=cfg.corrected_weight_decay,
      eps=getattr(cfg, 'eps', 1e-8)
    )

  elif cfg.optim == 'nadamw':
    from torch.optim import NAdam
    param_groups = get_param_groups_default(model, cfg)
    optimizer = NAdam(
      param_groups,
      lr=cfg.lr,
      betas=[cfg.beta1, cfg.beta2],
      weight_decay=cfg.weight_decay,
      decoupled_weight_decay=True,
      fused=cfg.fused_optim,
      eps=getattr(cfg, 'eps', 1e-8)
    )
  
  elif cfg.optim == 'sgd':
    from torch.optim import SGD
    param_groups = get_param_groups_default(model, cfg)
    optimizer = SGD(
      param_groups,
      lr=cfg.lr,
      momentum=cfg.beta1,
      dampening=cfg.dampening,
      weight_decay=cfg.weight_decay,
    )
  
  elif cfg.optim == 'signSGD':
    from .signSGD import signSGD
    param_groups = get_param_groups_default(model, cfg)
    optimizer = signSGD(
      param_groups,
      lr=cfg.lr,
      momentum=cfg.beta1,
      dampening=cfg.dampening,
      weight_decay=cfg.weight_decay,
    )
  
  elif cfg.optim == 'sfo_adamw':
    import schedulefree
    param_groups = get_param_groups_default(model, cfg)
    # warmup steps for schedulefree must be specified here
    warmup_steps = cfg.warmup_steps if isinstance(cfg.warmup_steps, int) \
      else int(cfg.warmup_steps * cfg.steps_budget)
    optimizer = schedulefree.AdamWScheduleFree(
      param_groups,
      lr=cfg.lr,
      warmup_steps=warmup_steps,
      betas=[cfg.beta1, cfg.beta2],
      weight_decay=cfg.weight_decay,
    )

  elif cfg.optim == 'muon':
    from dion import Muon
    from .custom_muon import get_param_groups_muon
    param_groups = get_param_groups_muon(model, cfg)
    optimizer = Muon(
      param_groups,
      lr=cfg.lr,
      betas=[cfg.beta1, cfg.beta2],
      weight_decay=cfg.weight_decay,
    )

  elif cfg.optim == 'dion':
    from dion import Dion
    from .custom_muon import get_param_groups_muon
    param_groups = get_param_groups_muon(model, cfg)
    optimizer = Dion(
      param_groups,
      lr=cfg.lr,
      betas=[cfg.beta1, cfg.beta2],
      weight_decay=cfg.weight_decay,
    )
  
  else:
    raise NotImplementedError(f"Not implemented optim: {cfg.optim}.")
  
  return optimizer


def initialize_scheduler(optimizer, cfg):

  if cfg.scheduler is None:
    return None
  
  ## Number of warmup steps
  # either specified directly (int) or as a fraction of steps_budget (float)
  if getattr(cfg, 'warmup_steps', None) is not None:
    warmup_steps = cfg.warmup_steps if isinstance(cfg.warmup_steps, int) else int(cfg.warmup_steps * cfg.steps_budget)

  ## Number of cooldown steps
  # either specified directly (int) or as a fraction of steps_budget (float)
  if getattr(cfg, 'cooldown_steps', None) is not None:
    cooldown_steps = cfg.cooldown_steps if isinstance(cfg.cooldown_steps, int) else int(cfg.cooldown_steps * cfg.steps_budget)

  ##Final LR of the schedule
  # either specified directly via `lr_end` or as a fraction of top lr via `lr_end_pct`
  if getattr(cfg, 'lr_end', None) is not None or getattr(cfg, 'lr_end_pct', None) is not None:
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

  elif cfg.scheduler == "warmup_linear_decay":
    scheduler = WarmupLinearDecay(
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
def get_param_groups_default(model, cfg):
  """
    Create param groups for a Transformer model.
    Bias and normalization layers are excluded from weight decay.
  """

  # filter out parameters that do not require grad
  named_param_dict = {n: p for n,p in model.named_parameters() if p.requires_grad}
  param_names = named_param_dict.keys()

  # normaliz
  norm_param_names = [n for n in param_names if "norm" in n]
  norm_params = [p for n, p in named_param_dict.items() if n in norm_param_names]

  # bias
  bias_param_names = [n for n in param_names if "bias" in n and n not in norm_param_names]
  bias_params = [p for n, p in named_param_dict.items() if n in bias_param_names]

  # all the ohers params
  norm_and_bias_names = norm_param_names + bias_param_names
  other_param_names = [n for n in param_names if n not in norm_and_bias_names]
  other_params = [p for n, p in named_param_dict.items() if n in other_param_names]

  # assemble param groups
  param_groups = [
    dict(params=norm_params,    weight_decay=0.0,),
    dict(params=bias_params,    weight_decay=0.0,),
    dict(params=other_params,   weight_decay=cfg.weight_decay,),
  ]

  # # sanity check
  # print("bias_param_names:\n\t" + "\n\t".join(bias_param_names))
  # print("norm_param_names:\n\t" + "\n\t".join(norm_param_names))
  # print("other_param_names:\n\t" + "\n\t".join(other_param_names))

  return param_groups