
import torch
from torch import nn
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.checkpoint.state_dict import _init_optim_state


# TODO: replace this with dist.checkpoint.state_dict calls!


def _strip_orig_mod_keys(sd: dict):
    fixed = {}
    for k, v in sd.items():
        if k.startswith("_orig_mod."):
            fixed[k[len("_orig_mod."):]] = v
        else:
            fixed[k] = v
    return fixed


def get_full_model_state_dict(model, fsdp2):
  if not fsdp2:
    return model.state_dict()

  model_sd = model.state_dict() # sharded DTensors, sharded across devices
  full_sd = {} # full torch.Tensor, on cpu
  for n, p in model_sd.items(): # offload params to cpu one by one to avoid peaking GPU mem
    full_p = p.full_tensor()
    if torch.distributed.get_rank() == 0:
      full_sd[n] = full_p.cpu()
    else:
      del full_p
  
  return full_sd


def get_full_optimizer_state_dict(optim, fsdp2):
  if not fsdp2:
    return optim.state_dict()

  is_rank_zero = torch.distributed.get_rank() == 0
  sharded_sd = optim.state_dict()
  sharded_state = sharded_sd["state"]
  full_state = {}
  for group_id, sharded_group in sharded_state.items():
    group_state = {}
    for attr, sharded_tensor in sharded_group.items():
      if isinstance(sharded_tensor, DTensor): 
        # `exp_avg` in AdamW is `DTensor`
        full_tensor = sharded_tensor.full_tensor()
      else: 
        # `step` in AdamW is plain tensor
        full_tensor = sharded_tensor
      if is_rank_zero:
        group_state[attr] = full_tensor.cpu()
      else:
        del full_tensor
    if is_rank_zero:
      full_state[group_id] = group_state
    else:
      del group_state
  if is_rank_zero:
    return {"param_groups": sharded_sd["param_groups"], "state": full_state}
  else:
      return {}


def load_model(model, full_sd, fsdp2):
  if not fsdp2:
    model.load_state_dict(_strip_orig_mod_keys(full_sd))
    return

  # Build a sharded DTensor state_dict from a full state_dict:
  # use `distribute_tensor` to convert torch.Tensor into DTensor,
  # using the same placements and device mesh from model.state_dict(). 
  full_sd = _strip_orig_mod_keys(full_sd) # Tensors, device=`cpu`
  model_sd = model.state_dict() # sharded DTensors, device=`meta`
  sharded_sd = {}
  for n, p in full_sd.items():
    model_p = model_sd.get(n)
    sharded_tensor = distribute_tensor(p, model_p.device_mesh, model_p.placements)
    sharded_sd[n] = nn.Parameter(sharded_tensor)

  # Finally load sharded DTensor state dicts into model.
  # (`assign=True` since we cannot call `copy_` on meta tensor).
  model.load_state_dict(sharded_sd, assign=True)


def load_optim(optim, full_sd, fsdp2):
  if not fsdp2:
    optim.load_state_dict(full_sd)
    return

  _init_optim_state(optim)
  param_groups = optim.state_dict()["param_groups"]
  state = optim.state_dict()["state"]

  full_sd = full_sd
  full_param_groups = full_sd["param_groups"]
  full_state = full_sd["state"]

  for param_group, full_param_group in zip(param_groups, full_param_groups):
    for key, value in full_param_group.items():
      if key == "params":
        continue
      param_group[key] = value
    for pid, full_pid in zip(param_group["params"], full_param_group["params"]):
      if pid not in state:
        continue
      param_state = state[pid]
      full_param_state = full_state[full_pid]
      for attr, full_tensor in full_param_state.items():
        sharded_tensor = param_state[attr]
        if isinstance(sharded_tensor, DTensor):
          # exp_avg is DTensor
          param_state[attr] = distribute_tensor(
              full_tensor,
              sharded_tensor.device_mesh,
              sharded_tensor.placements,
          )
        else:
          # step is plain tensor
          param_state[attr] = full_tensor
  
  # Finally load sharded DTensor state dicts into model.
  optim.load_state_dict({
    "param_groups": param_groups, 
    "state": state
  })
  
