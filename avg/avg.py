"""Parent abstract class for averaging weights offline."""

import abc
import math
from contextlib import nullcontext

import torch
from torch import distributed as dist
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP

from engine.engine import _move_to_device


class AvgEngine(torch.nn.Module):
  """A parent abstract module to average weigths offline."""

  def __init__(self, model, cfg, device, local_rank):
    super().__init__()

    self.micro_steps = 0

    self.seq_len = cfg.seq_len
    self.dtype = cfg.dtype
    self.device = device
    self.torch_compile = cfg.torch_compile

    # AMP
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
    self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Loss
    self.criterion = CrossEntropyLoss()
  
    # DDP Model on CUDA
    self.model = model.to(device)
    if torch.distributed.is_initialized():
      self.model = DDP(self.model, device_ids=[local_rank])

    # Compile
    if cfg.torch_compile:
      print(f"Compiling the model...")
      self.model = torch.compile(self.model)

    # Avg start
    has_pct = getattr(cfg, "avg_burnin_pct", None) is not None
    has_step = getattr(cfg, "avg_start_step", None) is not None
    if not has_pct and not has_step:
      raise ValueError("Missing hyperparameter: avg_burnin_pct or avg_start_step")
    if has_step and has_pct:
      raise ValueError("Both avg_burnin_pct and avg_start_step are defined")
    if has_step:
      self.avg_start_step = int(cfg.avg_start_step)
    else:
      self.avg_every_steps = cfg.steps_budget * getattr(cfg, "avg_burnin_pct", 0.0)

    # Avg freq
    has_pct = getattr(cfg, "avg_every_pct", None) is not None
    has_step = getattr(cfg, "avg_every_steps", None) is not None
    if not has_pct and not has_step:
      raise ValueError("Missing hyperparameter: lawa_every_steps or lawa_every_pct")
    if has_step and has_pct:
      raise ValueError("Both lawa_every_steps and lawa_every_pct are defined")
    if has_step:
      self.avg_every_steps = int(cfg.avg_every_steps)
    else:
      self.avg_every_steps = math.ceil(cfg.steps_budget * cfg.avg_every_pct)


  @abc.abstractmethod
  def maybe_update_buffer(self):
    pass

  @abc.abstractmethod
  def prepare_for_eval(self):
    pass

  @torch.no_grad()
  def eval(self, dataloader):
    """Evaluate model on a dataloader."""

    self.model.eval()

    # Compute loss on dataloader
    total_loss = 0.0
    num_batches = 0
    for batch in dataloader:
      inputs, targets = _move_to_device(batch, self.seq_len, self.device)
      with self.ctx:
        output = self.model(inputs)
        logits = getattr(output, 'logits', output)
        loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

      if torch.isnan(loss) or loss is None:
        raise ValueError("Validation loss is nan")

      total_loss += loss.item()
      num_batches += 1

    # reduce loss across processes
    if dist.is_initialized():
      total_loss_tensor = torch.tensor([total_loss], device=self.device)
      num_batches_tensor = torch.tensor([num_batches], device=self.device, dtype=torch.int)
      dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
      dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
      total_loss = total_loss_tensor.item() / dist.get_world_size()
      num_batches = num_batches_tensor.item() // dist.get_world_size() # superflous if drop_last=True in dataloader

    # calculate average loss
    avg_loss = total_loss / num_batches

    return avg_loss
