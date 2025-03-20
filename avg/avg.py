
import torch

from torch import distributed as dist
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext

from models import get_param_groups
from optim import intialize_optimizer, initalize_scheduler

from engine.engine import _move_to_device

class AvgEngine(torch.nn.Module):
  """
  A module containing model, optimizer, scheduler, grad scaler.
  Wraps together a training step. Takes care of grad accumulation.
  """
  def __init__(
      self,
      model,
      cfg,
      device,
      local_rank,
      ckpt,
      ):
    super().__init__()
    
    self.micro_steps = 0
    self.accumulated_samples = 0

    self.seq_len = cfg.seq_len
    self.accumulation_steps = cfg.grad_accumulation_steps
    self.grad_clip = cfg.grad_clip
    self.dtype = cfg.dtype

    self.device = device
    
    # Load model state dict
    if cfg.resume:
      model.load_state_dict(ckpt['state_dict'])
      self.micro_steps = ckpt['micro_step']

    # Move model to device and to DDP
    self.model = model.to(device)
    if torch.distributed.is_initialized():
      self.model = DDP(self.model, device_ids=[local_rank])

    # Compile
    if cfg.torch_compile:
      print(f"Compiling the model...")
      self.model = torch.compile(self.model)

    # AMP
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
    self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Grad scaler if training in fp16, if enabled=False, scaler is a no-op
    self.scaler = torch.amp.GradScaler(enabled=(self.dtype == 'float16'))

    # Loss
    self.criterion = CrossEntropyLoss()

    # Optimizer
    param_groups = get_param_groups(model, cfg.weight_decay)
    self.optimizer = intialize_optimizer(param_groups, cfg)
    self.scheduler = initalize_scheduler(self.optimizer, cfg)

    if cfg.resume:
      self.optimizer.load_state_dict(ckpt['optimizer'])
      self.scheduler.load_state_dict(ckpt['scheduler'])
      self.scaler.load_state_dict(ckpt['scaler'])

  def update_buffer():
    """Update the buffer with the current model parameters."""
    pass

  def eval():
    pass
