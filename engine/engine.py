
import torch

from torch import distributed as dist
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext

from models import get_param_groups
from optim import intialize_optimizer, initalize_scheduler


def _move_to_device(batch, seq_len, device_type, device):
  """Slice batch to get inputs and targets, and move them to device."""
  
  inputs = batch['input_ids'][:,:seq_len]
  targets = batch['input_ids'][:,1:(seq_len+1)]

  if device_type == 'cuda':
    # pin arrays allows to move them to GPU asynchronously (non_blocking=True)
    inputs = inputs.pin_memory().to(device, non_blocking=True)
    targets = targets.pin_memory().to(device, non_blocking=True)
  else:
    inputs, targets = inputs.to(device), targets.to(device)

  return inputs, targets


class TorchEngine(torch.nn.Module):
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
    dtype = cfg.dtype

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
    self.model_class_name = type(model).__name__ # ouch, ugly
    if cfg.torch_compile:
      print(f"Compiling the model...")
      self.model = torch.compile(self.model)

    # AMP
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    self.device_type = device_type

    # Grad scaler if training in fp16, if enabled=False, scaler is a no-op
    self.scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

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


  def step(self, batch):
    """Wraps a fwd pass, backwd pass, and optimization step."""
    
    self.model.train()
    
    self.micro_steps += 1
    self.accumulated_samples += 1

    inputs, targets = _move_to_device(batch, self.seq_len, self.device_type, self.device)

    # sync gradients at the last accumulation step
    if torch.distributed.is_initialized():
      self.model.require_backward_grad_sync = \
        (self.accumulated_samples == self.accumulation_steps)

    # forward pass with autocasting
    with self.ctx:
      logits = self.model(inputs)
      loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
      loss = loss / self.accumulation_steps

    # detach for logging (scale up to undo the division above)
    loss_val = loss.detach() * self.accumulation_steps
    if torch.isnan(loss_val):
      raise ValueError("Train loss is nan")

    # backward pass, with gradient scaling if training in fp16
    self.scaler.scale(loss).backward()

    if self.grad_clip:
      self.scaler.unscale_(self.optimizer)
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

    # step after accumulation
    if self.accumulated_samples == self.accumulation_steps:
      self.accumulated_samples = 0

      # step the optimizer, step the scaler if training in fp16
      self.scaler.step(self.optimizer)
      self.scaler.update()

      # flush the gradients
      self.optimizer.zero_grad(set_to_none=True) 
      
      # step the scheduler
      if self.scheduler:
        self.scheduler.step()
  
    return loss_val


  @torch.no_grad()
  def eval(self, validloader):
    """Evaluate model on a dataloader."""
    
    self.model.eval()
    
    # Compute loss on validloader
    losses = []
    for batch in validloader:
      inputs, targets = _move_to_device(batch, self.seq_len, self.device_type, self.device)
      logits = self.model(inputs)
      loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
      losses.append(loss.item())
    
    total_loss = sum(losses)

    # Reduce loss over processes
    if not dist.is_initialized():
      mean_loss = total_loss / len(validloader)
    else:
      total_loss_tensor = torch.tensor([total_loss], device=self.device)
      dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
      mean_loss = total_loss_tensor.item() / (dist.get_world_size() * len(validloader))
      
    return mean_loss
