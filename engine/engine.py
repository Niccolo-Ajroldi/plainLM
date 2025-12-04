from contextlib import nullcontext

import torch
from torch import distributed as dist
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP

from optim import intialize_optimizer, initialize_scheduler
from checkpoint_utils import load_checkpoint

MAX_ALLOWED_LOSS = 5.0
MIN_STEPS_BEFORE_CHECK = 5_000


def _move_to_device(batch, seq_len, device):
  """Slice batch to get inputs and targets, and move them to device."""
  inputs = batch["input_ids"][:, :seq_len]
  targets = batch["input_ids"][:, 1 : (seq_len + 1)]

  if device.type == 'cuda':
    # pin arrays allows to move them to GPU asynchronously (non_blocking=True)
    inputs = inputs.pin_memory().to(device, non_blocking=True)
    targets = targets.pin_memory().to(device, non_blocking=True)
  else:
    inputs, targets = inputs.to(device), targets.to(device)

  return inputs, targets


class TorchEngine(torch.nn.Module):
  """
  A module containing model, optimizer, scheduler.
  Wraps together a training step. Takes care of grad accumulation.
  """

  def __init__(self, model, cfg, device):
    super().__init__()

    self.steps = 0
    self.micro_steps = 0
    self.accumulated_samples = 0

    self.seq_len = cfg.seq_len
    self.accumulation_steps = cfg.grad_accumulation_steps
    self.grad_clip = cfg.grad_clip
    self.dtype = cfg.dtype
    self.intra_doc_masking = getattr(cfg, "intra_doc_masking", False)
    self.abort_on_bad_loss = getattr(cfg, "abort_on_bad_loss", False)

    device = torch.device(device)
    self.device = device

    if self.dtype == "float16":
      raise NotImplementedError("Gradient scaler not supported, please use bf16.")

    # Load model state dict
    ckpt = None
    if cfg.resume:
      ckpt = load_checkpoint(cfg)
      model.load_state_dict(ckpt["model_state"])
      self.steps = ckpt["step"]
      self.micro_steps = ckpt["step"] * cfg.grad_accumulation_steps

    # Move model to device and to DDP
    self.model = model.to(device)
    if dist.is_initialized():
      self.model = DDP(self.model, device_ids=[device.index])

    # Compile
    if cfg.torch_compile:
      print("Compiling the model...")
      self.model = torch.compile(self.model)

    # AMP
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[self.dtype]
    self.ctx = (
      nullcontext()
      if device.type == "cpu"
      else torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    )

    # Loss
    self.criterion = CrossEntropyLoss()

    # If we are running with NOS, we define the optimizers in main.
    if hasattr(cfg, 'optim'):
      # We might have multiple optimizers
      self.optimizers = intialize_optimizer(model, cfg)
      
      # Schedulers: one per optim (same tho)
      self.schedulers = {}
      for name, optim in self.optimizers.items():
        self.schedulers[name] = initialize_scheduler(optim, cfg)
      
    if cfg.resume:
      rank = dist.get_rank() if dist.is_initialized() else 0
      for name, optim in self.optimizers.items():
          if name == "zero1adamw":
              optim.load_state_dict(ckpt[f"optimizer_{name}_rank{rank}_state"])
          else:
              optim.load_state_dict(ckpt[f"optimizer_{name}_state"])          
      for name, scheduler in self.schedulers.items():
          scheduler.load_state_dict(ckpt[f"scheduler_{name}_state"])


  def step(self, batch):
    """Wraps a fwd pass, bwd pass, and optimization step."""

    self.model.train()

    self.micro_steps += 1
    self.accumulated_samples += 1

    inputs, targets = _move_to_device(batch, self.seq_len, self.device)

    # sync (reduce) gradients at the last accumulation step
    if dist.is_initialized():
      self.model.require_backward_grad_sync = self.accumulated_samples == self.accumulation_steps

    # forward pass with autocasting
    with self.ctx:
      output = self.model(inputs)
      logits = getattr(output, "logits", output)
      loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
      loss = loss / self.accumulation_steps

    # detach for logging (scale up to undo the division above)
    loss_val = loss.detach() * self.accumulation_steps
    if torch.isnan(loss_val):
      raise ValueError("Train loss is nan")
    if self.abort_on_bad_loss and loss_val > MAX_ALLOWED_LOSS and self.steps  > MIN_STEPS_BEFORE_CHECK:
      raise ValueError(f"Train loss {loss_val} exceeds {MAX_ALLOWED_LOSS} at step {self.steps}.")

    # backward pass
    loss.backward()

    # step after accumulation
    if self.accumulated_samples == self.accumulation_steps:
      self.accumulated_samples = 0
      self.steps +=1

      if self.grad_clip:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

      # step the optimizers, flush the grads
      for n, optim in self.optimizers.items():
        optim.step()
        optim.zero_grad(set_to_none=True)

      # step the schedulers
      for scheduler in self.schedulers.values():
        scheduler.step()

    return loss_val

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
        logits = getattr(output, "logits", output)
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
      total_loss = total_loss_tensor.item()
      num_batches = num_batches_tensor.item()

    # calculate average loss
    avg_loss = total_loss / num_batches

    return avg_loss
