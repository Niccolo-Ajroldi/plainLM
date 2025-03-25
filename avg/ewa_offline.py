
from torch.nn.parallel import DistributedDataParallel as DDP
import torch

from avg.avg import AvgEngine


class EWAOffline(AvgEngine):
  """A module to average weigths."""

  def __init__(self, model, cfg, device, local_rank):
    # Init parent class. Here Model model the EMA!
    super().__init__(model, cfg, device, local_rank)

    # EMA-specific hyperparameters
    if not hasattr(cfg, 'ewa_beta'):
      raise KeyError('Missing EWA beta.')
    if cfg.ewa_beta < 0 or cfg.ewa_beta > 1:
      raise ValueError('Invalid EWA beta value.')
    self.beta = cfg.ewa_beta


  @torch.no_grad()
  def maybe_update_buffer(self, ckpt_path, step):
    """Update EMA from a ckpt_path."""
    beta = self.beta
    if step >= self.avg_start_step and step % self.avg_every_steps == 0:

      print(f"Update EMA")

      # Load checkpoint in CUDA
      ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
      state_dict = ckpt['state_dict']
      if isinstance(self.model, DDP):
        state_dict = {"module." + k: v for k, v in state_dict.items()}
      
      # Update EMA
      for n, p in self.model.named_parameters():
        p.mul_(beta).add_(state_dict[n], alpha=1-beta)


  @torch.no_grad()
  def prepare_for_eval(self):
    """Prepare for evaluation, by loading the avg into self.model."""
    pass

