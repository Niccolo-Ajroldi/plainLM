import torch
import math

class AdamC(torch.optim.Optimizer):
  """
    AdamC from A. Defazio.
    Gradients increase near the end of training due to an unintended interaction 
    between weight decay, normalization layers, and the learning rate schedule.
    AdamC fixes this behavior.
    https://arxiv.org/abs/2506.02285
  """

  def __init__(self, params, lr=0.001, betas=(0.9, 0.999), weight_decay=0., eps=1e-8):
    if not 0.0 <= lr:
      raise ValueError(f"Invalid learning rate: {lr}")
    if not 0.0 <= eps:
      raise ValueError(f"Invalid epsilon value: {eps}")
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
    if not 0.0 <= weight_decay:
      raise ValueError(f"Invalid weight_decay value: {weight_decay}")

    defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
    super(AdamC, self).__init__(params, defaults)

    self.state['tot_steps'] = 0 # this is necessary for bias correction


  @torch.no_grad()
  def step(self, closure=None):

    self.state["tot_steps"] += 1
    step_t = self.state["tot_steps"]

    for group in self.param_groups:

      lr = group['lr']
      beta1, beta2 = group['betas']
      weight_decay = group['weight_decay']
      eps = group['eps']
      max_lr = self.defaults['lr'] if group['corrected_weight_decay'] else None # TODO: check!

      for p in group['params']:
        if p.grad is None:
            continue 
        p_state = self.state[p]

        ## Weight Decay
        # p.mul_(1 - lr * weight_decay)

        # (Corrected) Weight Decay
        wd_scale = lr if max_lr is None else lr ** 2 / max_lr
        p.mul_(1. - wd_scale * weight_decay)

        # m,v initialization
        if 'm' not in p_state:
          p_state['m'] = torch.zeros_like(p, device=p.device)
        if 'v' not in p_state:
          p_state['v'] = torch.zeros_like(p, device=p.device)
        m = p_state['m']
        v = p_state['v']

        # Decay 1st, 2nd moment
        m.mul_(beta1).add_(p.grad, alpha=(1.-beta1))
        v.mul_(beta2).addcmul_(p.grad, p.grad, value=(1.-beta2))

        # Bias correction
        bias_correction1 = 1 - beta1 ** step_t
        bias_correction2 = 1 - beta2 ** step_t
        bias_correction2_sqrt = math.sqrt(bias_correction2)

        # Update p
        denom = (v.sqrt() / bias_correction2_sqrt).add_(eps)
        p.addcdiv_(m, denom, value=-lr/bias_correction1)
        
        # Store per-step update on CPU
        p_state["u"] = (m / (denom * bias_correction1)).detach().to("cpu", non_blocking=True)
        p_state['lr'] = lr
