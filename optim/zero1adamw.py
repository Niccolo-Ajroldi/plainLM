""""
ZeRO-1 AdamW implementation.
"""

from typing import List
import torch
import torch.distributed as dist
from torch.futures import Future


class ZeRO1AdamW(torch.optim.Optimizer):
    """
    Zero1 AdamW.
    
    Supports AdamC's correction of weight decay (https://arxiv.org/abs/2506.02285).

    Parameters are replicated across devices.
    Expect allreduced gradients.
    States are sharded along the first dimension.
    For each param `p`:
    - Each device updates a chunk of `p` (`chunk_size = p.size(0) // self.world_size`).
    - Each device stores optimizer states for such chunk only.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3, 
        betas: tuple[float, float] = (0.9, 0.999), 
        weight_decay: float = 0.01,
        adamc_wd=False,
        eps: float = 1e-8, 
    ):
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
        if not isinstance(adamc_wd, bool):
            raise ValueError(f"adamc_wd must be a boolean, got {type(adamc_wd)}")
    
        if not dist.is_initialized():
          raise ValueError('Using ZeRO1AdamW in a non-distributed run.')
    
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            adamc_wd=adamc_wd,
            eps=eps,
        )
        super().__init__(params, defaults)

        # Initalize state.
        for group in self.param_groups:
            for p in group['params']:
                chunk_size = p.size(0) // self.world_size
                exp_avg = torch.zeros_like(p[:chunk_size])
                exp_avg_sq = torch.zeros_like(p[:chunk_size])
                self.state[p] = dict(step=0, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq)


    @torch.compile()
    @torch.no_grad()
    def step(self):
        rank = self.rank
        world_size = self.world_size
        all_gather_futures: List[Future] = []

        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            beta1, beta2 = group['betas']
            eps = group['eps']
            max_lr = self.defaults['lr'] if group['adamc_wd'] else None

            for p in group['params']:
                if p.grad is None:
                    continue

                # Retrieve p slice and g slice
                chunk_size = p.shape[0] // world_size
                start = rank * chunk_size
                end = (rank + 1) * chunk_size
                p_slice = p[start:end]
                g_slice = p.grad[start:end]

                state = self.state[p]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # (Corrected) Weight Decay
                if wd != 0:
                    wd_scale = lr if max_lr is None else lr ** 2 / max_lr
                    p_slice.mul_(1 - wd_scale * wd)

                # Update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t

                # Update params and allgather
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (bias2 ** 0.5 / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(update, alpha=-1.0)

                all_gather_futures.append(
                    dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                )

        torch.futures.wait_all(all_gather_futures)
