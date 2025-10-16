"""Torch-compilable llama-style RoPE"""
# thank you, @jonasgeiping!

from typing import Tuple

import torch


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, condense_ratio: int = 1):
  inv_freqs = 1.0 / (
    theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=torch.device("cpu")) / dim)
  )
  t = torch.arange(end, dtype=torch.float32, device=inv_freqs.device) / condense_ratio
  freqs = torch.outer(t, inv_freqs).float()
  return torch.stack(
    [torch.cos(freqs)[None, :, None, :], torch.sin(freqs)[None, :, None, :]], dim=4,
  )


@torch.compile
def apply_rotary_emb_complex_like(
  q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  # https://github.com/t-vi/lit-llama/blob/9e5eb8b1b376d8ae24e79278008b7190961062e3/lit_llama/model.py
  # cast because the reference does
  qk_r2 = torch.cat([q, k], dim=2).unflatten(dim=-1, sizes=(-1, 2)).float()
  rotated_qk_r2 = torch.stack(
    [
      qk_r2[..., 0] * freqs_cis[..., 0] - qk_r2[..., 1] * freqs_cis[..., 1],
      qk_r2[..., 1] * freqs_cis[..., 0] + qk_r2[..., 0] * freqs_cis[..., 1],
    ],
    -1,
  ).flatten(3)
  rotated_qk = rotated_qk_r2
  return torch.split(rotated_qk.type_as(q), q.shape[2], dim=2)  # type: ignore
