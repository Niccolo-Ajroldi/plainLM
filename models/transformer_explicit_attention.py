"""Transformer++, a simple LLama-style Transformer, supporting RMSNorm, RoPE, GLU"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass
from typing import Tuple

from .components import RMSNorm, MLP, GLU, MLPReluSquared
from .embeddings import precompute_freqs_cis


@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int
    dim: int
    expand: float
    n_layers: int
    n_heads: int
    mlp: str = 'mlp'
    rmsorm_eps: float = 1e-6
    tie_embeddings: bool = False


MLP_CLASSES = {
    "mlp": MLP,
    "glu": GLU,
    "mlp_relu_sq": MLPReluSquared
}

def apply_rotary_emb_complex_like(
    q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor
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


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        
        self.w_qkv = nn.Linear(cfg.dim, 3*cfg.dim, bias=False)
        self.w_out = nn.Linear(cfg.dim, cfg.dim, bias=False)
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(cfg.seq_len, cfg.seq_len)).view(1, 1, cfg.seq_len, cfg.seq_len))
    
    def forward(self, x, freqs_cis):
        bsz, seqlen, d = x.shape # (bsz, seqlen, d)
        
        q, k, v = self.w_qkv(x).split(d, dim=2) # (bsz, seqlen, d)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim) # (bsz, seqlen, nh, h_dim)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim) # (bsz, seqlen, nh, h_dim)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim) # (bsz, seqlen, nh, h_dim)
        
        q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis) # (bsz, seqlen, nh, h_dim)
        
        q = q.transpose(1, 2) # (bsz, nh, seqlen, h_dim)
        k = k.transpose(1, 2) # (bsz, nh, seqlen, h_dim)
        v = v.transpose(1, 2) # (bsz, nh, seqlen, h_dim)
        
        # out = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (bsz, nh, seqlen, h_dim)

        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:seqlen,:seqlen] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = att @ v # (B, nh, seqlen, seqlen) x (B, nh, seqlen, h_dim) -> (B, nh, seqlen, h_dim)

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, d) # (bsz, seqlen, d)
        
        return self.w_out(out)

class Block(nn.Module):
    def __init__(self, layer_id: int, cfg: ModelConfig):
        super().__init__()
        self.attn = Attention(cfg)
        self.attn_norm = RMSNorm(cfg.dim, cfg.rmsorm_eps)
        self.mlp = MLP_CLASSES[cfg.mlp](dim=cfg.dim, hidden_dim=int(cfg.expand * cfg.dim))
        self.mlp_norm = RMSNorm(cfg.dim, cfg.rmsorm_eps)
        self.layer_id = layer_id
    
    def forward(self, x, freqs_cis):
        # x: (bsz, seqlen, dim)
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        x = x + self.mlp(self.mlp_norm(x))
        return x

class TransformerExplicit(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_layers = cfg.n_layers
        head_dim = cfg.dim // cfg.n_heads; assert cfg.dim % cfg.n_heads == 0
        
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([Block(idx, cfg) for idx in range(cfg.n_layers)])
        self.out_norm = RMSNorm(cfg.dim, cfg.rmsorm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        
        self.freqs_cis = precompute_freqs_cis(head_dim, cfg.seq_len, 500000)[0:cfg.seq_len]
        
        # init all weights, scale residual branches
        self.apply(self._init_weights)
        self._scale_residual_branches()
        
        if cfg.tie_embeddings:
            self.tie_weights()

    def forward(self, x):
        # x: (bsz, seqlen)
        x = self.embed_tokens(x) # (bsz, seqlen, dim)
        self.freqs_cis = self.freqs_cis.to(x.device)
        for layer in self.layers:
            x = layer(x, self.freqs_cis) # (bsz, seqlen, dim)
        return self.lm_head(self.out_norm(x)) # (bsz, seqlen, vocab_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_branches(self):
        for n, p in self.named_parameters():
            if n.endswith('fc2.weight'): # mlp/glu output layer
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.n_layers))
            if n.endswith('w_out.weight'): # attn output layer
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.n_layers))

    def tie_weights(self):
        self.lm_head.weight = self.embed_tokens.weight

    def count_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
            if not self.lm_head.weight is self.embed_tokens.weight:  # if no weight tying
                n_params -= self.lm_head.weight.numel()
        return n_params

