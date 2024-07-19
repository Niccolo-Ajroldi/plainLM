
import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, dim: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-6)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # x: (bsz, T, dim)
        output = self._norm(x.float()).type_as(x) # (bsz, T, dim)
        return output * self.weight


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # x: (bsz, T, dim)
        return self.fc2(F.silu(self.fc1(x)))


class GLU(nn.Module):
    """fused GLU"""
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(dim, 2*hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # x: (bsz, T, dim)
        x, z = self.fc1(x).split(self.hidden_dim, dim=2)
        return self.fc2(F.silu(x) * z)

class MLPReluSquared(nn.Module):
    """MLP with ReLU squared"""
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # x: (bsz, T, dim)
        return self.fc2(F.relu(self.fc1(x)).pow(2))
