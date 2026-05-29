"""
GrafoPropagation v26-APEX — Transformer Building Blocks
LocalConvMix · SwiGLU · TransformerBlock (stochastic depth)

(c) 2025-2026 Claudio Fernandes. All rights reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import RMSNorm
from .attention import VonMisesFisherAttention


class LocalConvMix(nn.Module):
    """
    Local convolutional token mixer: depthwise separable convolution
    over the token dimension, providing local context before global attention.
    """

    def __init__(self, d: int, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.norm = RMSNorm(d)
        self.dw = nn.Conv1d(d, d, kernel, padding=(kernel - 1) // 2, groups=d, bias=False)
        self.pw = nn.Conv1d(d, d, 1, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.drop = nn.Dropout(dropout)
        nn.init.kaiming_normal_(self.dw.weight, nonlinearity="linear")
        nn.init.xavier_uniform_(self.pw.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).transpose(1, 2).contiguous()  # (B, D, T)
        return x + self.drop(self.act(self.pw(self.dw(h))).transpose(1, 2).contiguous())


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network (Noam Shazeer, 2020):
    out = Dropout( W_d( SiLU(gate) * linear ) )
    Packed as a single matrix for efficiency.
    """

    def __init__(self, d: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.W_gu = nn.Linear(d, 2 * d_ff, bias=False)
        self.Wd = nn.Linear(d_ff, d, bias=False)
        self.drop = nn.Dropout(dropout)
        nn.init.kaiming_normal_(self.W_gu.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.Wd.weight, gain=1.0 / (12 ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = self.W_gu(x).chunk(2, dim=-1)
        return self.drop(self.Wd(F.silu(g) * u))


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block:
        x ← x + StochDrop( vMF-Attn( RMSNorm(x) ) )
        x ← x + StochDrop( SwiGLU(  RMSNorm(x) ) )

    Stochastic depth is linearly scheduled per layer.
    """

    def __init__(
        self,
        d: int,
        n_heads: int,
        head_dim: int,
        d_ff: int,
        dropout: float = 0.1,
        stoch_depth: float = 0.0,
        kappa_init: float = 4.0,
        use_kappa_weights: bool = True,
        rope=None,
        dual_scale: bool = True,
        asymmetric_qk: bool = True,
        kappa_max: float = 30.0,
        entropy_reg: float = 0.01,
    ):
        super().__init__()
        self.sd = stoch_depth
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.attn = VonMisesFisherAttention(
            d, n_heads, head_dim, dropout,
            kappa_init, use_kappa_weights, rope,
            dual_scale=dual_scale, asymmetric_qk=asymmetric_qk,
            kappa_max=kappa_max, entropy_reg=entropy_reg,
        )
        self.ffn = SwiGLU(d, d_ff, dropout)

    def _stochastic_drop(self, residual: torch.Tensor) -> torch.Tensor:
        if not self.training or self.sd == 0.0:
            return residual
        keep = (torch.rand(residual.shape[0], 1, 1, device=residual.device) > self.sd).float()
        return residual * keep / (1.0 - self.sd)

    def forward(self, x: torch.Tensor,
                fmask=None, tmask=None):
        attn_out, ent_loss = self.attn(self.norm1(x), fmask, tmask)
        x = x + self._stochastic_drop(attn_out)
        x = x + self._stochastic_drop(self.ffn(self.norm2(x)))
        return x, ent_loss
