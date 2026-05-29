"""
GrafoPropagation v26-APEX — Output Heads
PoolingFusion · MultiSampleDropoutHead · MultiLabelHead

(c) 2025-2026 Claudio Fernandes. All rights reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import RMSNorm


class PoolingFusion(nn.Module):
    """
    Gated fusion of the thought representation (System-2 output)
    and the mean-pooled sequence representation.
    The gate is conditioned on the concatenation of both views.
    """

    def __init__(self, d: int):
        super().__init__()
        self.gate = nn.Linear(2 * d, d, bias=True)
        self.norm = RMSNorm(d)
        nn.init.xavier_uniform_(self.gate.weight, gain=0.5)
        nn.init.zeros_(self.gate.bias)

    def forward(self, think: torch.Tensor, seq: torch.Tensor,
                pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        think    : (B, D) thought representation (mean over K tokens).
        seq      : (B, T, D) sequence representation.
        pad_mask : (B, T) bool, True where token is padding.

        Returns
        -------
        (B, D) fused representation.
        """
        valid = (~pad_mask).to(seq.dtype).unsqueeze(-1)  # (B, T, 1)
        seq_avg = (seq * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        g = torch.sigmoid(self.gate(torch.cat([think, seq_avg], -1)))
        return self.norm(g * think + (1.0 - g) * seq_avg)


class MultiSampleDropoutHead(nn.Module):
    """
    Classification head with multi-sample dropout averaging at inference.
    During training a single forward pass; at eval the mean of k forward
    passes through the dropout mask gives an implicit ensemble.
    """

    def __init__(self, d: int, n_classes: int, dropout: float = 0.1, k: int = 5):
        super().__init__()
        self.k = k
        self.dp = dropout
        self.fc1 = nn.Linear(d, d)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(d, n_classes)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.02)
        nn.init.zeros_(self.fc2.bias)

    def _once(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.dropout(self.act(self.fc1(x)), p=self.dp, training=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._once(x)
        return torch.stack([self._once(x) for _ in range(self.k)]).mean(0)


class MultiLabelHead(nn.Module):
    """
    Multi-label binary classification head used for dictionary pre-training.
    Predicts which vocabulary tokens appear in the definition(s) of the
    input word → binary cross-entropy with positive-class reweighting.
    """

    def __init__(self, d: int, vocab_size: int):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(d, vocab_size)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
