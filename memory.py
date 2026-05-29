"""
GrafoPropagation v26-APEX — Memory & Graph Connectivity
GlobalWorkspaceMemory · DynamicGrafoConnect

(c) 2025-2026 Claudio Fernandes. All rights reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import RMSNorm


# ─────────────────────────────────────────────────────────────────────────────
# Global Workspace Memory
# ─────────────────────────────────────────────────────────────────────────────

class GlobalWorkspaceMemory(nn.Module):
    """
    Learnable memory bank slots prepended to the token sequence.
    Inspired by Global Workspace Theory: a small set of shared slots
    that broadcast and compress information across all layers.
    """

    def __init__(self, slots: int, d: int):
        super().__init__()
        self.slots = slots
        self.bank = nn.Parameter(torch.randn(1, slots, d) * 0.02)
        self.norm = RMSNorm(d)

    def expand_context(self, x: torch.Tensor, B: int) -> torch.Tensor:
        """Prepend memory slots to the input sequence."""
        return torch.cat([self.norm(self.bank).expand(B, -1, -1), x], dim=1)

    def extract_and_update(self, ctx: torch.Tensor, B: int) -> torch.Tensor:
        """
        Remove memory slots from the sequence and slowly update them
        with the mean of the slot representations (EMA-style).
        """
        new = ctx[:, :self.slots].mean(0, keepdim=True)
        if self.training:
            self.bank.data.lerp_(new.detach().data, 0.01)
        return ctx[:, self.slots:]


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic GrafoConnect
# ─────────────────────────────────────────────────────────────────────────────

class DynamicGrafoConnect(nn.Module):
    """
    Learned cross-layer skip-connection graph.

    For each target layer ℓ, computes a weighted sum of CLS representations
    from all previous layers, modulated by the temporal curvature signal.
    This allows the model to dynamically route information across non-adjacent
    layers, forming a learned computation graph.
    """

    def __init__(self, n_layers: int, d: int):
        super().__init__()
        self.n_layers = n_layers
        # Adjacency weight matrix: A[tgt, src] = importance of layer src → tgt
        self.A = nn.Parameter(torch.zeros(n_layers, n_layers))
        # Per-source linear projections
        self.src = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_layers)])
        for p in self.src:
            nn.init.eye_(p.weight)
        # Gating mechanism
        self.gate = nn.Linear(2 * d, d, bias=True)
        nn.init.zeros_(self.gate.bias)
        self.norm = RMSNorm(d)
        # Curvature modulation
        self.time_mod = nn.Linear(1, n_layers, bias=False)

    def forward(self, hist: list, tgt: int, curve: torch.Tensor):
        """
        Parameters
        ----------
        hist  : list of (B, D) tensors — CLS representations from previous layers.
        tgt   : int — target layer index.
        curve : (B,) — temporal curvature scalar per sample.

        Returns
        -------
        (B, D) aggregated skip-connection delta, or None if no history.
        """
        n = len(hist)
        if n == 0:
            return None

        # Modulate adjacency by curvature
        impact = torch.sigmoid(self.time_mod(curve.unsqueeze(-1)))  # (B, L)
        dyn_A = self.A[tgt, :n].unsqueeze(0) * (1.0 + impact[:, :n])  # (B, n)
        w = F.softmax(dyn_A, dim=-1)  # (B, n)

        # Weighted aggregation of projected source representations
        agg = sum(self.src[i](hist[i]) * w[:, i].unsqueeze(-1) for i in range(n))

        # Gate: blend between last source and aggregated
        g = torch.sigmoid(self.gate(torch.cat([hist[-1], agg], dim=-1)))
        return self.norm(g * agg)
