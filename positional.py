"""
GrafoPropagation v26-APEX — Positional & Temporal Modules
RoPERotator · TemporalTransitionEmbedding (Log-Map + Parallel Transport)

(c) 2025-2026 Claudio Fernandes. All rights reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import RMSNorm


# ─────────────────────────────────────────────────────────────────────────────
# Rotary Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class RoPERotator(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Applies rotation in complex-paired sub-spaces to normalised direction
    vectors, making angular distances position-relative.
    """

    def __init__(self, head_dim: int, max_len: int = 512, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        inv = 1.0 / (
            base ** (torch.arange(0, head_dim // 2, dtype=torch.float32) / (head_dim // 2))
        )
        self.register_buffer("inv_freq", inv)
        self.head_dim = head_dim
        self._build(max_len)

    def _build(self, n: int):
        t = torch.arange(n, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        emb = torch.cat([torch.outer(t, self.inv_freq)] * 2, dim=-1)
        self.register_buffer("cos_c", emb.cos(), persistent=False)
        self.register_buffer("sin_c", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        mu : (B, T, H, D) normalised direction vectors.

        Returns
        -------
        Rotated & re-normalised (B, T, H, D).
        """
        B, T, H, D = mu.shape
        if T > self.cos_c.shape[0]:
            self._build(T * 2)
        c = self.cos_c[:T].to(mu.dtype).view(1, T, 1, D)
        s = self.sin_c[:T].to(mu.dtype).view(1, T, 1, D)
        return F.normalize(mu * c + self._rotate_half(mu) * s, p=2, dim=-1, eps=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Transition Embedding
# ─────────────────────────────────────────────────────────────────────────────

class TemporalTransitionEmbedding(nn.Module):
    """
    Riemannian temporal embedding that:
      1. Projects tokens onto the unit hypersphere.
      2. Computes geodesic velocities via the spherical Log-Map.
      3. Transports velocities forward via Parallel Transport.
      4. Optionally modulates attention weights with temporal decay.
      5. Estimates local curvature via second-order angle differences.

    Returns
    -------
    (temb, tmask, curve)
        temb  : (B, T, D) temporal additive embeddings
        tmask : (B, T, T) attention modulation mask  (or None)
        curve : (B,) scalar curvature per sample
    """

    def __init__(self, d: int, n_feat: int = 6,
                 ratio: float = 2.0, modulate: bool = True):
        super().__init__()
        self.modulate = modulate
        hidden = int(d * ratio)
        self.proj = nn.Sequential(
            nn.Linear(d, hidden), RMSNorm(hidden),
            nn.GELU(approximate="tanh"), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), RMSNorm(hidden),
            nn.GELU(approximate="tanh"), nn.Dropout(0.1),
            nn.Linear(hidden, d),
        )
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if modulate:
            self.temporal_decay = nn.Parameter(torch.tensor(1.0))
            self.temporal_bias = nn.Parameter(torch.tensor(0.0))
        self.log_curv = nn.Parameter(torch.zeros(1))
        self.eps = 1e-8

    # ── Riemannian helpers ────────────────────────────────────────────

    def _log_map(self, base: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Spherical Log-Map: tangent vector at `base` pointing toward `target`."""
        dot = (base * target).sum(-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(dot)
        perp = target - dot * base
        sin_theta = torch.sin(theta).clamp(min=1e-6)
        factor = torch.where(
            theta < 1e-4,
            1.0 + theta.pow(2) / 6.0 + theta.pow(4) / 120.0,
            theta / sin_theta,
        )
        return factor * perp

    def _parallel_transport(self, v: torch.Tensor,
                            src: torch.Tensor,
                            dst: torch.Tensor) -> torch.Tensor:
        """Schild's-ladder-style parallel transport along the geodesic src→dst."""
        log_d = self._log_map(src, dst)
        norm2 = log_d.pow(2).sum(-1, keepdim=True).clamp(min=1e-8)
        proj = (v * log_d).sum(-1, keepdim=True) / norm2
        return v - 2.0 * proj * log_d

    # ── Forward ───────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        xn = F.normalize(x.float(), p=2, dim=-1, eps=self.eps)

        # Geodesic arc lengths between consecutive tokens
        cos_t = (xn[:, :-1] * xn[:, 1:]).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(cos_t)  # (B, T-1)

        # Log-map velocities
        v_raw = self._log_map(xn[:, :-1], xn[:, 1:])  # (B, T-1, D)

        # Parallel-transport chain
        transported = [torch.zeros(B, D, device=x.device, dtype=v_raw.dtype)]
        v_cur = v_raw[:, 0]
        transported.append(v_cur)
        for t in range(1, T - 1):
            v_cur = self._parallel_transport(v_cur, xn[:, t - 1], xn[:, t])
            transported.append(v_cur)
        v_vectors = torch.stack(transported, dim=1)  # (B, T, D)

        # Scale by learnable curvature
        curv_scale = torch.exp(self.log_curv).clamp(0.1, 10.0)
        v_vectors = v_vectors * curv_scale

        # Project to embedding space
        t_emb = self.proj(v_vectors)  # (B, T, D)

        # Temporal attention mask (optional)
        theta_padded = F.pad(theta, (1, 0), value=0.0)  # (B, T)
        t_mask = None
        if self.modulate:
            cs = torch.cumsum(theta_padded, -1)
            dist = (cs.unsqueeze(-1) - cs.unsqueeze(-2)).abs()
            dec = torch.clamp(self.temporal_decay, min=0.0)
            t_mask = (torch.exp(-dec * dist) + self.temporal_bias).to(x.dtype)

        # Scalar curvature estimate (Menger-style)
        omega = theta_padded[:, 1:] - theta_padded[:, :-1]
        omega = F.pad(omega, (1, 0), value=0.0)
        raw_k = (omega.abs() / (theta_padded * theta_padded + self.eps)).clamp(0, 20)
        curve = raw_k.mean(-1)  # (B,)

        return t_emb, t_mask, curve
