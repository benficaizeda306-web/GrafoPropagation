"""
GrafoPropagation v26-APEX — von Mises-Fisher Attention
Dual-Scale Kappa · Asymmetric Q/K · Gated Output · Entropy Regularisation

(c) 2025-2026 Claudio Fernandes. All rights reserved.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import RMSNorm


class VonMisesFisherAttention(nn.Module):
    """
    vMF Attention: queries and keys live on the unit hypersphere;
    similarity is cosine (inner product of unit vectors) scaled by
    a learnable concentration κ rather than √d.

    Innovations over standard MHA
    ──────────────────────────────
    * Dual-scale κ: per-token local κ × global (pooled) κ gate.
    * Asymmetric Q/K projections for improved expressivity.
    * Gated output: element-wise sigmoid gate on the value aggregation.
    * Entropy regularisation: penalises overly peaked distributions.
    * Optional RoPE on normalised directions.
    """

    def __init__(
        self,
        d: int,
        n_heads: int,
        head_dim: int,
        dropout: float = 0.1,
        kappa_init: float = 4.0,
        use_kappa_weights: bool = True,
        rope=None,
        dual_scale: bool = True,
        asymmetric_qk: bool = True,
        kappa_max: float = 30.0,
        entropy_reg: float = 0.01,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.hd = head_dim
        self.dp = dropout
        self.eps = 1e-8
        self.ukw = use_kappa_weights
        self.rope = rope
        self.dual_scale = dual_scale
        self.asym_qk = asymmetric_qk
        self.kappa_max = kappa_max
        self.entropy_reg = entropy_reg
        self._sc = 1.0 / math.sqrt(head_dim)

        # Direction projections
        self.W_mu_q = nn.Linear(d, n_heads * head_dim, bias=False)
        self.W_mu_k = (nn.Linear(d, n_heads * head_dim, bias=False)
                       if asymmetric_qk else self.W_mu_q)
        self.Wv = nn.Linear(d, n_heads * head_dim, bias=False)
        self.Wo = nn.Linear(n_heads * head_dim, d, bias=False)
        self.W_gate = nn.Linear(d, n_heads * head_dim, bias=False)

        # Concentration projections
        self.W_kappa_local = nn.Linear(d, n_heads, bias=True)
        if dual_scale:
            self.W_kappa_global = nn.Linear(d, n_heads, bias=True)

        # Initialisations
        g = 1.0 / math.sqrt(2)
        for w in [self.W_mu_q, self.Wv, self.Wo, self.W_gate]:
            nn.init.xavier_uniform_(w.weight, gain=g)
        if asymmetric_qk:
            nn.init.xavier_uniform_(self.W_mu_k.weight, gain=g)
        nn.init.xavier_uniform_(self.W_kappa_local.weight, gain=0.1)
        nn.init.constant_(self.W_kappa_local.bias, math.log(max(kappa_init - 1.0, 1e-4)))
        if dual_scale:
            nn.init.xavier_uniform_(self.W_kappa_global.weight, gain=0.1)
            nn.init.zeros_(self.W_kappa_global.bias)

        self.tau = nn.Parameter(torch.ones(n_heads) * 2.0)
        self.bias_q = nn.Parameter(torch.zeros(n_heads))

    # ── κ computation ─────────────────────────────────────────────────

    def get_kappa(self, x: torch.Tensor) -> torch.Tensor:
        kappa_loc = torch.clamp(
            F.softplus(self.W_kappa_local(x)) + 1e-4,
            max=self.kappa_max,
        )  # (B, T, H)
        if self.dual_scale:
            x_pool = x.mean(dim=1, keepdim=True)  # (B, 1, D)
            kappa_glo = torch.sigmoid(self.W_kappa_global(x_pool))
            return kappa_loc * kappa_glo
        return kappa_loc

    # ── Entropy regularisation ────────────────────────────────────────

    def entropy_loss(self, attn: torch.Tensor) -> torch.Tensor:
        H = -(attn * (attn + self.eps).log()).sum(-1).mean()
        return F.relu(math.log(2.0) - H)

    # ── Forward ───────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor,
                fmask: torch.Tensor = None,
                tmask: torch.Tensor = None):
        """
        Parameters
        ----------
        x     : (B, T, D)
        fmask : (B, T) float mask; −∞ for padding positions.
        tmask : (B, T, T) temporal modulation from TemporalTransitionEmbedding.

        Returns
        -------
        (out, ent_loss) : (B, T, D), scalar
        """
        B, T, D = x.shape

        mu_q = F.normalize(
            self.W_mu_q(x).view(B, T, self.n_heads, self.hd), p=2, dim=-1, eps=self.eps,
        )
        mu_k = F.normalize(
            self.W_mu_k(x).view(B, T, self.n_heads, self.hd), p=2, dim=-1, eps=self.eps,
        )
        if self.rope is not None:
            mu_q = self.rope(mu_q)
            mu_k = self.rope(mu_k)

        kappa = self.get_kappa(x)  # (B, T, H)
        mu_q_h = mu_q.permute(0, 2, 1, 3)  # (B, H, T, D)
        mu_k_h = mu_k.permute(0, 2, 1, 3)
        S = torch.matmul(mu_q_h, mu_k_h.transpose(-2, -1))  # (B, H, T, T)

        if self.ukw:
            kh = kappa.permute(0, 2, 1)  # (B, H, T)
            S = torch.sqrt(kh.unsqueeze(-1) * kh.unsqueeze(-2) + self.eps) * S

        scores = (
            self.tau.view(1, self.n_heads, 1, 1) * (S * self._sc)
            + self.bias_q.view(1, self.n_heads, 1, 1)
        )

        # Temporal modulation
        if tmask is not None:
            pl = T - tmask.shape[-1]
            if pl > 0:
                te = torch.zeros(B, T, T, device=x.device, dtype=x.dtype)
                te[:, pl:, pl:] = tmask
                scores = scores + te.unsqueeze(1)
            else:
                scores = scores + tmask.unsqueeze(1)

        # Padding mask
        if fmask is not None:
            pl = T - fmask.shape[-1]
            if pl > 0:
                fe = torch.zeros(B, T, device=x.device, dtype=x.dtype)
                fe[:, pl:] = fmask
                scores = scores + fe[:, None, None, :]
            else:
                scores = scores + fmask[:, None, None, :]

        attn = F.softmax(scores, dim=-1)
        ent_loss = self.entropy_loss(attn) * self.entropy_reg
        attn = F.dropout(attn, p=self.dp if self.training else 0.0,
                         training=self.training)

        v = self.Wv(x).view(B, T, self.n_heads, self.hd).permute(0, 2, 1, 3)
        av = torch.matmul(attn, v)  # (B, H, T, D)
        gate = torch.sigmoid(
            self.W_gate(x).view(B, T, self.n_heads, self.hd).permute(0, 2, 1, 3),
        )
        out = (gate * av).permute(0, 2, 1, 3).reshape(B, T, self.n_heads * self.hd)
        return self.Wo(out), ent_loss
