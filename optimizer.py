"""
GrafoPropagation v26-APEX — Optimiser Utilities
EMA · Lookahead · AWP · GradCentralization · WarmupCosineLR

(c) 2025-2026 Claudio Fernandes. All rights reserved.
"""

import copy
import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Exponential Moving Average
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    """
    Shadow model maintained as an EMA of training weights.
    Evaluated with `ema.shadow` for better generalisation at inference.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.lerp_(m.float(), 1.0 - self.decay)
        for sb, mb in zip(self.shadow.buffers(), model.buffers()):
            sb.copy_(mb)


# ─────────────────────────────────────────────────────────────────────────────
# Lookahead
# ─────────────────────────────────────────────────────────────────────────────

class Lookahead(torch.optim.Optimizer):
    """
    Lookahead optimiser wrapper (Zhang et al., 2019).
    Slow weights are updated every k fast steps:
        slow ← slow + α · (fast - slow)
    """

    def __init__(self, base: torch.optim.Optimizer, k: int = 6, alpha: float = 0.5):
        self._b = base
        self.k = k
        self.alpha = alpha
        self._steps = 0
        self._slow = {}
        # Mirror base attributes for compatibility
        self.param_groups = base.param_groups
        self.defaults = getattr(base, "defaults", {})

    @property
    def state(self):
        return self._b.state

    def zero_grad(self, set_to_none: bool = True):
        self._b.zero_grad(set_to_none=set_to_none)

    def _ensure_slow(self):
        if self._slow:
            return
        for g in self.param_groups:
            for p in g["params"]:
                self._slow[id(p)] = p.data.clone().detach()

    def step(self, closure=None):
        loss = self._b.step(closure)
        self._steps += 1
        self._ensure_slow()
        if self._steps % self.k == 0:
            for g in self.param_groups:
                for p in g["params"]:
                    s = self._slow[id(p)]
                    s.add_(self.alpha * (p.data - s))
                    p.data.copy_(s)
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial Weight Perturbation (AWP)
# ─────────────────────────────────────────────────────────────────────────────

class AWP:
    """
    Adversarial Weight Perturbation (Shafahi et al. / Wu et al., 2020).
    Perturbs model weights in the direction of the gradient and
    recomputes the loss, producing a smoother loss landscape.
    """

    def __init__(self, model: nn.Module, scaler, eps: float = 0.005, lr: float = 0.01):
        self.model = model
        self.scaler = scaler
        self.eps = eps
        self.lr = lr
        self._backup = {}
        self._on = False

    def perturb(self):
        if self._on:
            return
        sc = self.scaler.get_scale() if self.scaler.is_enabled() else 1.0
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                g = p.grad.float() / (sc + 1e-8)
                gn = g.norm()
                if gn > 0 and torch.isfinite(gn):
                    self._backup[n] = p.data.clone()
                    delta = (self.lr * g / (gn + 1e-8)).clamp_(-self.eps, self.eps)
                    p.data.add_(delta.to(p.dtype))
        self._on = True

    def restore(self):
        for n, p in self.model.named_parameters():
            if n in self._backup:
                p.data.copy_(self._backup[n])
        self._backup.clear()
        self._on = False


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Centralisation
# ─────────────────────────────────────────────────────────────────────────────

def _gc_hook(g: torch.Tensor) -> torch.Tensor:
    """Per-parameter gradient centralisation hook."""
    return g - g.mean(tuple(range(1, g.dim())), keepdim=True) if g.dim() > 1 else g


def register_gc(model: nn.Module) -> list:
    """
    Register gradient centralisation hooks on all non-embedding parameters
    with ndim > 1.

    Returns
    -------
    list of hook handles (call handle.remove() to de-register).
    """
    return [
        p.register_hook(_gc_hook)
        for n, p in model.named_parameters()
        if p.requires_grad and p.dim() > 1 and "embed" not in n
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Warmup-Cosine LR Schedule
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosineLR:
    """
    Cosine decay schedule with linear warmup.

    factor(step) ∈ [min_lr_frac, 1.0]
    Multiply by base_lr to get effective LR.
    """

    def __init__(self, total_steps: int, warmup_frac: float, min_lr_frac: float):
        self.T = max(total_steps, 1)
        self.W = max(int(warmup_frac * total_steps), 1)
        self.mf = min_lr_frac

    def factor(self, step: int) -> float:
        if step < self.W:
            return step / self.W
        p = min(max((step - self.W) / max(self.T - self.W, 1), 0.0), 1.0)
        return self.mf + (1.0 - self.mf) * 0.5 * (1.0 + math.cos(math.pi * p))
