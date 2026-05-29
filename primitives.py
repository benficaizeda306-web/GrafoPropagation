"""
GrafoPropagation v26-APEX — Primitive Layers
RMSNorm · Truncated-Normal Init · Character Embedding Builder

(c) 2025-2026 Claudio Fernandes. All rights reserved.
"""

import math
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-Mean-Square normalisation (no mean-subtraction bias)."""

    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        rms = xf.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (xf * rms * self.scale.float()).to(x.dtype)


def trunc_normal_(tensor: torch.Tensor,
                  mean: float = 0.0, std: float = 0.02,
                  a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    """In-place truncated-normal init (clips at ±2σ by default)."""
    with torch.no_grad():
        nn.init.normal_(tensor, mean, std)
        tensor.clamp_(mean + a * std, mean + b * std)
        for _ in range(10):
            mask = (tensor < mean + a * std) | (tensor > mean + b * std)
            if not mask.any():
                break
            nn.init.normal_(tensor[mask], mean, std)
            tensor[mask].clamp_(mean + a * std, mean + b * std)
    return tensor


def build_character_embeddings(char_vocab: list, dim: int,
                               device: torch.device) -> torch.Tensor:
    """
    Construct char-level embeddings via Fibonacci-sphere initialisation for
    alphabetic characters; noise-augmented unit vector for punctuation;
    zero vector for whitespace.

    Returns
    -------
    torch.Tensor of shape (len(char_vocab), dim)
    """
    letters = set("abcdefghijklmnopqrstuvwxyz") | set("0123456789")
    punct = set(char_vocab) - letters - {" "}
    n = len(char_vocab)
    emb = torch.empty(n, dim, device=device)

    letter_indices = [i for i, ch in enumerate(char_vocab) if ch in letters]
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
    for idx, i in enumerate(letter_indices):
        y = 1.0 - (idx / float(max(len(letter_indices) - 1, 1))) * 2.0
        r = math.sqrt(max(0.0, 1.0 - y * y))
        base = torch.tensor(
            [math.cos(phi * idx) * r, y, math.sin(phi * idx) * r],
            device=device,
        )
        emb[i] = (
            torch.cat([base, torch.randn(dim - 3, device=device) * 0.05])
            if dim > 3 else base
        )

    for i in [i for i, ch in enumerate(char_vocab) if ch in punct]:
        base = torch.cat([
            torch.randn(2, device=device) * 0.1,
            torch.tensor([1.0], device=device),
        ])
        emb[i] = (
            torch.cat([base, torch.randn(dim - 3, device=device) * 0.05])
            if dim > 3 else base
        )

    if " " in char_vocab:
        emb[char_vocab.index(" ")].zero_()

    return emb
