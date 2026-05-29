"""
GrafoPropagation v26-APEX — Configuration
==========================================

Fully configurable architecture via dataclass with ``from_dict()`` /
``to_dict()`` / ``update()`` for easy scaling experiments.

(c) 2025-2026 Claudio Fernandes. All rights reserved.
"""

from __future__ import annotations

import math
import random
import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

import numpy as np
import torch


@dataclass
class CFG:
    """
    Master configuration for GrafoPropagation v26-APEX.

    Every architectural and training hyper-parameter is exposed here so that
    users can freely scale the model (e.g. d_model=256, n_layers=6) for
    capacity testing.

    Usage
    -----
    >>> cfg = CFG()                        # defaults (~990k params)
    >>> cfg = CFG(d_model=128, n_layers=4) # scale up
    >>> cfg = CFG.from_dict({"d_model": 256, "n_heads": 8})
    >>> cfg.to_dict()                      # serialise
    """

    # ── Meta ──────────────────────────────────────────────────────────
    VERSION: str = "v26-APEX"
    seed: int = 42

    # ── Device (computed at init-time, excluded from serialisation) ───
    device_str: str = "auto"       # "auto" | "cuda" | "cpu"
    amp_dtype_str: str = "auto"    # "auto" | "bfloat16" | "float16" | "float32"

    # ── Dataset ────────────────────────────────────────────────────────
    dataset_name: str = "ag_news"
    max_train: int = 120_000
    max_val: int = 5_000
    n_classes: int = 4
    tokenizer_vocab_size: int = 10_000
    tokenizer_path: str = "/tmp/pico_tok_agnews_10k_v26.json"
    pad_token: str = "[PAD]"
    cls_token: str = "[CLS]"
    unk_token: str = "[UNK]"
    max_len: int = 128
    char_vocab: List[str] = field(
        default_factory=lambda: list(" abcdefghijklmnopqrstuvwxyz0123456789.,!?\"'()-/:;%&*$@")
    )
    char_dim: int = 24

    # ── Architecture ──────────────────────────────────────────────────
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    head_dim: int = 16
    d_ff: int = 320
    dropout: float = 0.1
    stoch_depth: float = 0.05
    conv_kernel: int = 3

    # ── System-2 ──────────────────────────────────────────────────────
    memory_slots: int = 6
    K_think: int = 2
    d_cot_ff: int = 128
    div_weight: float = 0.01
    search_branches: int = 3
    max_refinements: int = 3
    refine_epsilon: float = 1e-3
    latent_actions: int = 6
    mcts_simulations: int = 12
    mcts_rollout_depth: int = 2
    c_puct: float = 1.25

    # ── vMF ───────────────────────────────────────────────────────────
    vmf_kappa_init: float = 4.0
    vmf_use_kappa_weights: bool = True
    vmf_kappa_max: float = 30.0
    vmf_dual_scale: bool = True
    vmf_entropy_reg: float = 0.01
    vmf_asymmetric_qk: bool = True

    # ── Temporal & RoPE ───────────────────────────────────────────────
    use_temporal_transition: bool = True
    temporal_mlp_ratio: float = 2.0
    temporal_modulate_attention: bool = True
    temporal_n_features: int = 6
    rope_base: float = 10000.0
    rope_max_seq: int = 512

    # ── Regularisation ────────────────────────────────────────────────
    token_dropout_prob: float = 0.10
    token_dropout_apply: float = 0.20
    focal_gamma: float = 2.0
    mixup_alpha: float = 0.20
    mixup_prob_base: float = 0.15
    mixup_prob_min: float = 0.04
    label_smooth_base: float = 0.10
    label_smooth_min: float = 0.03

    # ── Dictionary Pre-Training ───────────────────────────────────────
    dict_epochs: int = 1500
    dict_lambda: float = 1.0
    dict_max_defs: int = 5
    dict_batch_size: int = 256
    dict_pos_weight: float = 50.0

    # ── Fine-Tuning ───────────────────────────────────────────────────
    epochs: int = 30
    batch_size: int = 64
    grad_accum: int = 2
    wd: float = 1e-4
    clip_grad: float = 1.0
    base_lr_max: float = 1e-3
    warmup_frac: float = 0.05
    min_lr_frac: float = 0.08
    ema_decay: float = 0.9995
    awp_eps: float = 0.005
    awp_lr: float = 0.010
    awp_start_ep: int = 6
    la_k: int = 6
    la_alpha: float = 0.50
    msd_k: int = 5
    attn_log_freq: int = 200
    checkpoint_every: int = 5
    checkpoint_dir: str = "./ckpt_apex_v26_dict"

    # ── Quantum LR ────────────────────────────────────────────────────
    use_quantum_lr: bool = True

    # ── Derived properties (not serialised) ───────────────────────────

    @property
    def device(self) -> torch.device:
        if self.device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_str)

    @property
    def amp_dtype(self) -> torch.dtype:
        if self.amp_dtype_str == "auto":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            elif torch.cuda.is_available():
                return torch.float16
            return torch.float32
        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return mapping.get(self.amp_dtype_str, torch.float32)

    # ── Serialisation helpers ─────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain ``dict`` copy (excludes computed properties)."""
        d = asdict(self)
        # Convert list fields for JSON safety
        if isinstance(d.get("char_vocab"), list):
            d["char_vocab"] = list(d["char_vocab"])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CFG":
        """Create a CFG from a dict, ignoring unknown keys."""
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)

    def update(self, **kwargs) -> "CFG":
        """Return a **new** CFG with the given fields overridden."""
        d = self.to_dict()
        d.update(kwargs)
        return CFG.from_dict(d)

    def count_parameters(self) -> int:
        """
        Rough parameter count estimate based on architecture dimensions.
        Useful for planning scaling experiments without instantiating the model.
        """
        d = self.d_model
        L = self.n_layers
        H = self.n_heads
        hd = self.head_dim
        dff = self.d_ff
        V = self.tokenizer_vocab_size
        A = self.latent_actions

        # Embedding
        emb = V * d + self.char_dim * d

        # Per transformer block
        # vMF attention: Q,K,V,O + gate = 5 * d * (H*hd) + kappa stuff
        attn_params = 5 * d * (H * hd) + d * H * 2 + H * 2  # rough
        # SwiGLU: d * 2*dff + dff * d
        ffn_params = d * 2 * dff + dff * d
        # Norms
        norm_params = d * 2

        blocks = L * (attn_params + ffn_params + norm_params)

        # Temporal
        temporal = int(d * d * self.temporal_mlp_ratio * 2.5) if self.use_temporal_transition else 0

        # System-2
        sys2_base = d * self.K_think
        sys2_search = attn_params  # one search block
        sys2_wm = A * d * d + d * d + dff * d
        sys2_ac = d * d + A * d + d
        sys2 = sys2_base + sys2_search + sys2_wm + sys2_ac

        # Heads
        head_params = d * d + d * self.n_classes  # MSD head
        dict_head_params = d * d + d * V  # Multi-label head

        # Global memory
        mem = self.memory_slots * d

        # GrafoConnect
        grafo = L * d * d + 2 * d * d + d

        total = emb + blocks + temporal + sys2 + head_params + dict_head_params + mem + grafo
        return total


def set_seed(s: int):
    """Deterministic seed across Python, NumPy, and PyTorch."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
