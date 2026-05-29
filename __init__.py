"""
GrafoPropagation v26-APEX
=========================

Geometric von Mises-Fisher (vMF) Networks with WordNet Pre-training.

A compact (~990k parameter) text classification architecture featuring:
  - Character-seed token embeddings (Fibonacci-sphere init)
  - Riemannian Temporal Transition Embedding (Log-Map + Parallel Transport)
  - RoPE-enhanced vMF Dual-Scale Attention
  - Dynamic GrafoConnect cross-layer skip graph
  - System-2 Latent Search with GumbelMCTS
  - Global Workspace Memory
  - Dictionary Pre-Training via WordNet

(c) 2025-2026 Claudio Fernandes. All rights reserved.
      PROPRIETARY AND CONFIDENTIAL — see LICENSE.
"""

__version__ = "0.26.0"
__author__  = "Claudio Fernandes"

# ── Core ────────────────────────────────────────────────────────────────
from .config import CFG, set_seed
from .model import GrafoPropagation

# ── Layers (re-exported for advanced users) ─────────────────────────────
from .primitives import RMSNorm, trunc_normal_, build_character_embeddings
from .positional import RoPERotator, TemporalTransitionEmbedding
from .attention import VonMisesFisherAttention
from .transformer import LocalConvMix, SwiGLU, TransformerBlock
from .memory import GlobalWorkspaceMemory, DynamicGrafoConnect
from .system2 import (
    ResidualWorldModel,
    PolicyValueHead,
    GumbelMCTS,
    System2LatentSearch,
)
from .heads import PoolingFusion, MultiSampleDropoutHead, MultiLabelHead

# ── Data ────────────────────────────────────────────────────────────────
from .tokenizer_utils import build_or_load_tokenizer
from .datasets import (
    TextDataset,
    DictionaryDataset,
    collate_dict,
    build_wordnet_multilabel,
)

# ── Losses ──────────────────────────────────────────────────────────────
from .losses import focal_ce, token_dropout

# ── Optimizer ───────────────────────────────────────────────────────────
from .optimizer import (
    EMA,
    Lookahead,
    AWP,
    register_gc,
    WarmupCosineLR,
)

# ── Quantum ─────────────────────────────────────────────────────────────
from .quantum import quantum_lr_modulation

# ── Training ────────────────────────────────────────────────────────────
from .train import run_training, pretrain_dictionary, train_epoch, evaluate

# ── Logging ─────────────────────────────────────────────────────────────
from .logging_utils import log, console, RUN_ID

__all__ = [
    # Core
    "CFG",
    "set_seed",
    "GrafoPropagation",
    # Layers
    "RMSNorm",
    "trunc_normal_",
    "build_character_embeddings",
    "RoPERotator",
    "TemporalTransitionEmbedding",
    "VonMisesFisherAttention",
    "LocalConvMix",
    "SwiGLU",
    "TransformerBlock",
    "GlobalWorkspaceMemory",
    "DynamicGrafoConnect",
    "ResidualWorldModel",
    "PolicyValueHead",
    "GumbelMCTS",
    "System2LatentSearch",
    "PoolingFusion",
    "MultiSampleDropoutHead",
    "MultiLabelHead",
    # Data
    "build_or_load_tokenizer",
    "TextDataset",
    "DictionaryDataset",
    "collate_dict",
    "build_wordnet_multilabel",
    # Losses
    "focal_ce",
    "token_dropout",
    # Optimizer
    "EMA",
    "Lookahead",
    "AWP",
    "register_gc",
    "WarmupCosineLR",
    # Quantum
    "quantum_lr_modulation",
    # Training
    "run_training",
    "pretrain_dictionary",
    "train_epoch",
    "evaluate",
    # Logging
    "log",
    "console",
    "RUN_ID",
]
