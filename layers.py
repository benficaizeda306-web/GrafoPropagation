"""
GrafoPropagation v26-APEX — Unified Layer Re-exports
=====================================================

Backward-compatibility shim: the original monolithic script imported
all layers from a single ``.layers`` namespace.  This module simply
re-exports every public layer so that old code continues to work.

New code should import directly from the sub-modules:

    from grafopropagation.primitives import RMSNorm
    from grafopropagation.attention import VonMisesFisherAttention

(c) 2025-2026 Claudio Fernandes. All rights reserved.
"""

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

__all__ = [
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
]
