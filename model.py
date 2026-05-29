"""
GrafoPropagation v26-APEX — Main Model
=======================================

A ~990 k-parameter text classifier built from:
  * Character-seed token embeddings (Fibonacci-sphere init)
  * Riemannian Temporal Transition Embedding
  * Local Depthwise Convolutional Mixer
  * Global Workspace Memory slots
  * RoPE-enhanced vMF Dual-Scale Attention
  * Dynamic GrafoConnect cross-layer skip graph
  * System-2 Latent Search with GumbelMCTS
  * Gated Pool-Fusion + Multi-Sample Dropout head

Dictionary Pre-Training (WordNet)
----------------------------------
The model includes a `MultiLabelHead` (`dict_head`) that predicts
vocabulary tokens appearing in the WordNet definition(s) of a given
input word.  Call `forward(..., return_dict_logits=True)` during
dictionary pre-training.

(c) 2025-2026 Claudio Fernandes. All rights reserved.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG
from .primitives import RMSNorm, build_character_embeddings
from .positional import RoPERotator, TemporalTransitionEmbedding
from .transformer import LocalConvMix, TransformerBlock
from .memory import GlobalWorkspaceMemory, DynamicGrafoConnect
from .system2 import System2LatentSearch
from .heads import PoolingFusion, MultiSampleDropoutHead, MultiLabelHead


class GrafoPropagation(nn.Module):
    """
    GrafoPropagation v26-APEX

    Fully configurable via the ``CFG`` dataclass — pass any architecture
    dimensions to scale the model up or down.

    Example
    -------
    >>> from grafopropagation import CFG, GrafoPropagation
    >>> cfg = CFG(d_model=128, n_layers=4, n_heads=8, head_dim=32)
    >>> model = GrafoPropagation(cfg, tokenizer)
    """

    def __init__(self, cfg: CFG, tokenizer):
        super().__init__()
        d = cfg.d_model
        L = cfg.n_layers
        dev = cfg.device

        # ── Character-seeded embeddings ───────────────────────────────
        char_emb = build_character_embeddings(cfg.char_vocab, cfg.char_dim, dev)
        self.register_buffer("char_emb_buf", char_emb)
        self.char_proj = nn.Linear(cfg.char_dim, d, bias=False).to(dev)
        nn.init.xavier_uniform_(self.char_proj.weight)

        vocab = tokenizer.get_vocab()
        tok2str = {v: k for k, v in vocab.items()}
        char2idx = {ch: i for i, ch in enumerate(cfg.char_vocab)}

        tvecs = torch.zeros(cfg.tokenizer_vocab_size, cfg.char_dim, device=dev)
        for tid in range(cfg.tokenizer_vocab_size):
            ts = tok2str.get(tid, "")
            if ts:
                idx = [char2idx.get(ch.lower(), char2idx.get(" ", 0)) for ch in ts]
                tvecs[tid] = char_emb[idx].sum(0)

        init_emb = self.char_proj(tvecs).detach()
        unk_id = tokenizer.token_to_id(cfg.unk_token)
        if unk_id is not None:
            mask = torch.ones(cfg.tokenizer_vocab_size, dtype=torch.bool, device=dev)
            mask[[0, 1, unk_id]] = False
            init_emb[unk_id] = init_emb[mask].mean(0)

        self.embed = nn.Embedding(cfg.tokenizer_vocab_size, d, padding_idx=0)
        self.embed.weight.data.copy_(init_emb)
        self.embed_scale = d ** 0.5
        self.embed_drop = nn.Dropout(cfg.dropout)

        # ── Positional & temporal ─────────────────────────────────────
        self.temporal_emb = (
            TemporalTransitionEmbedding(
                d, cfg.temporal_n_features,
                cfg.temporal_mlp_ratio,
                cfg.temporal_modulate_attention,
            )
            if cfg.use_temporal_transition else None
        )

        # ── Local mixer ───────────────────────────────────────────────
        self.conv_mix = LocalConvMix(d, cfg.conv_kernel, cfg.dropout)

        # ── Global workspace ──────────────────────────────────────────
        self.global_memory = GlobalWorkspaceMemory(cfg.memory_slots, d)

        # ── Transformer backbone ──────────────────────────────────────
        rope = RoPERotator(cfg.head_dim, cfg.rope_max_seq, cfg.rope_base).to(dev)
        sd_list = [cfg.stoch_depth * i / max(L - 1, 1) for i in range(L)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d, cfg.n_heads, cfg.head_dim, cfg.d_ff,
                cfg.dropout, sd_list[i],
                cfg.vmf_kappa_init, cfg.vmf_use_kappa_weights,
                rope,
                dual_scale=cfg.vmf_dual_scale,
                asymmetric_qk=cfg.vmf_asymmetric_qk,
                kappa_max=cfg.vmf_kappa_max,
                entropy_reg=cfg.vmf_entropy_reg,
            )
            for i in range(L)
        ])

        # ── Graph skip-connections ────────────────────────────────────
        self.grafo = DynamicGrafoConnect(L, d)

        # ── System-2 search ───────────────────────────────────────────
        self.system2 = System2LatentSearch(
            K=cfg.K_think, d=d,
            n_heads=cfg.n_heads, head_dim=cfg.head_dim,
            d_ff=cfg.d_cot_ff, dropout=cfg.dropout,
            branches=cfg.search_branches,
            max_iters=cfg.max_refinements,
            epsilon=cfg.refine_epsilon,
            n_actions=cfg.latent_actions,
            mcts_sims=cfg.mcts_simulations,
            rollout_depth=cfg.mcts_rollout_depth,
            device=dev,
            kappa_init=cfg.vmf_kappa_init,
            use_kappa_weights=cfg.vmf_use_kappa_weights,
        )

        # ── Output ────────────────────────────────────────────────────
        self.final_norm = RMSNorm(d)
        self.pool_fusion = PoolingFusion(d)
        self.head = MultiSampleDropoutHead(d, cfg.n_classes, cfg.dropout, cfg.msd_k)
        self.dict_head = MultiLabelHead(d, cfg.tokenizer_vocab_size)

    # ── Encoder ───────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor, fmask: torch.Tensor):
        """
        Parameters
        ----------
        x     : (B, T, D) token embeddings
        fmask : (B, T)    float attention mask (−∞ for padding)

        Returns
        -------
        pooled    : (B, D)
        total_ent : scalar entropy-regularisation loss
        think     : (B, D) mean System-2 thought representation
        """
        B = x.shape[0]
        tmask, curve = None, torch.zeros(B, device=x.device)

        if self.temporal_emb is not None:
            temb, tmask, curve = self.temporal_emb(x)
            x = x + temb

        x = self.conv_mix(x)
        x = self.global_memory.expand_context(x, B)

        total_ent = x.new_zeros(())
        cls_hist = []
        ti = self.global_memory.slots  # offset for first real token

        for l, block in enumerate(self.blocks):
            delta = self.grafo(cls_hist, l, curve)
            if delta is not None:
                x = torch.cat([
                    x[:, :ti],
                    x[:, ti:ti + 1] + delta.unsqueeze(1),
                    x[:, ti + 1:],
                ], dim=1)
            x, ent_loss = block(x, fmask, tmask)
            total_ent = total_ent + ent_loss
            cls_hist.append(x[:, ti].detach())

        x = self.global_memory.extract_and_update(x, B)
        x, _ = self.system2(x, fmask)

        K = self.system2.K
        think = self.final_norm(x[:, :K]).mean(1)  # (B, D)
        seq = self.final_norm(x[:, K:])  # (B, T, D)

        pooled = self.pool_fusion(think, seq, (fmask == float("-inf")))
        return pooled, total_ent, think

    # ── Forward ───────────────────────────────────────────────────────

    def forward(
        self,
        ids: torch.Tensor,
        fmask: torch.Tensor,
        return_dict_logits: bool = False,
    ):
        """
        Parameters
        ----------
        ids               : (B, T) token ids
        fmask             : (B, T) float mask
        return_dict_logits: if True, also return dict pre-training logits

        Returns
        -------
        logits      : (B, n_classes)
        ent_loss    : scalar
        dict_logits : (B, vocab_size) or None
        """
        emb = self.embed_drop(self.embed(ids) * self.embed_scale)
        pooled, ent_loss, think = self.encode(emb, fmask)
        logits = self.head(pooled)
        dict_logits = self.dict_head(pooled) if return_dict_logits else None
        return logits, ent_loss, dict_logits
