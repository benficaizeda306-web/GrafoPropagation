#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════╗
║  GrafoPropagation v26-APEX²  ·  System-2 Edition · Yahoo Answers              ║
╠════════════════════════════════════════════════════════════════════════════════╣
║  Base: v25-APEX (10.18 M)  — Cirurgias de precisão para máxima accuracy:       ║
║                                                                                ║
║  1. MCTS → IterativeWorldRefinement (IWR)                  [overhead removal]  ║
║     GumbelMCTS eliminado. Substituído por n_steps passos diferenciáveis.       ║
║     Policy soft-mixture sobre todas as acções em batch (sem loop Python).      ║
║     Mesmo forward para train e eval — sem mode-switch, sem @no_grad branches.  ║
║     O(n_steps·A) vs O(sims·⌈log₂A⌉·depth) do MCTS original.                  ║
║     Temperature aprendida controla exploração do world model.                  ║
║                                                                                ║
║  2. RoPE² — Second-Order Positional Encoding               [+1 param α]        ║
║     θ_p = p·θ₀ + α·p(p−1)/2·θ₀   (init α=0 → RoPE clássico no arranque)     ║
║     Captura aceleração angular. Computado on-the-fly: sem buffers pré-build.   ║
║     Crítico para Q+Content+Answer onde max_len subiu para 192.                 ║
║                                                                                ║
║  3. XOR Complementarity na vMF Attention                   [pré-RoPE semântico]║
║     soft_sign = 2·σ(5·μ_pre_rope) − 1  →  xor_sim = ssᵀ/D                   ║
║     scores += tanh(xor_w) · xor_sim   (init neutro: xor_w = 0)               ║
║     Opera no espaço semântico pré-RoPE: position-independent.                  ║
║     Captura complementaridade entre tokens (antónimos, contraste tópico).      ║
║                                                                                ║
║  4. Conditional V-Gate — Inversão Direcional do Gate       [+~200 params/layer]║
║     cond_dir ∈ R^(H×D), cond_scale ∈ R^H  (por head, pré-RoPE space)         ║
║     condition = σ(⟨μ_pre_rope, dir_normalized⟩ · scale)                       ║
║     gate_cond = cond · g + (1−cond) · (1−g)   ← inversão ARC-style           ║
║     Permite ao modelo "inverter" o gate consoante a direcção semântica.        ║
║                                                                                ║
║  5. max_len 128→192  /  batch 64→48  /  grad_accum 2→3  (eff_batch≈144)       ║
║     Yahoo Answers: Title+Content+BestAnswer → ~175 tok médios. Com 128 havia   ║
║     truncation massiva na parte mais informativa (a resposta).                 ║
║                                                                                ║
║  Parâmetros: ≈ 10.18 M  (net +~1.2k params vs v25, idêntico na prática)       ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

import subprocess, sys, os, importlib, math, time, random, copy
import warnings, datetime, json
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════
# AUTO-INSTALL
# ═══════════════════════════════════════════════════════════════════════
def _ok(mod):
    try: __import__(mod); return True
    except ImportError: return False

def _install():
    needed = {
        'pennylane':           'pennylane',
        'pennylane_lightning':  'pennylane-lightning',
        'datasets':            'datasets',
        'tokenizers':          'tokenizers',
    }
    missing = [p for m, p in needed.items() if not _ok(m)]
    if missing:
        print(f"[BOOT] Installing: {missing}")
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '--upgrade',
             '--no-cache-dir'] + missing, stdout=subprocess.DEVNULL)
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '--upgrade',
                 '--no-cache-dir', 'pennylane-lightning-gpu'],
                stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            pass
        importlib.invalidate_caches()

_install()

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from collections import deque
import pennylane as qml
from datasets import load_dataset
from tokenizers import (Tokenizer, models, pre_tokenizers, decoders,
                        trainers, normalizers)
from tokenizers.processors import TemplateProcessing

try:
    from rich.console import Console
    from rich.table   import Table
    from rich.markup  import escape as _esc
    console  = Console(width=180)
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    class _FC:
        def print(self, *a, **kw): print(*a)
        def rule(self, *a, **kw):  print('─' * 90)
    console = _FC()

torch.backends.cudnn.benchmark   = True
torch.set_float32_matmul_precision('high')

RUN_ID   = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
_LOG_BUF: list = []

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO v26-APEX²  ·  ~10.18 M params
# ═══════════════════════════════════════════════════════════════════════
class CFG:
    VERSION   = 'v26-APEX²'
    seed      = 42
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_dtype = (torch.bfloat16
                 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                 else torch.float16)

    # ── Dataset ────────────────────────────────────────────────────────
    dataset_name         = 'yahoo_answers_topics'
    max_train            = 150_000
    max_val              = 5_000
    n_classes            = 10
    tokenizer_vocab_size = 30_000
    # Reusa tokenizer v25 se existir (mesmos dados/settings BPE)
    tokenizer_path       = '/tmp/pico_tok_yahoo_answers_30k_v25.json'
    pad_token = '[PAD]'; cls_token = '[CLS]'; unk_token = '[UNK]'
    # ✦ v26: Aumentado de 128→192 para capturar title+content+answer completo
    max_len = 192
    char_vocab = list(' abcdefghijklmnopqrstuvwxyz0123456789.,!?"\''
                      '()-/:;%&*$@')
    char_dim = 48

    # ── Arquitectura ≈ 10.18 M ─────────────────────────────────────────
    d_model     = 192
    n_layers    = 4
    n_heads     = 8
    head_dim    = 24
    d_ff        = 896
    dropout     = 0.15
    stoch_depth = 0.10
    conv_kernel = 3

    # ── System-2 (MCTS removido, IWR adicionado) ───────────────────────
    memory_slots       = 12
    K_think            = 4
    d_cot_ff           = 512
    div_weight         = 0.01
    search_branches    = 3
    max_refinements    = 3
    refine_epsilon     = 1e-3
    latent_actions     = 6
    # ✦ v26: Substitui mcts_simulations + mcts_rollout_depth
    n_refinement_steps = 4

    # ── Temporal & RoPE²-vMF ──────────────────────────────────────────
    use_temporal_transition     = True
    temporal_mlp_ratio          = 2.0
    temporal_modulate_attention = True
    temporal_n_features         = 6
    vmf_kappa_init              = 4.0
    vmf_use_kappa_weights       = True
    vmf_kappa_max               = 30.0
    rope_base                   = 10000.0
    rope_max_seq                = 512

    # ── Regularização ──────────────────────────────────────────────────
    token_dropout_prob  = 0.10
    token_dropout_apply = 0.20
    focal_gamma         = 2.0
    mixup_alpha         = 0.20
    mixup_prob_base     = 0.15
    mixup_prob_min      = 0.04
    label_smooth_base   = 0.10
    label_smooth_min    = 0.03

    # ── Treino (ajustado para max_len=192) ─────────────────────────────
    epochs       = 30
    # ✦ v26: batch 64→48, accum 2→3  →  eff_batch = 144  (≈ mesmo que v25 128)
    batch_size   = 48
    grad_accum   = 3
    wd           = 1e-4
    clip_grad    = 1.0
    base_lr_max  = 3e-4
    warmup_frac  = 0.05
    min_lr_frac  = 0.08
    ema_decay    = 0.9995
    awp_eps      = 0.005
    awp_lr       = 0.010
    awp_start_ep = 6
    la_k         = 6
    la_alpha     = 0.50
    msd_k        = 5
    attn_log_freq    = 200
    checkpoint_every = 5
    checkpoint_dir   = './ckpt_apex2_v26_yahoo'


def set_seed(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(CFG.seed)

# ═══════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════
_STYLE = {
    'INFO': 'dim', 'WARN': 'yellow', 'ERROR': 'bold red',
    'METRIC': 'bold green', 'ATTN': 'bold magenta',
    'SYS2': 'bold cyan', 'IWR': 'bold yellow',
}

def _ts(): return datetime.datetime.utcnow().isoformat() + 'Z'

def log(msg: str, level: str = 'INFO'):
    _LOG_BUF.append({'ts': _ts(), 'run': RUN_ID, 'lvl': level, 'msg': msg})
    s = _STYLE.get(level, '')
    if HAS_RICH:
        console.print(f'[{s}][{level}] {_esc(msg)}[/{s}]' if s else f'[{level}] {_esc(msg)}')
    else:
        print(f'[{level}] {msg}')

# ═══════════════════════════════════════════════════════════════════════
# QUANTUM LR MODULATION
# ═══════════════════════════════════════════════════════════════════════
def _choose_backend(name='lightning.gpu', wires=8):
    try:
        dev = qml.device(name, wires=wires)
        log(f'[QDEV] {name}', 'INFO'); return dev
    except Exception as e:
        log(f'[QDEV] fallback lightning.qubit ({e})', 'WARN')
        return qml.device('lightning.qubit', wires=wires)

_dev_lr = _choose_backend('lightning.gpu', 8)

@qml.qnode(_dev_lr, interface='torch', diff_method=None)
def _qlr_circuit(epoch_idx):
    for i in range(8):
        if (epoch_idx >> i) & 1: qml.PauliX(wires=i)
    for i in range(8): qml.RY(0.5 + 0.1 * i, wires=i)
    for i in range(7): qml.CZ(wires=[i, i + 1])
    for i in range(8): qml.RX(0.7 - 0.05 * i, wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(8)]

_lr_q = deque(maxlen=3)
def quantum_lr_modulation(epoch: int) -> float:
    v = 0.7 + 0.6 * ((torch.stack(_qlr_circuit(epoch)).sum().item() + 8.0) / 16.0)
    _lr_q.append(v)
    return sum(_lr_q) / len(_lr_q)

# ═══════════════════════════════════════════════════════════════════════
# PRIMITIVAS BASE
# ═══════════════════════════════════════════════════════════════════════
def _trunc_normal_(t: torch.Tensor, mean=0.0, std=0.02, a=-2.0, b=2.0):
    with torch.no_grad():
        nn.init.normal_(t, mean, std)
        t.clamp_(mean + a * std, mean + b * std)
        for _ in range(10):
            m = (t < mean + a * std) | (t > mean + b * std)
            if not m.any(): break
            nn.init.normal_(t[m], mean, std)
            t[m].clamp_(mean + a * std, mean + b * std)


def build_character_embeddings(char_vocab, dim, device):
    letters     = set('abcdefghijklmnopqrstuvwxyz') | set('0123456789')
    punct_chars = set(char_vocab) - letters - {' '}
    n   = len(char_vocab)
    emb = torch.empty(n, dim, device=device)
    li  = [i for i, ch in enumerate(char_vocab) if ch in letters]
    phi = math.pi * (3. - math.sqrt(5.))
    for idx, i in enumerate(li):
        y = 1. - (idx / float(max(len(li) - 1, 1))) * 2.
        r = math.sqrt(max(0.0, 1. - y * y))
        base = torch.tensor([math.cos(phi * idx) * r, y,
                              math.sin(phi * idx) * r], device=device)
        emb[i] = (torch.cat([base, torch.randn(dim - 3, device=device) * 0.05])
                  if dim > 3 else base)
    for i in [i for i, ch in enumerate(char_vocab) if ch in punct_chars]:
        base = torch.cat([torch.randn(2, device=device) * 0.1,
                          torch.tensor([1.0], device=device)])
        emb[i] = (torch.cat([base, torch.randn(dim - 3, device=device) * 0.05])
                  if dim > 3 else base)
    if ' ' in char_vocab: emb[char_vocab.index(' ')].zero_()
    return emb


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(d))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf  = x.float()
        rms = xf.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (xf * rms * self.scale.float()).to(x.dtype)

# ═══════════════════════════════════════════════════════════════════════
# TEMPORAL TRANSITION EMBEDDING (RIEMANNIAN LOG-MAP) — VERSÃO ESTÁVEL
# ═══════════════════════════════════════════════════════════════════════
class TemporalTransitionEmbedding(nn.Module):
    def __init__(self, d: int, n_feat: int = 6, ratio: float = 2.0,
                 modulate: bool = True):
        super().__init__()
        self.modulate = modulate
        hidden = int(d * ratio)
        self.proj = nn.Sequential(
            nn.Linear(d, hidden), RMSNorm(hidden),
            nn.GELU(approximate='tanh'), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), RMSNorm(hidden),
            nn.GELU(approximate='tanh'), nn.Dropout(0.1),
            nn.Linear(hidden, d))
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        if modulate:
            self.temporal_decay = nn.Parameter(torch.tensor(1.0))
            self.temporal_bias  = nn.Parameter(torch.tensor(0.0))
        self.eps = 1e-8

    def _log_map(self, x_t: torch.Tensor, x_tp1: torch.Tensor) -> torch.Tensor:
        dot = (x_t * x_tp1).sum(-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(dot)
        perp = x_tp1 - dot * x_t
        sin_theta = torch.sin(theta).clamp(min=1e-6)
        factor = theta / sin_theta
        small  = (theta < 1e-4).float()
        factor = (1 - small) * factor + small * (1.0 + theta.pow(2) / 6.0)
        return factor * perp

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        xn = F.normalize(x.float(), p=2, dim=-1, eps=self.eps)
        cos_t = (xn[:, :-1] * xn[:, 1:]).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(cos_t)
        theta_padded = F.pad(theta, (1, 0), value=0.0)
        v_t_1 = self._log_map(xn[:, :-1], xn[:, 1:])
        v_zero = torch.zeros(B, 1, D, device=x.device, dtype=v_t_1.dtype)
        v_vectors = torch.cat([v_zero, v_t_1], dim=1)
        t_emb = self.proj(v_vectors)
        t_mask = None
        if self.modulate:
            cs   = torch.cumsum(theta_padded, -1)
            dist = (cs.unsqueeze(-1) - cs.unsqueeze(-2)).abs()
            dec  = torch.clamp(self.temporal_decay, min=0.0)
            t_mask = (torch.exp(-dec * dist) + self.temporal_bias).to(x.dtype)
        omega  = theta_padded[:, 1:] - theta_padded[:, :-1]
        omega  = F.pad(omega, (1, 0), value=0.0)
        raw_k  = (omega.abs() / (theta_padded * theta_padded + self.eps)).clamp(0, 20)
        curve  = raw_k.mean(-1)
        return t_emb, t_mask, curve

# ═══════════════════════════════════════════════════════════════════════
# RoPE²  —  Second-Order Positional Encoding
# ✦ v26: α aprendido. θ_p = p·θ₀ + α·p(p−1)/2·θ₀
#         init α=0 → RoPE clássico no arranque, depois aprende aceleração.
#         Computado on-the-fly: sem buffers pré-build, sem _build().
# ═══════════════════════════════════════════════════════════════════════
class RoPERotator(nn.Module):
    def __init__(self, hd: int, max_len: int = 512, base: float = 10000.0):
        super().__init__()
        assert hd % 2 == 0
        inv = 1.0 / (base ** (torch.arange(0, hd // 2, dtype=torch.float32) / (hd // 2)))
        self.register_buffer('inv_freq', inv)
        self.hd    = hd
        # ✦ v26: parâmetro de aceleração angular (init=0 → RoPE clássico)
        self.alpha = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _rot(x: torch.Tensor) -> torch.Tensor:
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], -1)

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        B, T, H, D = mu.shape
        p      = torch.arange(T, dtype=mu.dtype, device=mu.device)
        inv    = self.inv_freq.to(mu.dtype)
        # θ_p = p·inv_freq + α·p(p−1)/2·inv_freq
        theta  = (torch.outer(p, inv)
                  + self.alpha.to(mu.dtype) * torch.outer(p * (p - 1) * 0.5, inv))
        theta  = torch.cat([theta, theta], dim=-1)             # (T, D)
        cos_c  = theta.cos().view(1, T, 1, D)
        sin_c  = theta.sin().view(1, T, 1, D)
        return F.normalize(mu * cos_c + self._rot(mu) * sin_c, p=2, dim=-1, eps=1e-8)

# ═══════════════════════════════════════════════════════════════════════
# vMF ATTENTION  —  Output Gate  +  XOR Complementarity  +  Conditional Gate
# ✦ v26 changes:
#   (A) XOR Complementarity Term:
#       soft_sign = 2σ(5·μ_pre_rope)−1  →  xor_sim = ssᵀ/D
#       scores += tanh(xor_w) · xor_sim   (init neutro: xor_w=0)
#       Opera em μ_pre_rope: position-independent, espaço semântico puro.
#   (B) Conditional V-Gate:
#       cond_dir ∈ R^(H×D), cond_scale ∈ R^H  (pré-RoPE space)
#       condition = σ(⟨μ_pre_rope, cond_dir_norm⟩ · scale)
#       gate_cond = cond·g + (1−cond)·(1−g)  ← inversão direcional
# ═══════════════════════════════════════════════════════════════════════
class VonMisesFisherAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, hd: int, dropout: float = 0.1,
                 kappa_init: float = 4.0, use_kappa_weights: bool = True,
                 rope: 'RoPERotator | None' = None):
        super().__init__()
        self.n_heads = n_heads; self.hd = hd
        self.dp  = dropout; self.eps = 1e-8
        self.ukw = use_kappa_weights; self.rope = rope
        self._sc = 1.0 / math.sqrt(hd)

        self.W_mu    = nn.Linear(d, n_heads * hd, bias=False)
        self.W_kappa = nn.Linear(d, n_heads,      bias=True)
        self.Wv      = nn.Linear(d, n_heads * hd, bias=False)
        self.Wo      = nn.Linear(n_heads * hd, d, bias=False)
        self.W_gate  = nn.Linear(d, n_heads * hd, bias=False)

        # ✦ v26 (A): XOR complementarity weight (init=0 → neutro)
        self.xor_weight = nn.Parameter(torch.zeros(1))

        # ✦ v26 (B): Conditional gate direction (pré-RoPE semantic space)
        self.cond_dir   = nn.Parameter(torch.randn(n_heads, hd) * 0.01)
        self.cond_scale = nn.Parameter(torch.ones(n_heads))

        g = 1.0 / math.sqrt(2)
        for w in [self.W_mu, self.Wv, self.Wo]:
            nn.init.xavier_uniform_(w.weight, gain=g)
        nn.init.xavier_uniform_(self.W_gate.weight, gain=g)
        nn.init.xavier_uniform_(self.W_kappa.weight, gain=0.1)
        nn.init.constant_(self.W_kappa.bias, math.log(max(kappa_init - 1.0, 1e-4)))
        self.tau    = nn.Parameter(torch.ones(n_heads) * 2.0)
        self.bias_q = nn.Parameter(torch.zeros(n_heads))

    def get_kappa(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(F.softplus(self.W_kappa(x)) + 1e-4,
                           max=CFG.vmf_kappa_max)

    def forward(self, x: torch.Tensor, fmask=None, tmask=None) -> torch.Tensor:
        B, T, D = x.shape

        # ── μ normalizado pré-RoPE (espaço semântico puro) ──────────────
        mu_normed = F.normalize(
            self.W_mu(x).view(B, T, self.n_heads, self.hd),
            p=2, dim=-1, eps=self.eps)                         # (B,T,H,D)

        # ── μ com RoPE (espaço posicional) ──────────────────────────────
        if self.rope is not None:
            mu = self.rope(mu_normed)
        else:
            mu = mu_normed

        kappa   = self.get_kappa(x)
        mu_h    = mu.permute(0, 2, 1, 3)               # (B,H,T,D) — pós-RoPE
        mu_h_s  = mu_normed.permute(0, 2, 1, 3)        # (B,H,T,D) — pré-RoPE

        # ── Atenção vMF (coseno × kappa × tau) ─────────────────────────
        S = torch.matmul(mu_h, mu_h.transpose(-2, -1))
        if self.ukw:
            kh = kappa.permute(0, 2, 1)
            S  = torch.sqrt(kh.unsqueeze(-1) * kh.unsqueeze(-2) + self.eps) * S
        scores = (self.tau.view(1, self.n_heads, 1, 1) * (S * self._sc)
                  + self.bias_q.view(1, self.n_heads, 1, 1))

        # ── ✦ v26 (A): XOR Complementarity Term ─────────────────────────
        # soft_sign em fp32 para estabilidade; depois cast para dtype do mu
        soft_sign = (2.0 * torch.sigmoid(mu_h_s.float() * 5.0) - 1.0
                     ).to(mu_h.dtype)                          # (B,H,T,D) ∈(−1,1)
        xor_sim   = torch.matmul(soft_sign,
                                 soft_sign.transpose(-2, -1)) / self.hd  # (B,H,T,T)
        scores    = scores + torch.tanh(self.xor_weight) * xor_sim

        # ── Masks ────────────────────────────────────────────────────────
        if tmask is not None:
            pl = T - tmask.shape[-1]
            if pl > 0:
                te = torch.zeros(B, T, T, device=x.device, dtype=x.dtype)
                te[:, pl:, pl:] = tmask
                scores = scores + te.unsqueeze(1)
            else:
                scores = scores + tmask.unsqueeze(1)
        if fmask is not None:
            pl = T - fmask.shape[-1]
            if pl > 0:
                fe = torch.zeros(B, T, device=x.device, dtype=x.dtype)
                fe[:, pl:] = fmask
                scores = scores + fe[:, None, None, :]
            else:
                scores = scores + fmask[:, None, None, :]

        attn = F.dropout(F.softmax(scores, dim=-1),
                         p=self.dp if self.training else 0.0,
                         training=self.training)

        v    = self.Wv(x).view(B, T, self.n_heads, self.hd).permute(0, 2, 1, 3)
        av   = torch.matmul(attn, v)                           # (B,H,T,D)

        # ── Gate base (output gate do v25) ───────────────────────────────
        gate = torch.sigmoid(
            self.W_gate(x).view(B, T, self.n_heads, self.hd)
            .permute(0, 2, 1, 3))                              # (B,H,T,D)

        # ── ✦ v26 (B): Conditional Gate — inversão direcional ────────────
        cond_dir_n = F.normalize(self.cond_dir.to(x.dtype), dim=-1)    # (H,D)
        cond_proj  = (mu_h_s * cond_dir_n.view(1, self.n_heads, 1, self.hd)
                      ).sum(-1, keepdim=True)                  # (B,H,T,1)
        scale_c    = self.cond_scale.view(1, self.n_heads, 1, 1).to(x.dtype)
        condition  = torch.sigmoid(cond_proj * scale_c)        # (B,H,T,1)
        # Quando condition≈1 → gate normal;  condition≈0 → gate invertido
        gate_cond  = condition * gate + (1.0 - condition) * (1.0 - gate)

        out = (gate_cond * av).permute(0, 2, 1, 3).reshape(B, T, self.n_heads * self.hd)
        return self.Wo(out)

# ═══════════════════════════════════════════════════════════════════════
# MIXERS & BLOCK
# ═══════════════════════════════════════════════════════════════════════
class LocalConvMix(nn.Module):
    def __init__(self, d: int, k: int = 3, dp: float = 0.1):
        super().__init__()
        self.norm = RMSNorm(d)
        self.dw   = nn.Conv1d(d, d, k, padding=(k-1)//2, groups=d, bias=False)
        self.pw   = nn.Conv1d(d, d, 1, bias=False)
        self.act  = nn.GELU(approximate='tanh')
        self.drop = nn.Dropout(dp)
        nn.init.kaiming_normal_(self.dw.weight, nonlinearity='linear')
        nn.init.xavier_uniform_(self.pw.weight)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).transpose(1, 2).contiguous()
        return x + self.drop(self.act(self.pw(self.dw(h))).transpose(1, 2).contiguous())


class SwiGLU(nn.Module):
    def __init__(self, d: int, dff: int, dp: float = 0.1):
        super().__init__()
        self.W_gu = nn.Linear(d, 2 * dff, bias=False)
        self.Wd   = nn.Linear(dff, d,     bias=False)
        self.drop = nn.Dropout(dp)
        nn.init.kaiming_normal_(self.W_gu.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.Wd.weight, gain=1.0 / math.sqrt(12))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = self.W_gu(x).chunk(2, dim=-1)
        return self.drop(self.Wd(F.silu(g) * u))


class TransformerBlock(nn.Module):
    def __init__(self, d: int, nh: int, hd: int, dff: int,
                 dp: float = 0.1, sd: float = 0.0,
                 ki: float = 4.0, ukw: bool = True, rope=None):
        super().__init__()
        self.sd    = sd
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.attn  = VonMisesFisherAttention(d, nh, hd, dp, ki, ukw, rope)
        self.ffn   = SwiGLU(d, dff, dp)

    def _drop(self, r: torch.Tensor) -> torch.Tensor:
        if not self.training or self.sd == 0.0: return r
        keep = (torch.rand(r.shape[0], 1, 1, device=r.device) > self.sd).float()
        return r * keep / (1.0 - self.sd)

    def forward(self, x: torch.Tensor, fmask=None, tmask=None) -> torch.Tensor:
        x = x + self._drop(self.attn(self.norm1(x), fmask, tmask))
        return x + self._drop(self.ffn(self.norm2(x)))

# ═══════════════════════════════════════════════════════════════════════
# GLOBAL WORKSPACE MEMORY
# ═══════════════════════════════════════════════════════════════════════
class GlobalWorkspaceMemory(nn.Module):
    def __init__(self, slots: int, d: int):
        super().__init__()
        self.slots = slots
        self.bank  = nn.Parameter(torch.randn(1, slots, d) * 0.02)
        self.norm  = RMSNorm(d)
    def expand_context(self, x: torch.Tensor, B: int) -> torch.Tensor:
        return torch.cat([self.norm(self.bank).expand(B, -1, -1), x], dim=1)
    def extract_and_update(self, ctx: torch.Tensor, B: int) -> torch.Tensor:
        new = ctx[:, :self.slots].mean(0, keepdim=True)
        if self.training: self.bank.data.lerp_(new.detach().data, 0.01)
        return ctx[:, self.slots:]

# ═══════════════════════════════════════════════════════════════════════
# DYNAMIC GRAFO CONNECT
# ═══════════════════════════════════════════════════════════════════════
class DynamicGrafoConnect(nn.Module):
    def __init__(self, L: int, d: int):
        super().__init__()
        self.L        = L
        self.A        = nn.Parameter(torch.zeros(L, L))
        self.src      = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(L)])
        for p in self.src: nn.init.eye_(p.weight)
        self.gate     = nn.Linear(2 * d, d, bias=True)
        nn.init.zeros_(self.gate.bias)
        self.norm     = RMSNorm(d)
        self.time_mod = nn.Linear(1, L, bias=False)

    def forward(self, hist: list, tgt: int,
                curve: torch.Tensor) -> 'torch.Tensor | None':
        n = len(hist)
        if n == 0: return None
        impact  = torch.sigmoid(self.time_mod(curve.unsqueeze(-1)))
        dyn_A   = self.A[tgt, :n].unsqueeze(0) * (1.0 + impact[:, :n])
        w       = F.softmax(dyn_A, dim=-1)
        agg     = sum(self.src[i](hist[i]) * w[:, i].unsqueeze(-1) for i in range(n))
        g       = torch.sigmoid(self.gate(torch.cat([hist[-1], agg], dim=-1)))
        return self.norm(g * agg)

# ═══════════════════════════════════════════════════════════════════════
# RESIDUAL WORLD MODEL
# ═══════════════════════════════════════════════════════════════════════
class ResidualWorldModel(nn.Module):
    def __init__(self, d: int, n_actions: int, n_heads: int, hd: int,
                 d_ff: int, dp: float):
        super().__init__()
        self.n_actions  = n_actions
        self.input_norm = RMSNorm(d)
        self.action_W   = nn.Parameter(torch.empty(n_actions, d, d))
        for a in range(n_actions):
            nn.init.xavier_uniform_(self.action_W[a], gain=0.3)
        self.action_scale = nn.Parameter(torch.full((n_actions,), 0.1))
        self.post_norm    = RMSNorm(d)
        self.coherence    = TransformerBlock(d, n_heads, hd, d_ff, dp, 0.0,
                                             CFG.vmf_kappa_init,
                                             CFG.vmf_use_kappa_weights, rope=None)

    def forward(self, state: torch.Tensor,
                action_idx: torch.Tensor) -> torch.Tensor:
        B, K, D = state.shape
        normed  = self.input_norm(state)
        W       = self.action_W[action_idx]                    # (B, D, D)
        scale   = self.action_scale[action_idx].view(B, 1, 1)
        delta   = torch.bmm(normed, W.transpose(-1, -2)) * scale
        out     = self.post_norm(state + delta)
        return self.coherence(out)

# ═══════════════════════════════════════════════════════════════════════
# POLICY-VALUE HEAD
# ═══════════════════════════════════════════════════════════════════════
class PolicyValueHead(nn.Module):
    def __init__(self, d: int, n_actions: int):
        super().__init__()
        self.shared      = nn.Sequential(
            nn.Linear(d, d), RMSNorm(d), nn.GELU(approximate='tanh'))
        self.policy_head = nn.Linear(d, n_actions)
        self.value_head  = nn.Linear(d, 1)
        nn.init.xavier_uniform_(self.policy_head.weight)
        nn.init.xavier_uniform_(self.value_head.weight)

    def forward(self, state: torch.Tensor):
        h  = self.shared(state.mean(dim=1))
        pi = F.softmax(self.policy_head(h), dim=-1)
        v  = torch.tanh(self.value_head(h))
        return pi, v

# ═══════════════════════════════════════════════════════════════════════
# ✦ v26: ITERATIVE WORLD REFINEMENT  (substitui GumbelMCTS)
#
# Princípio:
#   Em cada passo, computar política π(s) sobre A acções latentes.
#   Aplicar world model a TODAS as acções em paralelo (batch B×A).
#   Combinar children por média pesada pela política → estado refinado.
#   Repetir n_steps vezes.
#
# Propriedades:
#   • Totalmente diferenciável — mesmo forward para train e eval.
#   • Sem branches if self.training / else (eliminado code path divergence).
#   • GPU-friendly: um bmm para todas as acções, sem loop Python por simulação.
#   • Temperature aprendida: softplus(T)·π controla exploração do world model.
#   • O(n_steps·A) vs O(sims·⌈log₂A⌉·depth·A) do MCTS.
# ═══════════════════════════════════════════════════════════════════════
class IterativeWorldRefinement(nn.Module):
    def __init__(self, world_model: ResidualWorldModel,
                 actor_critic: PolicyValueHead,
                 n_actions: int, n_steps: int, device: torch.device):
        super().__init__()
        self.world_model  = world_model
        self.actor_critic = actor_critic
        self.n_actions    = n_actions
        self.n_steps      = n_steps
        self.device       = device
        # Temperature aprendida para soft policy (init=1 → temperatura standard)
        self.temperature  = nn.Parameter(torch.ones(1))

    def forward(self, state: torch.Tensor):
        """
        state : (B, K, D)  — K think tokens
        returns: (refined_state, final_policy)
        """
        B, K, D = state.shape
        cur      = state
        last_pi  = None

        for _ in range(self.n_steps):
            pi, _ = self.actor_critic(cur)                     # (B, A)
            temp  = F.softplus(self.temperature).clamp(min=0.1)
            pi_s  = F.softmax(pi / temp, dim=-1)               # (B, A) soft policy

            # Expandir estado para todas as acções em batch
            cur_exp = (cur.unsqueeze(1)
                       .expand(B, self.n_actions, K, D)
                       .reshape(B * self.n_actions, K, D))
            a_idx   = (torch.arange(self.n_actions, device=self.device)
                       .unsqueeze(0).expand(B, -1)
                       .reshape(-1))                            # (B*A,)

            children = self.world_model(cur_exp, a_idx)        # (B*A, K, D)
            children = children.view(B, self.n_actions, K, D)

            # Policy-weighted mixture (diferenciável end-to-end)
            cur     = (children *
                       pi_s.unsqueeze(-1).unsqueeze(-1)).sum(1) # (B, K, D)
            last_pi = pi_s

        final_pi = (last_pi if last_pi is not None
                    else torch.full((B, self.n_actions),
                                    1.0 / self.n_actions,
                                    device=self.device))
        return cur, final_pi

# ═══════════════════════════════════════════════════════════════════════
# SYSTEM-2 LATENT SEARCH  (Beam-Search + IWR)
# ═══════════════════════════════════════════════════════════════════════
class System2LatentSearch(nn.Module):
    def __init__(self, K: int, d: int, n_heads: int, hd: int, d_ff: int,
                 dp: float, branches: int, max_iters: int, epsilon: float,
                 n_actions: int, n_refinement_steps: int,
                 device: torch.device):
        super().__init__()
        self.K = K; self.branches = branches
        self.max_iters = max_iters; self.eps = epsilon
        self.base_think   = nn.Parameter(torch.empty(1, K, d))
        _trunc_normal_(self.base_think, std=0.02)
        self.branch_noise = nn.Parameter(torch.randn(branches, 1, K, d) * 0.05)
        self.search_block = TransformerBlock(
            d, n_heads, hd, d_ff, dp, 0.0,
            CFG.vmf_kappa_init, CFG.vmf_use_kappa_weights, rope=None)
        self.norm         = RMSNorm(d)
        self.world_model  = ResidualWorldModel(d, n_actions, n_heads, hd, d_ff, dp)
        self.actor_critic = PolicyValueHead(d, n_actions)
        # ✦ v26: IWR substitui GumbelMCTS
        self.refinement   = IterativeWorldRefinement(
            self.world_model, self.actor_critic,
            n_actions=n_actions, n_steps=n_refinement_steps, device=device)
        log(f'  IWR: A={n_actions}  steps={n_refinement_steps}  '
            f'temp=learned  [GumbelMCTS removed]', 'IWR')

    def diversity_loss(self) -> torch.Tensor:
        if self.K <= 1: return self.base_think.new_zeros(())
        t = F.normalize(self.base_think.squeeze(0), dim=-1)
        return (t @ t.T).triu(1).pow(2).sum() * (2.0 / (self.K * (self.K - 1)))

    def forward(self, seq_x: torch.Tensor, fmask=None):
        B, T, D = seq_x.shape
        ext_mask = None
        if fmask is not None:
            ext_mask = torch.cat(
                [torch.zeros(B, self.K, device=fmask.device, dtype=fmask.dtype),
                 fmask], dim=1)
        best = self.base_think.expand(B, -1, -1)

        for it in range(self.max_iters):
            prev = best.detach()
            branch_t = best.unsqueeze(0) + self.branch_noise
            evals, thoughts = [], []
            for b in range(self.branches):
                h    = torch.cat([branch_t[b], seq_x], dim=1)
                hout = self.search_block(h, ext_mask)
                th   = self.norm(hout[:, :self.K])
                ksc  = self.search_block.attn.get_kappa(
                    hout[:, :self.K]).mean([1, 2])
                evals.append(ksc); thoughts.append(th)
            w    = F.softmax(torch.stack(evals, 0), 0)
            best = (torch.stack(thoughts, 0)
                    * w.unsqueeze(-1).unsqueeze(-1)).sum(0)
            if torch.norm(best - prev, dim=-1).mean() < self.eps and it > 0:
                break

        # ✦ v26: IWR — mesmo forward para train e eval, totalmente diferenciável
        refined, _ = self.refinement(best)
        return torch.cat([refined, seq_x], dim=1), ext_mask

# ═══════════════════════════════════════════════════════════════════════
# POOLING & HEAD
# ═══════════════════════════════════════════════════════════════════════
class PoolingFusion(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.gate = nn.Linear(2 * d, d, bias=True)
        self.norm = RMSNorm(d)
        nn.init.xavier_uniform_(self.gate.weight, gain=0.5)
        nn.init.zeros_(self.gate.bias)
    def forward(self, think: torch.Tensor, seq: torch.Tensor,
                pad_mask: torch.Tensor) -> torch.Tensor:
        valid   = (~pad_mask).to(seq.dtype).unsqueeze(-1)
        seq_avg = (seq * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        g = torch.sigmoid(self.gate(torch.cat([think, seq_avg], -1)))
        return self.norm(g * think + (1.0 - g) * seq_avg)


class MultiSampleDropoutHead(nn.Module):
    def __init__(self, d: int, nc: int, dp: float = 0.1, k: int = 5):
        super().__init__()
        self.k = k; self.dp = dp
        self.fc1 = nn.Linear(d, d); self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(d, nc)
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.02)
        nn.init.zeros_(self.fc2.bias)
    def _once(self, x): return self.fc2(
        F.dropout(self.act(self.fc1(x)), p=self.dp, training=True))
    def forward(self, x):
        return self._once(x) if self.training else \
            torch.stack([self._once(x) for _ in range(self.k)]).mean(0)

# ═══════════════════════════════════════════════════════════════════════
# GRAFOPROPAGATION v26-APEX²
# ═══════════════════════════════════════════════════════════════════════
class GrafoPropagation(nn.Module):
    def __init__(self, cfg: CFG, tokenizer):
        super().__init__()
        d, L, dev = cfg.d_model, cfg.n_layers, cfg.device

        char_emb = build_character_embeddings(cfg.char_vocab, cfg.char_dim, dev)
        self.register_buffer('char_emb_buf', char_emb)
        self.char_proj = nn.Linear(cfg.char_dim, d, bias=False).to(dev)
        nn.init.xavier_uniform_(self.char_proj.weight)

        vocab     = tokenizer.get_vocab()
        tok2str   = {v: k for k, v in vocab.items()}
        char2idx  = {ch: i for i, ch in enumerate(cfg.char_vocab)}
        tvecs     = torch.zeros(cfg.tokenizer_vocab_size, cfg.char_dim, device=dev)
        for tid in range(cfg.tokenizer_vocab_size):
            ts = tok2str.get(tid, '')
            if ts:
                idx = [char2idx.get(ch.lower(), char2idx.get(' ', 0)) for ch in ts]
                tvecs[tid] = char_emb[idx].sum(0)
        init_emb = self.char_proj(tvecs).detach()
        unk_id   = tokenizer.token_to_id(cfg.unk_token)
        if unk_id is not None:
            mask = torch.ones(cfg.tokenizer_vocab_size, dtype=torch.bool, device=dev)
            mask[[0, 1, unk_id]] = False
            init_emb[unk_id] = init_emb[mask].mean(0)

        self.embed       = nn.Embedding(cfg.tokenizer_vocab_size, d, padding_idx=0)
        self.embed.weight.data.copy_(init_emb)
        self.embed_scale = d ** 0.5
        self.embed_drop  = nn.Dropout(cfg.dropout)

        self.temporal_emb = (
            TemporalTransitionEmbedding(d, cfg.temporal_n_features,
                                        cfg.temporal_mlp_ratio,
                                        cfg.temporal_modulate_attention)
            if cfg.use_temporal_transition else None)

        self.conv_mix      = LocalConvMix(d, cfg.conv_kernel, cfg.dropout)
        self.global_memory = GlobalWorkspaceMemory(cfg.memory_slots, d)

        # ✦ v26: RoPERotator com alpha (2nd-order); shared por todos os blocos
        rope = RoPERotator(cfg.head_dim, cfg.rope_max_seq, cfg.rope_base).to(dev)

        sd_list = [cfg.stoch_depth * i / max(L - 1, 1) for i in range(L)]
        log('─── v26-APEX²: TransformerBlocks (RoPE²-vMF + XOR + CondGate) ───', 'ATTN')
        self.blocks = nn.ModuleList([
            TransformerBlock(d, cfg.n_heads, cfg.head_dim, cfg.d_ff,
                             cfg.dropout, sd_list[i],
                             cfg.vmf_kappa_init, cfg.vmf_use_kappa_weights, rope)
            for i in range(L)])
        log(f'  {L} blocos  d={d}  nh={cfg.n_heads}  hd={cfg.head_dim}  '
            f'd_ff={cfg.d_ff}  +RoPE²+XOR+CondGate', 'ATTN')

        self.grafo = DynamicGrafoConnect(L, d)

        log('─── System-2: BeamSearch + IWR (MCTS removed) ───', 'SYS2')
        self.system2 = System2LatentSearch(
            K=cfg.K_think, d=d, n_heads=cfg.n_heads, hd=cfg.head_dim,
            d_ff=cfg.d_cot_ff, dp=cfg.dropout,
            branches=cfg.search_branches, max_iters=cfg.max_refinements,
            epsilon=cfg.refine_epsilon,
            n_actions=cfg.latent_actions,
            n_refinement_steps=cfg.n_refinement_steps,   # ✦ v26
            device=dev)

        self.final_norm  = RMSNorm(d)
        self.pool_fusion = PoolingFusion(d)
        self.head        = MultiSampleDropoutHead(d, cfg.n_classes,
                                                   cfg.dropout, cfg.msd_k)

    def encode(self, x: torch.Tensor, fmask: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tmask, curve = None, torch.zeros(B, device=x.device)
        if self.temporal_emb:
            temb, tmask, curve = self.temporal_emb(x)
            x = x + temb
        x = self.conv_mix(x)
        x = self.global_memory.expand_context(x, B)
        cls_hist = []
        ti = self.global_memory.slots
        for l, block in enumerate(self.blocks):
            delta = self.grafo(cls_hist, l, curve)
            if delta is not None:
                x = torch.cat([x[:, :ti],
                               x[:, ti:ti+1] + delta.unsqueeze(1),
                               x[:, ti+1:]], dim=1)
            x = block(x, fmask, tmask)
            cls_hist.append(x[:, ti].detach())
        x = self.global_memory.extract_and_update(x, B)
        x, _ = self.system2(x, fmask)
        K     = self.system2.K
        think = self.final_norm(x[:, :K]).mean(1)
        seq   = self.final_norm(x[:, K:])
        return self.pool_fusion(think, seq, (fmask == float('-inf')))

    def forward(self, ids: torch.Tensor, fmask: torch.Tensor) -> torch.Tensor:
        emb = self.embed_drop(self.embed(ids) * self.embed_scale)
        return self.head(self.encode(emb, fmask))

# ═══════════════════════════════════════════════════════════════════════
# PERDAS & REGULARIZAÇÃO
# ═══════════════════════════════════════════════════════════════════════
def focal_ce(logits: torch.Tensor, targets: torch.Tensor,
             gamma: float = 2.0, ls: float = 0.0) -> torch.Tensor:
    C    = logits.size(-1)
    logp = F.log_softmax(logits, -1)
    if ls > 0.0:
        s = torch.full_like(logp, ls / (C - 1))
        s.scatter_(-1, targets.unsqueeze(-1), 1.0 - ls)
        ce = -(s * logp).sum(-1)
    else:
        ce = F.nll_loss(logp, targets, reduction='none')
    if gamma == 0.0: return ce.mean()
    return ((1.0 - torch.exp(-ce.detach())).pow(gamma) * ce).mean()


def token_dropout(ids: torch.Tensor, fm: torch.Tensor, unk_id: int,
                  tp: float = 0.10, ap: float = 0.20) -> torch.Tensor:
    if torch.rand(1).item() > ap: return ids
    pm = (fm == float('-inf'))
    cm = torch.zeros_like(pm); cm[:, 0] = True
    drp  = (torch.rand_like(ids.float()) < tp) & (~pm) & (~cm)
    out  = ids.clone(); out[drp] = unk_id
    return out

# ═══════════════════════════════════════════════════════════════════════
# EMA, LOOKAHEAD, AWP, GRADCENTRALIZATION, LR SCHEDULER
# ═══════════════════════════════════════════════════════════════════════
class EMA:
    def __init__(self, model, decay=0.9995):
        self.decay  = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.lerp_(m.float(), 1.0 - self.decay)
        for sb, mb in zip(self.shadow.buffers(), model.buffers()):
            sb.copy_(mb)


class Lookahead(torch.optim.Optimizer):
    def __init__(self, base, k=6, alpha=0.5):
        self._b = base; self.k = k; self.alpha = alpha
        self._steps = 0; self._slow: dict = {}
        self.param_groups = base.param_groups
        self.defaults     = getattr(base, 'defaults', {})
    @property
    def state(self): return self._b.state
    def zero_grad(self, set_to_none=True): self._b.zero_grad(set_to_none=set_to_none)
    def _ensure(self):
        if self._slow: return
        for g in self.param_groups:
            for p in g['params']:
                self._slow[id(p)] = p.data.clone().detach()
    def step(self, closure=None):
        loss = self._b.step(closure); self._steps += 1; self._ensure()
        if self._steps % self.k == 0:
            for g in self.param_groups:
                for p in g['params']:
                    s = self._slow[id(p)]
                    s.add_(self.alpha * (p.data - s))
                    p.data.copy_(s)
        return loss


class AWP:
    def __init__(self, model, scaler, eps=0.005, lr=0.01):
        self.model  = model; self.scaler = scaler
        self.eps    = eps;   self.lr     = lr
        self._bk: dict = {}; self._on = False
    def perturb(self):
        if self._on: return
        sc = self.scaler.get_scale() if self.scaler.is_enabled() else 1.0
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                g  = p.grad.float() / (sc + 1e-8)
                gn = g.norm()
                if gn > 0 and torch.isfinite(gn):
                    self._bk[n] = p.data.clone()
                    p.data.add_((self.lr * g / (gn + 1e-8))
                                .clamp_(-self.eps, self.eps).to(p.dtype))
        self._on = True
    def restore(self):
        for n, p in self.model.named_parameters():
            if n in self._bk: p.data.copy_(self._bk[n])
        self._bk.clear(); self._on = False


def _gc_hook(g):
    return g - g.mean(tuple(range(1, g.dim())), keepdim=True) if g.dim() > 1 else g
def register_gc(model):
    return [p.register_hook(_gc_hook)
            for n, p in model.named_parameters()
            if p.requires_grad and p.dim() > 1 and 'embed' not in n]


class WarmupCosineLR:
    def __init__(self, total: int, wf: float, mf: float):
        self.T = max(total, 1); self.W = max(int(wf * total), 1); self.mf = mf
    def factor(self, step: int) -> float:
        if step < self.W: return step / self.W
        p = min(max((step - self.W) / max(self.T - self.W, 1), 0.0), 1.0)
        return self.mf + (1.0 - self.mf) * 0.5 * (1.0 + math.cos(math.pi * p))

# ═══════════════════════════════════════════════════════════════════════
# PIPELINE DE DADOS  (Yahoo Answers)
# ═══════════════════════════════════════════════════════════════════════
def build_or_load_tokenizer(texts: list, cfg: CFG) -> Tokenizer:
    if os.path.exists(cfg.tokenizer_path):
        tok = Tokenizer.from_file(cfg.tokenizer_path)
        if tok.get_vocab_size() >= cfg.tokenizer_vocab_size - 200:
            log(f'Tokenizer carregado ({tok.get_vocab_size()} tokens)')
            return tok
    log('Treinando tokenizer BPE 30k…', 'INFO')
    tok = Tokenizer(models.BPE(unk_token=cfg.unk_token))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder       = decoders.ByteLevel()
    tok.normalizer    = normalizers.NFKC()
    sp = [cfg.pad_token, cfg.cls_token, cfg.unk_token]
    trainer = trainers.BpeTrainer(vocab_size=cfg.tokenizer_vocab_size,
                                  special_tokens=sp, min_frequency=1)
    tok.train_from_iterator(texts, trainer=trainer)
    cls_id = tok.token_to_id(cfg.cls_token)
    tok.post_processor = TemplateProcessing(
        single=f'{cfg.cls_token} $A',
        special_tokens=[(cfg.cls_token, cls_id)])
    tok.save(cfg.tokenizer_path)
    log(f'Tokenizer guardado → {cfg.tokenizer_path}')
    return tok


class TextDataset(Dataset):
    def __init__(self, hf_ds, tok: Tokenizer, cfg: CFG):
        pad_id = tok.token_to_id(cfg.pad_token)
        self.samples = []
        for item in hf_ds:
            title       = str(item.get('question_title',   ''))
            content     = str(item.get('question_content', ''))
            best_answer = str(item.get('best_answer',      ''))
            # Concatenação completa — com max_len=192 capturamos muito mais contexto
            text = title + ' ' + content + ' ' + best_answer
            ids  = tok.encode(text[:6000]).ids[:cfg.max_len]
            vl   = len(ids)
            inp  = ids + [pad_id] * (cfg.max_len - vl)
            fm   = [0.0] * vl + [float('-inf')] * (cfg.max_len - vl)
            self.samples.append((
                torch.tensor(inp,                   dtype=torch.long),
                torch.tensor(fm,                    dtype=torch.float32),
                torch.tensor(int(item['topic']),    dtype=torch.long)))
    def __len__(self):        return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

# ═══════════════════════════════════════════════════════════════════════
# CICLO DE TREINO
# ═══════════════════════════════════════════════════════════════════════
def _reg_params(cfg: CFG, epoch: int):
    p    = min(1.0, epoch / cfg.epochs)
    mixp = cfg.mixup_prob_base   - (cfg.mixup_prob_base   - cfg.mixup_prob_min)   * p
    ls   = cfg.label_smooth_base - (cfg.label_smooth_base - cfg.label_smooth_min) * p
    return mixp, ls


def train_epoch(model, ema, optimizer, scaler, loader, awp, cfg, epoch,
                gstep, lr_sched, base_opt, qlr_mod, unk_id):
    model.train()
    mixp, ls = _reg_params(cfg, epoch)
    n = len(loader); t0 = time.time()
    st = {'loss': 0.0, 'ce': 0.0, 'cor': 0, 'tot': 0}
    optimizer.zero_grad(set_to_none=True)

    for step, (ids, fm, lbl) in enumerate(loader):
        ids = ids.to(cfg.device, non_blocking=True)
        fm  = fm.to(cfg.device,  non_blocking=True)
        lbl = lbl.to(cfg.device, non_blocking=True)
        ids = token_dropout(ids, fm, unk_id, cfg.token_dropout_prob,
                            cfg.token_dropout_apply)
        use_mx = random.random() < mixp

        with autocast(device_type=cfg.device.type, dtype=cfg.amp_dtype):
            if use_mx:
                emb  = model.embed_drop(model.embed(ids) * model.embed_scale)
                lam  = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                idx2 = torch.randperm(emb.size(0), device=cfg.device)
                logits = model.head(model.encode(
                    lam * emb + (1.0 - lam) * emb[idx2], fm))
                lp = F.log_softmax(logits, -1)
                C  = cfg.n_classes
                t1 = torch.full_like(lp, ls / (C - 1))
                t1.scatter_(-1, lbl.unsqueeze(-1), 1.0 - ls)
                t2 = torch.full_like(lp, ls / (C - 1))
                t2.scatter_(-1, lbl[idx2].unsqueeze(-1), 1.0 - ls)
                ce = (lam * (-(t1 * lp).sum(-1).mean())
                      + (1.0 - lam) * (-(t2 * lp).sum(-1).mean()))
            else:
                logits = model(ids, fm)
                ce = focal_ce(logits, lbl, cfg.focal_gamma, ls)

            div  = model.system2.diversity_loss() * cfg.div_weight
            loss = ce + div

        scaler.scale(loss / cfg.grad_accum).backward()

        if (step + 1) % cfg.grad_accum == 0:
            scaler.unscale_(optimizer)
            if epoch >= cfg.awp_start_ep:
                awp.perturb()
                with autocast(device_type=cfg.device.type, dtype=cfg.amp_dtype):
                    scaler.scale(
                        focal_ce(model(ids, fm), lbl, cfg.focal_gamma, ls)
                        / cfg.grad_accum).backward()
                awp.restore()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model); gstep += 1
            base_opt.param_groups[0]['lr'] = (
                cfg.base_lr_max * lr_sched.factor(gstep) * qlr_mod)

        st['loss'] += float(loss.item()); st['ce'] += float(ce.item())
        if not use_mx:
            st['cor'] += (logits.argmax(-1) == lbl).sum().item()
            st['tot'] += lbl.size(0)

        if step % cfg.attn_log_freq == 0 or step == n - 1:
            ela = time.time() - t0
            eta = ela / (step + 1) * (n - step - 1)
            log(f'ep={epoch:2d} step={step:04d}/{n} '
                f'lr={base_opt.param_groups[0]["lr"]:.6f} '
                f'loss={loss.item():.5f} ce={ce.item():.5f} '
                f'tr_acc={st["cor"]/max(st["tot"],1)*100:.2f}% '
                f'{ela:.1f}s ETA={eta:.1f}s', 'ATTN')

    return {
        'loss':  st['loss'] / n, 'ce': st['ce'] / n,
        'acc':   st['cor'] / max(st['tot'], 1),
        'time_s': round(time.time() - t0, 1),
    }, gstep


@torch.no_grad()
def evaluate(model, loader, cfg: CFG) -> dict:
    model.eval()
    cor = tot = 0; vl = 0.0
    for ids, fm, lbl in loader:
        ids = ids.to(cfg.device, non_blocking=True)
        fm  = fm.to(cfg.device,  non_blocking=True)
        lbl = lbl.to(cfg.device, non_blocking=True)
        with autocast(device_type=cfg.device.type, dtype=cfg.amp_dtype):
            logits = model(ids, fm)
        vl  += F.cross_entropy(logits, lbl).item() * lbl.size(0)
        cor += (logits.argmax(-1) == lbl).sum().item()
        tot += lbl.size(0)
    return {'acc': cor / max(tot, 1), 'loss': vl / max(tot, 1)}

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    cfg = CFG()
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    if HAS_RICH:
        console.rule(f'[bold green]GrafoPropagation {cfg.VERSION} · '
                     f'Yahoo Answers · Run {RUN_ID}[/bold green]')
    log(f'device={cfg.device}  amp={cfg.amp_dtype}  '
        f'max_len={cfg.max_len}  eff_batch={cfg.batch_size * cfg.grad_accum}')

    # ── Dados ────────────────────────────────────────────────────────
    log('Carregando Yahoo Answers…', 'INFO')
    raw    = load_dataset('yahoo_answers_topics')
    tr_raw = (raw['train'].shuffle(cfg.seed)
              .select(range(min(cfg.max_train, len(raw['train'])))))
    va_raw = (raw['test'].shuffle(cfg.seed)
              .select(range(min(cfg.max_val, len(raw['test'])))))

    def _get_text(item):
        t = str(item.get('question_title',   ''))
        c = str(item.get('question_content', ''))
        a = str(item.get('best_answer',      ''))
        return t + ' ' + c + ' ' + a

    texts  = [_get_text(r) for r in tr_raw]
    tok    = build_or_load_tokenizer(texts, cfg)
    unk_id = tok.token_to_id(cfg.unk_token)

    tr_ds = TextDataset(tr_raw, tok, cfg)
    va_ds = TextDataset(va_raw, tok, cfg)
    tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=4, pin_memory=True, drop_last=True,
                       persistent_workers=True)
    va_ld = DataLoader(va_ds, batch_size=128, shuffle=False,
                       num_workers=4, pin_memory=True, persistent_workers=True)
    log(f'Train batches={len(tr_ld)}  Val batches={len(va_ld)}  '
        f'eff_bs={cfg.batch_size * cfg.grad_accum}')

    # ── Modelo ───────────────────────────────────────────────────────
    model  = GrafoPropagation(cfg, tok).to(cfg.device)
    ema    = EMA(model, cfg.ema_decay)
    total  = sum(p.numel() for p in model.parameters())
    log(f'Parâmetros totais: {total:,}  ({total/1e6:.3f} M)', 'SYS2')

    # ── Optimizador ──────────────────────────────────────────────────
    base_opt  = torch.optim.AdamW(model.parameters(), lr=0.0,
                                  betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=cfg.wd)
    optimizer = Lookahead(base_opt, cfg.la_k, cfg.la_alpha)
    scaler    = GradScaler('cuda', enabled=(cfg.amp_dtype == torch.float16))
    awp       = AWP(model, scaler, cfg.awp_eps, cfg.awp_lr)
    gc_h      = register_gc(model)
    log(f'GradCentralization hooks: {len(gc_h)}')

    total_steps = cfg.epochs * (len(tr_ld) // cfg.grad_accum)
    lr_sched    = WarmupCosineLR(total_steps, cfg.warmup_frac, cfg.min_lr_frac)
    best_acc    = 0.0; gstep = 0; history = []

    # ── Loop de épocas ───────────────────────────────────────────────
    for epoch in range(1, cfg.epochs + 1):
        qlr = quantum_lr_modulation(epoch)
        log(f'EPOCH {epoch}/{cfg.epochs}  qlr_mod={qlr:.4f}')

        tr_s, gstep = train_epoch(
            model, ema, optimizer, scaler, tr_ld, awp, cfg, epoch,
            gstep, lr_sched, base_opt, qlr, unk_id)
        va_s = evaluate(ema.shadow, va_ld, cfg)

        if HAS_RICH:
            t = Table(title=f'Epoch {epoch} · {cfg.VERSION} · Yahoo Answers',
                      show_lines=True)
            t.add_column('Métrica',    style='bold',         width=28)
            t.add_column('Valor',      style='green',        width=36)
            t.add_row('Train Loss',    f'{tr_s["loss"]:.6f}')
            t.add_row('Train CE',      f'{tr_s["ce"]:.6f}')
            t.add_row('Train Acc',     f'{tr_s["acc"]*100:.3f}%')
            t.add_row('Val Acc (EMA)', f'[bold]{va_s["acc"]*100:.3f}%[/bold]')
            t.add_row('Val Loss',      f'{va_s["loss"]:.6f}')
            t.add_row('Best so far',   f'[bold green]{best_acc*100:.3f}%[/bold green]')
            t.add_row('Tempo epoch',   f'{tr_s["time_s"]:.1f}s')
            console.print(t)

        log(f'EPOCH_END {epoch}  val_acc={va_s["acc"]*100:.4f}%  '
            f'tr_acc={tr_s["acc"]*100:.4f}%  t={tr_s["time_s"]:.1f}s', 'METRIC')
        history.append({'epoch': epoch, 'train': tr_s, 'val': va_s})

        if va_s['acc'] > best_acc:
            best_acc = va_s['acc']
            torch.save({
                'model':   model.state_dict(),
                'ema':     ema.shadow.state_dict(),
                'epoch':   epoch,
                'acc':     best_acc,
                'run_id':  RUN_ID,
                'version': cfg.VERSION,
                'history': history,
            }, os.path.join(cfg.checkpoint_dir, 'best_model.pt'))
            log(f'✓ Novo melhor: {best_acc*100:.4f}%', 'METRIC')

        if epoch % cfg.checkpoint_every == 0:
            p = os.path.join(cfg.checkpoint_dir, f'ep{epoch:03d}.pt')
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'ema': ema.shadow.state_dict(), 'best_acc': best_acc,
                        'run_id': RUN_ID, 'history': history}, p)
            log(f'Checkpoint periódico: {p}', 'METRIC')

    for h in gc_h: h.remove()
    log(f'DONE  best_val_acc={best_acc*100:.4f}%', 'METRIC')
    hist_p = os.path.join(cfg.checkpoint_dir, f'history_{RUN_ID}.json')
    with open(hist_p, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    log(f'History → {hist_p}')


if __name__ == '__main__':
    main()
