#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════╗
║  GrafoPropagation v28.1-APEX³  ·  Sphere Navigation Edition  ·  GLUE Multi-Task║
║  Laboratory Edition  ·  A100 40 GB  ·  ULTRA ARCHITECTURE                     ║
╠════════════════════════════════════════════════════════════════════════════════╣
║  Integração de Estado da Arte (OTIMIZADA):                                    ║
║  [1] Meta-Learning (Reptile Phase) - AdamW Outer-Loop & Stratified Sampling.  ║
║  [2] Differentiable NAS (DARTS) - Procura automática de arquitetura em FF.    ║
║  [3] Dynamic Checkpoint Ensemble - Greedy Model Soups para maximizar métricas.║
║  [4] Dynamic Prompt Tuning - Reparametrização MLP & Penalização Ortogonal.    ║
║  [5] Layer-Wise Learning Rate Decay (LLRD) - Topologia exponencial de LR.     ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

import subprocess, sys, os, importlib, math, time, random, copy
import warnings, datetime, json, csv, zipfile
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
        'pennylane_lightning': 'pennylane-lightning',
        'datasets':            'datasets',
        'tokenizers':          'tokenizers',
    }
    missing = [p for m, p in needed.items() if not _ok(m)]
    if missing:
        print(f'[BOOT] Installing: {missing}')
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

torch.backends.cudnn.benchmark    = True
torch.set_float32_matmul_precision('high')

RUN_ID   = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
_LOG_BUF: list = []

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO v28.1-APEX³
# ═══════════════════════════════════════════════════════════════════════
class CFG:
    VERSION   = 'v28.1-APEX³-SPHERE-GLUE-ULTRA'
    seed      = 42
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_dtype = (torch.bfloat16
                 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                 else torch.float16)

    TASK_ORDER = ['mnli', 'qqp', 'qnli', 'sst2', 'cola', 'stsb', 'mrpc', 'rte', 'wnli']

    GLUE_TASKS = {
        'mnli': {'keys': ['premise', 'hypothesis'],   'classes': 3, 'type': 'clf'},
        'qqp':  {'keys': ['question1', 'question2'],  'classes': 2, 'type': 'clf'},
        'qnli': {'keys': ['question', 'sentence'],    'classes': 2, 'type': 'clf'},
        'sst2': {'keys': ['sentence'],                'classes': 2, 'type': 'clf'},
        'cola': {'keys': ['sentence'],                'classes': 2, 'type': 'clf'},
        'stsb': {'keys': ['sentence1', 'sentence2'],  'classes': 1, 'type': 'reg'},
        'mrpc': {'keys': ['sentence1', 'sentence2'],  'classes': 2, 'type': 'clf'},
        'rte':  {'keys': ['sentence1', 'sentence2'],  'classes': 2, 'type': 'clf'},
        'wnli': {'keys': ['sentence1', 'sentence2'],  'classes': 2, 'type': 'clf'},
    }

    tokenizer_vocab_size = 30_000
    tokenizer_path       = '/tmp/pico_tok_glue_30k_v28.json'
    pad_token = '[PAD]'; cls_token = '[CLS]'; unk_token = '[UNK]'
    max_len   = 128
    char_vocab = list(' abcdefghijklmnopqrstuvwxyz0123456789.,!?"\''
                      '()-/:;%&*$@')
    char_dim   = 48

    d_model     = 192
    n_layers    = 4
    n_heads     = 8
    head_dim    = 24
    d_ff        = 896
    dropout     = 0.15
    stoch_depth = 0.10
    conv_kernel = 3

    sphere_n_anchors    = 24
    sphere_dp           = 0.10
    sphere_anchor_div_w = 0.005

    memory_slots       = 12
    K_think            = 4
    d_cot_ff           = 512
    div_weight         = 0.01
    search_branches    = 3
    max_refinements    = 3
    refine_epsilon     = 1e-3
    latent_actions     = 6
    n_refinement_steps = 4

    use_temporal_transition     = True
    temporal_mlp_ratio          = 2.0
    temporal_modulate_attention = True
    temporal_n_features         = 6
    vmf_kappa_init              = 4.0
    vmf_use_kappa_weights       = True
    vmf_kappa_max               = 30.0
    rope_base                   = 10000.0
    rope_max_seq                = 512

    token_dropout_prob  = 0.10
    token_dropout_apply = 0.20
    focal_gamma         = 2.0
    mixup_alpha         = 0.20
    mixup_prob_base     = 0.25
    mixup_prob_min      = 0.04
    label_smooth_base   = 0.12
    label_smooth_min    = 0.03

    epochs_per_task = 10
    batch_size      = 96
    grad_accum      = 2
    wd              = 1e-4
    clip_grad       = 1.0

    base_lr_max         = 3e-4
    base_lr_min         = 1e-6
    warmup_steps        = 200
    cosine_T_0          = 10
    cosine_T_mult       = 2
    lr_plateau_patience = 2
    lr_plateau_factor   = 0.5
    lr_backbone         = 3e-4
    lr_embeddings       = 1e-4
    lr_heads            = 5e-4
    llrd_decay          = 0.85 # Factor de decaimento Layer-wise

    ewc_lambda         = 0.5
    ewc_fisher_samples = 2000
    replay_buffer_size = 500
    replay_prob        = 0.1

    ema_decay      = 0.9995
    awp_eps        = 0.005
    awp_lr         = 0.010
    awp_start_ep   = 3
    la_k           = 6
    la_alpha       = 0.50
    msd_k          = 5
    attn_log_freq  = 200
    checkpoint_dir = './ckpt_apex3_v28_ultra'
    
    # ── Ultra-Features Hyperparams ─────────────────────────────────────
    prompt_tuning_len   = 8        # Tamanho do Soft Prompt latente
    prompt_ortho_weight = 0.05     # Força vetores ortogonais entre prompts
    meta_warmup_steps   = 500      # Outer steps do Meta-Learning
    meta_inner_lr       = 1e-4     # Learning rate do meta-inner-loop
    meta_outer_lr       = 3e-4     # Reduzido para AdamW no outer loop
    meta_inner_steps    = 5        # Número de inner steps por task
    nas_lr              = 1e-3     # Learning rate da super-rede DARTS
    soup_k              = 3        # Top-K checkpoints para ensemble dinâmico


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
    'LB': 'bold white on blue', 'LAB': 'bold white on purple',
    'SPH': 'bold white on dark_green', 'ULTRA': 'bold white on red',
    'META': 'bold white on dark_red',
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
# TEMPORAL & RoPE²
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

    def _log_map(self, x_t, x_tp1):
        dot   = (x_t * x_tp1).sum(-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(dot)
        perp  = x_tp1 - dot * x_t
        sin_t = torch.sin(theta).clamp(min=1e-6)
        factor = theta / sin_t
        small  = (theta < 1e-4).float()
        factor = (1 - small) * factor + small * (1.0 + theta.pow(2) / 6.0)
        return factor * perp

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        xn = F.normalize(x.float(), p=2, dim=-1, eps=self.eps)
        cos_t = (xn[:, :-1] * xn[:, 1:]).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(cos_t)
        theta_padded = F.pad(theta, (1, 0), value=0.0)
        v_t_1  = self._log_map(xn[:, :-1], xn[:, 1:])
        v_zero = torch.zeros(B, 1, D, device=x.device, dtype=v_t_1.dtype)
        v_vectors = torch.cat([v_zero, v_t_1], dim=1)
        t_emb  = self.proj(v_vectors)
        t_mask = None
        if self.modulate:
            cs   = torch.cumsum(theta_padded, -1)
            dist = (cs.unsqueeze(-1) - cs.unsqueeze(-2)).abs()
            dec  = torch.clamp(self.temporal_decay, min=0.0)
            t_mask = (torch.exp(-dec * dist) + self.temporal_bias).to(x.dtype)
        omega = theta_padded[:, 1:] - theta_padded[:, :-1]
        omega = F.pad(omega, (1, 0), value=0.0)
        raw_k = (omega.abs() / (theta_padded * theta_padded + self.eps)).clamp(0, 20)
        curve = raw_k.mean(-1)
        return t_emb, t_mask, curve

class RoPERotator(nn.Module):
    def __init__(self, hd: int, max_len: int = 512, base: float = 10000.0):
        super().__init__()
        assert hd % 2 == 0
        inv = 1.0 / (base ** (torch.arange(0, hd // 2, dtype=torch.float32) / (hd // 2)))
        self.register_buffer('inv_freq', inv)
        self.hd    = hd
        self.alpha = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _rot(x: torch.Tensor) -> torch.Tensor:
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], -1)

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        B, T, H, D = mu.shape
        p     = torch.arange(T, dtype=mu.dtype, device=mu.device)
        inv   = self.inv_freq.to(mu.dtype)
        theta = (torch.outer(p, inv)
                 + self.alpha.to(mu.dtype) * torch.outer(p * (p - 1) * 0.5, inv))
        theta = torch.cat([theta, theta], dim=-1)
        cos_c = theta.cos().view(1, T, 1, D)
        sin_c = theta.sin().view(1, T, 1, D)
        return F.normalize(mu * cos_c + self._rot(mu) * sin_c, p=2, dim=-1, eps=1e-8)

# ═══════════════════════════════════════════════════════════════════════
# SPHERE NAVIGATOR
# ═══════════════════════════════════════════════════════════════════════
class SphereNavigator(nn.Module):
    def __init__(self, d: int, n_anchors: int = 24, dp: float = 0.10):
        super().__init__()
        self.d        = d
        self.n_anchors = n_anchors
        self.eps      = 1e-8

        anchors = F.normalize(torch.randn(n_anchors, d), dim=-1)
        anchors = F.normalize(anchors + 0.01 * torch.randn_like(anchors), dim=-1)
        self.anchors = nn.Parameter(anchors)

        self.temp = nn.Parameter(torch.tensor(4.0))

        self.compass = nn.Sequential(
            nn.Linear(d + n_anchors, d, bias=False),
            RMSNorm(d),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dp),
            nn.Linear(d, d, bias=False),
        )
        nn.init.xavier_uniform_(self.compass[0].weight, gain=0.3)
        nn.init.xavier_uniform_(self.compass[4].weight, gain=0.1)

        self.step_scale      = nn.Parameter(torch.tensor(0.0))
        self.zone_bias_scale = nn.Parameter(torch.tensor(0.0))

        self.density_head = nn.Linear(d, 1, bias=True)
        nn.init.zeros_(self.density_head.weight)
        nn.init.zeros_(self.density_head.bias)

        self.norm = RMSNorm(d)
        self.drop = nn.Dropout(dp)

    def anchor_diversity_loss(self) -> torch.Tensor:
        an   = F.normalize(self.anchors.float(), dim=-1)
        sim  = an @ an.T                                      
        mask = 1.0 - torch.eye(self.n_anchors, device=an.device)
        return (sim.pow(2) * mask).sum() / (self.n_anchors * (self.n_anchors - 1))

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        x_n  = F.normalize(x.float(), dim=-1, eps=self.eps)           
        an_n = F.normalize(self.anchors.float(), dim=-1, eps=self.eps) 
        temp = F.softplus(self.temp).clamp(min=0.5, max=20.0)
        cos_sims  = torch.einsum('btd,kd->btk', x_n, an_n)            
        zone_w    = F.softmax(cos_sims * temp, dim=-1)                 
        zone_w_t  = zone_w.to(x.dtype)

        compass_in = torch.cat([x, zone_w_t], dim=-1)                  
        direction  = self.compass(compass_in)                           
        dir_n      = F.normalize(direction.float(), dim=-1, eps=self.eps)

        dot        = (x_n * dir_n).sum(-1, keepdim=True)               
        tangent    = dir_n - dot * x_n                                  

        step = torch.tanh(self.step_scale) * 0.2
        x_nav  = F.normalize(x_n + step * tangent, dim=-1, eps=self.eps).to(x.dtype)

        mag    = x.float().norm(dim=-1, keepdim=True).clamp(min=self.eps).to(x.dtype)
        x_nav  = x_nav * mag

        delta  = self.norm(self.drop(x_nav - x))
        x_out  = x + delta

        zone_sim   = torch.bmm(zone_w_t, zone_w_t.transpose(1, 2))    
        bias_scale = torch.tanh(self.zone_bias_scale)                  
        zone_bias  = bias_scale * zone_sim                             

        density = F.softplus(self.density_head(x).squeeze(-1)) + 0.1  

        return x_out, zone_bias, density

# ═══════════════════════════════════════════════════════════════════════
# vMF ATTENTION
# ═══════════════════════════════════════════════════════════════════════
class VonMisesFisherAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, hd: int, dropout: float = 0.1,
                 kappa_init: float = 4.0, use_kappa_weights: bool = True,
                 rope=None):
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
        self.xor_weight = nn.Parameter(torch.zeros(1))
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
        return torch.clamp(F.softplus(self.W_kappa(x)) + 1e-4, max=CFG.vmf_kappa_max)

    def forward(self, x: torch.Tensor, fmask=None, tmask=None,
                zone_bias=None) -> torch.Tensor:
        B, T, D = x.shape
        mu_normed = F.normalize(
            self.W_mu(x).view(B, T, self.n_heads, self.hd),
            p=2, dim=-1, eps=self.eps)
        mu = self.rope(mu_normed) if self.rope is not None else mu_normed
        kappa  = self.get_kappa(x)
        mu_h   = mu.permute(0, 2, 1, 3)
        mu_h_s = mu_normed.permute(0, 2, 1, 3)

        S = torch.matmul(mu_h, mu_h.transpose(-2, -1))
        if self.ukw:
            kh = kappa.permute(0, 2, 1)
            S  = torch.sqrt(kh.unsqueeze(-1) * kh.unsqueeze(-2) + self.eps) * S
        scores = (self.tau.view(1, self.n_heads, 1, 1) * (S * self._sc)
                  + self.bias_q.view(1, self.n_heads, 1, 1))

        soft_sign = (2.0 * torch.sigmoid(mu_h_s.float() * 5.0) - 1.0).to(mu_h.dtype)
        xor_sim   = torch.matmul(soft_sign, soft_sign.transpose(-2, -1)) / self.hd
        scores    = scores + torch.tanh(self.xor_weight) * xor_sim

        if zone_bias is not None:
            scores = scores + zone_bias.unsqueeze(1) 

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
        av   = torch.matmul(attn, v)
        gate = torch.sigmoid(
            self.W_gate(x).view(B, T, self.n_heads, self.hd).permute(0, 2, 1, 3))
        cond_dir_n = F.normalize(self.cond_dir.to(x.dtype), dim=-1)
        cond_proj  = (mu_h_s * cond_dir_n.view(1, self.n_heads, 1, self.hd)).sum(-1, keepdim=True)
        scale_c    = self.cond_scale.view(1, self.n_heads, 1, 1).to(x.dtype)
        condition  = torch.sigmoid(cond_proj * scale_c)
        gate_cond  = condition * gate + (1.0 - condition) * (1.0 - gate)

        out = (gate_cond * av).permute(0, 2, 1, 3).reshape(B, T, self.n_heads * self.hd)
        return self.Wo(out)

# ═══════════════════════════════════════════════════════════════════════
# MIXERS, NAS & BLOCKS
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

class DartsNASBlock(nn.Module):
    """Procura automática de arquitetura em tempo real para blocos Feed-Forward"""
    def __init__(self, d: int, dff: int, dp: float = 0.1):
        super().__init__()
        self.op1 = SwiGLU(d, dff, dp)
        self.op2 = LocalConvMix(d, 5, dp)
        self.op3 = nn.Identity()
        # Parâmetros estruturais
        self.arch_alphas = nn.Parameter(torch.zeros(3))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.arch_alphas, dim=0)
        return weights[0] * self.op1(x) + weights[1] * self.op2(x) + weights[2] * self.op3(x)

class TransformerBlock(nn.Module):
    def __init__(self, d: int, nh: int, hd: int, dff: int,
                 dp: float = 0.1, sd: float = 0.0,
                 ki: float = 4.0, ukw: bool = True, rope=None):
        super().__init__()
        self.sd    = sd
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.attn  = VonMisesFisherAttention(d, nh, hd, dp, ki, ukw, rope)
        # Substituímos a FFN clássica pela NAS block do laboratório
        self.ffn   = DartsNASBlock(d, dff, dp)

    def _drop(self, r: torch.Tensor) -> torch.Tensor:
        if not self.training or self.sd == 0.0: return r
        keep = (torch.rand(r.shape[0], 1, 1, device=r.device) > self.sd).float()
        return r * keep / (1.0 - self.sd)

    def forward(self, x: torch.Tensor, fmask=None, tmask=None,
                zone_bias=None) -> torch.Tensor:
        x = x + self._drop(self.attn(self.norm1(x), fmask, tmask, zone_bias))
        return x + self._drop(self.ffn(self.norm2(x)))

# ═══════════════════════════════════════════════════════════════════════
# GLOBAL WORKSPACE, GRAFO & SYSTEM-2
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

    def forward(self, hist: list, tgt: int, curve: torch.Tensor):
        n = len(hist)
        if n == 0: return None
        impact  = torch.sigmoid(self.time_mod(curve.unsqueeze(-1)))
        dyn_A   = self.A[tgt, :n].unsqueeze(0) * (1.0 + impact[:, :n])
        w       = F.softmax(dyn_A, dim=-1)
        agg     = sum(self.src[i](hist[i]) * w[:, i].unsqueeze(-1) for i in range(n))
        g       = torch.sigmoid(self.gate(torch.cat([hist[-1], agg], dim=-1)))
        return self.norm(g * agg)

class ResidualWorldModel(nn.Module):
    def __init__(self, d, n_actions, n_heads, hd, d_ff, dp):
        super().__init__()
        self.n_actions  = n_actions
        self.input_norm = RMSNorm(d)
        self.action_W   = nn.Parameter(torch.empty(n_actions, d, d))
        for a in range(n_actions):
            nn.init.xavier_uniform_(self.action_W[a], gain=0.3)
        self.action_scale = nn.Parameter(torch.full((n_actions,), 0.1))
        self.post_norm    = RMSNorm(d)
        self.coherence    = TransformerBlock(d, n_heads, hd, d_ff, dp, 0.0,
                                             CFG.vmf_kappa_init, CFG.vmf_use_kappa_weights, rope=None)

    def forward(self, state, action_idx):
        B, K, D = state.shape
        normed  = self.input_norm(state)
        W       = self.action_W[action_idx]
        scale   = self.action_scale[action_idx].view(B, 1, 1)
        delta   = torch.bmm(normed, W.transpose(-1, -2)) * scale
        out     = self.post_norm(state + delta)
        return self.coherence(out)

class PolicyValueHead(nn.Module):
    def __init__(self, d, n_actions):
        super().__init__()
        self.shared      = nn.Sequential(nn.Linear(d, d), RMSNorm(d), nn.GELU(approximate='tanh'))
        self.policy_head = nn.Linear(d, n_actions)
        self.value_head  = nn.Linear(d, 1)
        nn.init.xavier_uniform_(self.policy_head.weight)
        nn.init.xavier_uniform_(self.value_head.weight)

    def forward(self, state):
        h  = self.shared(state.mean(dim=1))
        pi = F.softmax(self.policy_head(h), dim=-1)
        v  = torch.tanh(self.value_head(h))
        return pi, v

class IterativeWorldRefinement(nn.Module):
    def __init__(self, world_model, actor_critic, n_actions, n_steps, device):
        super().__init__()
        self.world_model  = world_model
        self.actor_critic = actor_critic
        self.n_actions    = n_actions
        self.n_steps      = n_steps
        self.device       = device
        self.temperature  = nn.Parameter(torch.ones(1))

    def forward(self, state):
        B, K, D = state.shape
        cur = state; last_pi = None
        for _ in range(self.n_steps):
            pi, _ = self.actor_critic(cur)
            temp  = F.softplus(self.temperature).clamp(min=0.1)
            pi_s  = F.softmax(pi / temp, dim=-1)
            cur_exp = cur.unsqueeze(1).expand(B, self.n_actions, K, D).reshape(B * self.n_actions, K, D)
            a_idx   = torch.arange(self.n_actions, device=self.device).unsqueeze(0).expand(B, -1).reshape(-1)
            children = self.world_model(cur_exp, a_idx).view(B, self.n_actions, K, D)
            cur    = (children * pi_s.unsqueeze(-1).unsqueeze(-1)).sum(1)
            last_pi = pi_s
        final_pi = (last_pi if last_pi is not None
                    else torch.full((B, self.n_actions), 1.0 / self.n_actions, device=self.device))
        return cur, final_pi

class System2LatentSearch(nn.Module):
    def __init__(self, K, d, n_heads, hd, d_ff, dp, branches, max_iters,
                 epsilon, n_actions, n_refinement_steps, device):
        super().__init__()
        self.K = K; self.branches = branches
        self.max_iters = max_iters; self.eps = epsilon
        self.base_think   = nn.Parameter(torch.empty(1, K, d))
        _trunc_normal_(self.base_think, std=0.02)
        self.branch_noise = nn.Parameter(torch.randn(branches, 1, K, d) * 0.05)
        self.search_block = TransformerBlock(d, n_heads, hd, d_ff, dp, 0.0,
                                             CFG.vmf_kappa_init, CFG.vmf_use_kappa_weights, rope=None)
        self.norm         = RMSNorm(d)
        self.world_model  = ResidualWorldModel(d, n_actions, n_heads, hd, d_ff, dp)
        self.actor_critic = PolicyValueHead(d, n_actions)
        self.refinement   = IterativeWorldRefinement(self.world_model, self.actor_critic,
                                                      n_actions, n_refinement_steps, device)

    def diversity_loss(self) -> torch.Tensor:
        if self.K <= 1: return self.base_think.new_zeros(())
        t = F.normalize(self.base_think.squeeze(0), dim=-1)
        return (t @ t.T).triu(1).pow(2).sum() * (2.0 / (self.K * (self.K - 1)))

    def forward(self, seq_x, fmask=None):
        B, T, D = seq_x.shape
        ext_mask = None
        if fmask is not None:
            ext_mask = torch.cat([torch.zeros(B, self.K, device=fmask.device, dtype=fmask.dtype), fmask], dim=1)
        best = self.base_think.expand(B, -1, -1)
        for it in range(self.max_iters):
            prev   = best.detach()
            branch_t = best.unsqueeze(0) + self.branch_noise
            evals, thoughts = [], []
            for b in range(self.branches):
                h    = torch.cat([branch_t[b], seq_x], dim=1)
                hout = self.search_block(h, ext_mask)
                th   = self.norm(hout[:, :self.K])
                ksc  = self.search_block.attn.get_kappa(hout[:, :self.K]).mean([1, 2])
                evals.append(ksc); thoughts.append(th)
            w    = F.softmax(torch.stack(evals, 0), 0)
            best = (torch.stack(thoughts, 0) * w.unsqueeze(-1).unsqueeze(-1)).sum(0)
            if torch.norm(best - prev, dim=-1).mean() < self.eps and it > 0: break
        refined, _ = self.refinement(best)
        return torch.cat([refined, seq_x], dim=1), ext_mask

class PoolingFusion(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.gate = nn.Linear(2 * d, d, bias=True)
        self.norm = RMSNorm(d)
        nn.init.xavier_uniform_(self.gate.weight, gain=0.5)
        nn.init.zeros_(self.gate.bias)

    def forward(self, think, seq, pad_mask):
        valid   = (~pad_mask).to(seq.dtype).unsqueeze(-1)
        seq_avg = (seq * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        g = torch.sigmoid(self.gate(torch.cat([think, seq_avg], -1)))
        return self.norm(g * think + (1.0 - g) * seq_avg)

class MultiSampleDropoutHead(nn.Module):
    def __init__(self, d: int, nc: int, dp: float = 0.1, k: int = 5, is_reg: bool = False):
        super().__init__()
        self.k = k; self.dp = dp; self.is_reg = is_reg
        self.fc1 = nn.Linear(d, d)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(d, nc)
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.02); nn.init.zeros_(self.fc2.bias)

    def _single(self, x):
        return self.fc2(F.dropout(self.act(self.fc1(x)), p=self.dp, training=True))

    def forward(self, x):
        if self.training:
            out = torch.stack([self._single(x) for _ in range(self.k)], dim=0).mean(0)
        else:
            out = self.fc2(self.act(self.fc1(x)))
        return out.squeeze(-1) if self.is_reg else out

# ═══════════════════════════════════════════════════════════════════════
# AUTOMATIC PROMPT TUNING MODULE (OTIMIZADO)
# ═══════════════════════════════════════════════════════════════════════
class DynamicOrthogonalPromptTuning(nn.Module):
    """Soft Prompts Contínuos com Reparametrização MLP e Penalização Ortogonal (P-Tuning v2 style)"""
    def __init__(self, cfg: CFG):
        super().__init__()
        self.length = cfg.prompt_tuning_len
        self.prompt_embeddings = nn.ParameterDict({
            task: nn.Parameter(torch.randn(self.length, cfg.d_model) * 0.02)
            for task in cfg.TASK_ORDER
        })
        # MLP de reparametrização para estabilizar a injeção condicional
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            RMSNorm(cfg.d_model * 2),
            nn.GELU(approximate='tanh'),
            nn.Linear(cfg.d_model * 2, cfg.d_model)
        )
        
    def get_orthogonal_loss(self) -> torch.Tensor:
        """Força as prompts latentes de cada task a serem ortogonais entre si."""
        loss = 0.0
        tasks = list(self.prompt_embeddings.keys())
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                p_i = self.prompt_embeddings[tasks[i]].mean(0)
                p_j = self.prompt_embeddings[tasks[j]].mean(0)
                sim = F.cosine_similarity(p_i, p_j, dim=0)
                loss += sim.pow(2)
        return loss

    def forward(self, emb, task, fmask):
        B = emb.size(0)
        raw_p = self.prompt_embeddings[task]
        p = self.mlp(raw_p).unsqueeze(0).expand(B, -1, -1)
        
        # Pad the fmask accordingly (prompts are always valid info, i.e., 0.0)
        if fmask is not None:
            p_mask = torch.zeros(B, self.length, device=fmask.device, dtype=fmask.dtype)
            fmask = torch.cat([p_mask, fmask], dim=1)
        return torch.cat([p, emb], dim=1), fmask

# ═══════════════════════════════════════════════════════════════════════
# GRAFOPROPAGATION MULTI-TASK  v28.1 ULTRA
# ═══════════════════════════════════════════════════════════════════════
class GrafoPropagationMT(nn.Module):
    def __init__(self, cfg: CFG, tokenizer):
        super().__init__()
        d, L, dev = cfg.d_model, cfg.n_layers, cfg.device

        char_emb = build_character_embeddings(cfg.char_vocab, cfg.char_dim, dev)
        self.register_buffer('char_emb_buf', char_emb)
        self.char_proj = nn.Linear(cfg.char_dim, d, bias=False).to(dev)
        nn.init.xavier_uniform_(self.char_proj.weight)

        vocab    = tokenizer.get_vocab()
        tok2str  = {v: k for k, v in vocab.items()}
        char2idx = {ch: i for i, ch in enumerate(cfg.char_vocab)}
        tvecs    = torch.zeros(cfg.tokenizer_vocab_size, cfg.char_dim, device=dev)
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
        
        # O novo módulo Otimizado de Prompt Tuning Ortogonal
        self.task_prompts = DynamicOrthogonalPromptTuning(cfg)

        self.temporal_emb = (
            TemporalTransitionEmbedding(d, cfg.temporal_n_features,
                                        cfg.temporal_mlp_ratio,
                                        cfg.temporal_modulate_attention)
            if cfg.use_temporal_transition else None)

        self.conv_mix      = LocalConvMix(d, cfg.conv_kernel, cfg.dropout)
        self.global_memory = GlobalWorkspaceMemory(cfg.memory_slots, d)
        self.sphere_nav    = SphereNavigator(d, cfg.sphere_n_anchors, cfg.sphere_dp)

        rope = RoPERotator(cfg.head_dim, cfg.rope_max_seq, cfg.rope_base).to(dev)
        sd_list = [cfg.stoch_depth * i / max(L - 1, 1) for i in range(L)]
        self.blocks = nn.ModuleList([
            TransformerBlock(d, cfg.n_heads, cfg.head_dim, cfg.d_ff,
                             cfg.dropout, sd_list[i],
                             CFG.vmf_kappa_init, CFG.vmf_use_kappa_weights, rope)
            for i in range(L)])

        self.grafo   = DynamicGrafoConnect(L, d)
        self.system2 = System2LatentSearch(
            K=cfg.K_think, d=d, n_heads=cfg.n_heads, hd=cfg.head_dim,
            d_ff=cfg.d_cot_ff, dp=cfg.dropout,
            branches=cfg.search_branches, max_iters=cfg.max_refinements,
            epsilon=cfg.refine_epsilon, n_actions=cfg.latent_actions,
            n_refinement_steps=cfg.n_refinement_steps, device=dev)

        self.final_norm  = RMSNorm(d)
        self.pool_fusion = PoolingFusion(d)

        self.heads = nn.ModuleDict({
            task: MultiSampleDropoutHead(
                d, info['classes'], cfg.dropout, cfg.msd_k,
                is_reg=(info['type'] == 'reg')
            ) for task, info in cfg.GLUE_TASKS.items()
        })
        
        self.prompt_len = cfg.prompt_tuning_len

    def encode(self, x: torch.Tensor, fmask: torch.Tensor, task: str) -> torch.Tensor:
        B = x.shape[0]
        
        # 1. Injetamos o Soft Prompt contínuo reparametrizado
        x, fmask = self.task_prompts(x, task, fmask)
        
        tmask, curve = None, torch.zeros(B, device=x.device)

        if self.temporal_emb:
            temb, tmask, curve = self.temporal_emb(x)
            x = x + temb

        x = self.conv_mix(x)
        x, zone_bias, _density = self.sphere_nav(x)

        x = self.global_memory.expand_context(x, B)
        slots = self.global_memory.slots
        zone_bias_full = F.pad(zone_bias, (slots, 0, slots, 0))

        cls_hist = []
        ti = slots
        for l, block in enumerate(self.blocks):
            delta = self.grafo(cls_hist, l, curve)
            if delta is not None:
                x = torch.cat([
                    x[:, :ti],
                    x[:, ti:ti+1] + delta.unsqueeze(1),
                    x[:, ti+1:]
                ], dim=1)
            x = block(x, fmask, tmask, zone_bias_full)
            cls_hist.append(x[:, ti].detach())

        x = self.global_memory.extract_and_update(x, B)
        x, _ = self.system2(x, fmask)
        K     = self.system2.K
        think = self.final_norm(x[:, :K]).mean(1)
        
        # Ignoramos o soft prompt no average pooling final
        seq_body = self.final_norm(x[:, K + self.prompt_len:])
        pad_mask = (fmask[:, self.prompt_len:] == float('-inf'))
        
        return self.pool_fusion(think, seq_body, pad_mask)

    def forward(self, ids: torch.Tensor, fmask: torch.Tensor, task: str) -> torch.Tensor:
        emb = self.embed_drop(self.embed(ids) * self.embed_scale)
        return self.heads[task](self.encode(emb, fmask, task))

# ═══════════════════════════════════════════════════════════════════════
# PERDAS & MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════
def focal_ce(logits: torch.Tensor, targets: torch.Tensor,
             gamma: float = 2.0, ls: float = 0.0) -> torch.Tensor:
    C    = logits.size(-1)
    logp = F.log_softmax(logits, -1)
    if ls > 0.0:
        s = torch.full_like(logp, ls / max(C - 1, 1))
        s.scatter_(-1, targets.unsqueeze(-1), 1.0 - ls)
        ce = -(s * logp).sum(-1)
    else:
        ce = F.nll_loss(logp, targets, reduction='none')
    if gamma == 0.0: return ce.mean()
    return ((1.0 - torch.exp(-ce.detach())).pow(gamma) * ce).mean()

def pearson_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if preds.numel() <= 1:
        return F.mse_loss(preds, targets)
    p = preds  - preds.mean()
    t = targets - targets.mean()
    num   = (p * t).sum()
    denom = (p.pow(2).sum() * t.pow(2).sum()).sqrt().clamp(min=1e-8)
    return 1.0 - num / denom

def regression_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return 0.5 * F.mse_loss(preds, targets) + 0.5 * pearson_loss(preds, targets)

def compute_task_metric(preds: list, labels: list, task: str) -> float:
    p = np.array(preds,  dtype=np.float32)
    l = np.array(labels, dtype=np.float32)

    if task == 'cola':
        pi = (p > 0.5).astype(float)
        tp = np.sum((pi == 1) & (l == 1))
        tn = np.sum((pi == 0) & (l == 0))
        fp = np.sum((pi == 1) & (l == 0))
        fn = np.sum((pi == 0) & (l == 1))
        denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        return float((tp*tn - fp*fn) / max(denom, 1.0))
    elif task == 'stsb':
        p_c = p - p.mean()
        l_c = l - l.mean()
        num  = float(np.dot(p_c, l_c))
        denom = float(np.linalg.norm(p_c) * np.linalg.norm(l_c))
        return num / max(denom, 1e-8)
    elif task in ('mrpc', 'qqp'):
        pi = np.round(p).astype(float)
        tp = np.sum((pi == 1) & (l == 1))
        fp = np.sum((pi == 1) & (l == 0))
        fn = np.sum((pi == 0) & (l == 1))
        prec = float(tp / max(tp + fp, 1))
        rec  = float(tp / max(tp + fn, 1))
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        acc  = float(np.mean(np.round(p) == l))
        return (f1 + acc) / 2.0
    else:
        return float(np.mean(np.round(p) == l))

def slerp_mixup(emb1: torch.Tensor, emb2: torch.Tensor, lam: float) -> torch.Tensor:
    eps = 1e-8
    e1 = emb1.float(); e2 = emb2.float()
    e1_n = F.normalize(e1, dim=-1, eps=eps)
    e2_n = F.normalize(e2, dim=-1, eps=eps)
    dot   = (e1_n * e2_n).sum(-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(dot)                     
    sin_t = torch.sin(theta).clamp(min=eps)
    w1 = torch.sin((1.0 - lam) * theta) / sin_t
    w2 = torch.sin(lam * theta) / sin_t
    parallel = (theta.abs() < 1e-4).float()
    w1 = (1.0 - parallel) * w1 + parallel * (1.0 - lam)
    w2 = (1.0 - parallel) * w2 + parallel * lam
    mix = w1 * e1_n + w2 * e2_n
    mag = (1.0 - lam) * e1.norm(dim=-1, keepdim=True) + lam * e2.norm(dim=-1, keepdim=True)
    return (F.normalize(mix, dim=-1, eps=eps) * mag).to(emb1.dtype)

def token_dropout(ids, fm, unk_id, tp=0.10, ap=0.20):
    if torch.rand(1).item() > ap: return ids
    pm  = (fm == float('-inf'))
    cm  = torch.zeros_like(pm); cm[:, 0] = True
    drp = (torch.rand_like(ids.float()) < tp) & (~pm) & (~cm)
    out = ids.clone(); out[drp] = unk_id
    return out

# ═══════════════════════════════════════════════════════════════════════
# UTILS DE TREINO & ENSEMBLE
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

class Lookahead:
    def __init__(self, base, k=6, alpha=0.5):
        self._b    = base
        self.k     = k
        self.alpha = alpha
        self._steps = 0
        self._slow: dict = {}
        self.param_groups = base.param_groups
        self.defaults     = getattr(base, 'defaults', {})

    @property
    def state(self): return self._b.state

    def zero_grad(self, set_to_none=True):
        self._b.zero_grad(set_to_none=set_to_none)

    def _ensure(self):
        if self._slow: return
        for g in self.param_groups:
            for p in g['params']:
                self._slow[id(p)] = p.data.clone().detach()

    def _sync(self):
        self._steps += 1
        self._ensure()
        if self._steps % self.k == 0:
            for g in self.param_groups:
                for p in g['params']:
                    s = self._slow[id(p)]
                    s.add_(self.alpha * (p.data - s))
                    p.data.copy_(s)

    def step(self, closure=None):
        loss = self._b.step(closure)
        self._sync()
        return loss

    def state_dict(self):
        return {
            'base_state_dict': self._b.state_dict(),
            'slow': {k: v.cpu().clone() for k, v in self._slow.items()},
            'steps': self._steps
        }

    def load_state_dict(self, sd):
        self._b.load_state_dict(sd['base_state_dict'])
        dev = self._b.param_groups[0]['params'][0].device
        self._slow  = {k: v.to(dev) for k, v in sd['slow'].items()}
        self._steps = sd['steps']

class AWP:
    def __init__(self, model, eps=0.005, lr=0.01):
        self.model = model
        self.eps   = eps
        self.lr    = lr
        self._bk: dict = {}
        self._on = False

    def perturb(self):
        if self._on: return
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                g  = p.grad.float()
                gn = g.norm()
                if gn > 0 and torch.isfinite(gn):
                    self._bk[n] = p.data.clone()
                    delta = (self.lr * g / (gn + 1e-8)).clamp_(-self.eps, self.eps).to(p.dtype)
                    p.data.add_(delta)
        self._on = True

    def restore(self):
        for n, p in self.model.named_parameters():
            if n in self._bk: p.data.copy_(self._bk[n])
        self._bk.clear(); self._on = False

class DynamicGreedySoup:
    """Ensemble Dinâmico de Top-K Checkpoints durante a Task."""
    def __init__(self, k=3):
        self.k = k
        self.ckpts = []

    def add(self, metric: float, model_state: dict):
        self.ckpts.append((metric, copy.deepcopy(model_state)))
        self.ckpts.sort(key=lambda x: x[0], reverse=True)
        if len(self.ckpts) > self.k:
            self.ckpts.pop()
            
    def merge(self) -> dict:
        if not self.ckpts: return None
        merged = copy.deepcopy(self.ckpts[0][1])
        for metric, state in self.ckpts[1:]:
            for k in merged.keys():
                if merged[k].dtype.is_floating_point:
                    merged[k] += state[k]
        n = len(self.ckpts)
        for k in merged.keys():
            if merged[k].dtype.is_floating_point:
                merged[k] /= n
        return merged

class LaboratoryLRScheduler:
    def __init__(self, optimizer, cfg: CFG, steps_per_epoch: int):
        self.optimizer       = optimizer
        self.cfg             = cfg
        self.steps_per_epoch = steps_per_epoch
        self.current_step    = 0
        self.current_epoch   = 0
        self.best_metric     = -float('inf')
        self.patience_counter = 0
        self.T_cur           = cfg.cosine_T_0 * steps_per_epoch
        self.cycle_step      = 0
        self.min_lr          = cfg.base_lr_min
        self.base_lr         = cfg.base_lr_max
        self.base_lrs        = [g['lr'] for g in optimizer.param_groups]

    def _warmup_factor(self) -> float:
        if self.current_step < self.cfg.warmup_steps:
            return self.current_step / max(self.cfg.warmup_steps, 1)
        return 1.0

    def _cosine_factor(self) -> float:
        if self.T_cur <= 0: return 0.0
        frac = min(self.cycle_step / self.T_cur, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * frac))

    def step_epoch(self, metric: float = None):
        self.current_epoch += 1
        if metric is not None:
            if metric > self.best_metric + 1e-4:
                self.best_metric = metric
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            if self.patience_counter >= self.cfg.lr_plateau_patience:
                old = self.base_lr
                self.base_lr = max(self.base_lr * self.cfg.lr_plateau_factor, self.min_lr)
                log(f'[LAB] Plateau → LR base {old:.2e} → {self.base_lr:.2e}', 'LAB')
                self.patience_counter = 0
                self.T_cur  = self.cfg.cosine_T_0 * self.steps_per_epoch
                self.cycle_step = 0
        if self.cycle_step >= self.T_cur:
            self.T_cur *= self.cfg.cosine_T_mult
            self.cycle_step = 0
            log(f'[LAB] Cosine restart → período {self.T_cur / self.steps_per_epoch:.1f} épocas', 'LAB')

    def step_batch(self):
        self.current_step += 1
        self.cycle_step   += 1
        wf = self._warmup_factor()
        cf = self._cosine_factor()
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if group.get('layer_type', '') == 'nas_alphas':
                group['lr'] = self.cfg.nas_lr
                continue
            # Aplica o warmup e cosine decay relativo ao base_lr da layer (respeitando o LLRD já calculado)
            group['lr'] = max(base_lr * wf * cf, self.min_lr)

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

class ElasticWeightConsolidation:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model         = model
        self.device        = device
        self.task_memories = []

    @torch.no_grad()
    def compute_fisher(self, dataloader: DataLoader, task: str, num_samples: int = 2000):
        log(f'[EWC] Calculando Fisher para task {task} (n={num_samples})…', 'LAB')
        self.model.eval()
        fisher = {}; params = {}; n_accum = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param, device=self.device)
                params[name] = param.data.clone()
        for ids, fm, lbl in dataloader:
            if n_accum >= num_samples: break
            ids = ids.to(self.device); fm = fm.to(self.device); lbl = lbl.to(self.device)
            self.model.zero_grad()
            logits = self.model(ids, fm, task)
            is_reg = CFG.GLUE_TASKS[task]['type'] == 'reg'
            loss = (regression_loss(logits.squeeze(-1), lbl.float()) if is_reg
                    else F.cross_entropy(logits, lbl))
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)
            n_accum += ids.size(0)
        if n_accum == 0: return
        for name in fisher: fisher[name] /= n_accum
        self.task_memories.append({'task': task, 'fisher': fisher, 'params': params})
        log(f'[EWC] Fisher OK para {task} ({len(self.task_memories)} tasks em memória)', 'LAB')

    def penalty(self, model: nn.Module, lambda_ewc: float) -> torch.Tensor:
        if not self.task_memories:
            return torch.tensor(0.0, device=self.device)
        total = torch.tensor(0.0, device=self.device)
        for mem in self.task_memories:
            for name, param in model.named_parameters():
                if param.requires_grad and name in mem['fisher']:
                    diff = param - mem['params'][name]
                    total += (mem['fisher'][name] * diff.pow(2)).sum()
        return (lambda_ewc / 2.0) * total

class ExperienceReplayBuffer:
    def __init__(self, buffer_size: int = 500):
        self.buffer_size = buffer_size
        self.buffers: dict = {}

    def add_examples(self, task: str, dataloader: DataLoader):
        self.buffers[task] = []
        for ids, fm, lbl in dataloader:
            for i in range(ids.size(0)):
                if len(self.buffers[task]) >= self.buffer_size: break
                self.buffers[task].append((ids[i].cpu(), fm[i].cpu(), lbl[i].cpu()))
            if len(self.buffers[task]) >= self.buffer_size: break

    def has_data(self) -> bool:
        return bool(self.buffers)

# ═══════════════════════════════════════════════════════════════════════
# TOKENIZER & DATASETS
# ═══════════════════════════════════════════════════════════════════════
def build_global_tokenizer(cfg: CFG) -> Tokenizer:
    if os.path.exists(cfg.tokenizer_path):
        tok = Tokenizer.from_file(cfg.tokenizer_path)
        if tok.get_vocab_size() >= cfg.tokenizer_vocab_size - 200:
            log(f'Tokenizer carregado ({tok.get_vocab_size()} tokens)')
            return tok
    log('Treinando tokenizer BPE 30k global…', 'INFO')
    texts = []
    for task in cfg.TASK_ORDER:
        log(f'  Coletando {task}…', 'INFO')
        ds = load_dataset('glue', task, split='train')
        keys  = cfg.GLUE_TASKS[task]['keys']
        limit = min(len(ds), 100000)
        for i in range(limit):
            texts.append(' '.join([str(ds[i].get(k, '')) for k in keys]))
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
    return tok

class GlueDataset(Dataset):
    def __init__(self, hf_ds, tok: Tokenizer, cfg: CFG, task: str):
        pad_id   = tok.token_to_id(cfg.pad_token)
        keys     = cfg.GLUE_TASKS[task]['keys']
        is_reg   = cfg.GLUE_TASKS[task]['type'] == 'reg'
        self.samples = []
        for item in hf_ds:
            text = ' '.join([str(item.get(k, '')) for k in keys])
            ids  = tok.encode(text[:10000]).ids[:cfg.max_len]
            vl   = len(ids)
            inp  = ids + [pad_id] * (cfg.max_len - vl)
            fm   = [0.0] * vl + [float('-inf')] * (cfg.max_len - vl)
            lbl  = float(item['label']) if is_reg else int(item['label'])
            self.samples.append((
                torch.tensor(inp, dtype=torch.long),
                torch.tensor(fm,  dtype=torch.float32),
                torch.tensor(lbl, dtype=torch.float32 if is_reg else torch.long)
            ))
    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

class GlueTestDataset(Dataset):
    def __init__(self, hf_ds, tok: Tokenizer, cfg: CFG, task: str):
        pad_id = tok.token_to_id(cfg.pad_token)
        keys   = cfg.GLUE_TASKS[task]['keys']
        self.task    = task
        self.samples = []
        for item in hf_ds:
            text = ' '.join([str(item.get(k, '')) for k in keys])
            ids  = tok.encode(text[:10000]).ids[:cfg.max_len]
            vl   = len(ids)
            inp  = ids + [pad_id] * (cfg.max_len - vl)
            fm   = [0.0] * vl + [float('-inf')] * (cfg.max_len - vl)
            idx  = item.get('idx', item.get('id', -1))
            self.samples.append((
                torch.tensor(inp, dtype=torch.long),
                torch.tensor(fm,  dtype=torch.float32),
                idx
            ))
    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

LABEL_STR_MAP = {
    'mnli': ['entailment', 'neutral', 'contradiction'],
    'qnli': ['entailment', 'not_entailment'],
    'rte':  ['entailment', 'not_entailment'],
}

# ═══════════════════════════════════════════════════════════════════════
# META LEARNING & CICLOS DE TREINO (OTIMIZADOS)
# ═══════════════════════════════════════════════════════════════════════
def meta_learning_warmup(model, all_dataloaders: dict, cfg: CFG):
    """
    Meta-Learning via First-Order MAML (Reptile) Otimizado.
    Usa AdamW no Outer-Loop para melhor convergência da topologia do Transformer
    e Stratified Task Sampling via Queue.
    """
    log('[META] Iniciando fase de meta-aprendizagem de primeira ordem (AdamW + Stratified)...', 'META')
    model.train()
    
    tasks = list(all_dataloaders.keys())
    loaders_iter = {t: iter(dl) for t, dl in all_dataloaders.items()}
    
    # Outer optimizer melhorado (AdamW lida melhor com pesos do transformer)
    meta_opt = torch.optim.AdamW(model.parameters(), lr=cfg.meta_outer_lr, weight_decay=cfg.wd)
    
    task_queue = tasks.copy()
    random.shuffle(task_queue)
    
    for step in range(cfg.meta_warmup_steps):
        theta_init = {k: v.clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        
        # Seleciona garantindo amostragem estratificada
        if not task_queue:
            task_queue = tasks.copy()
            random.shuffle(task_queue)
        task = task_queue.pop()
        
        # Inner loop (SGD mantém-se ideal para simular a trajetória do gradiente limpo)
        inner_opt = torch.optim.SGD(model.parameters(), lr=cfg.meta_inner_lr, momentum=0.9)
        
        for inner_step in range(cfg.meta_inner_steps):
            try:
                ids, fm, lbl = next(loaders_iter[task])
            except StopIteration:
                loaders_iter[task] = iter(all_dataloaders[task])
                ids, fm, lbl = next(loaders_iter[task])
            
            ids = ids.to(cfg.device)
            fm = fm.to(cfg.device)
            lbl = lbl.to(cfg.device)
            is_reg = cfg.GLUE_TASKS[task]['type'] == 'reg'
            
            inner_opt.zero_grad()
            with autocast(device_type=cfg.device.type, dtype=cfg.amp_dtype):
                logits = model(ids, fm, task)
                loss = regression_loss(logits.squeeze(-1), lbl.float()) if is_reg else F.cross_entropy(logits, lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad) # Clip vital no inner-loop
            inner_opt.step()
        
        # Outer update Direto nos gradientes para o AdamW
        meta_opt.zero_grad()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.dtype.is_floating_point and name in theta_init:
                    # Pseudo-gradiente aponta na direção inversa da diferença 
                    param.grad = -(param.data - theta_init[name]) 
                    param.data.copy_(theta_init[name]) # Restaura o theta real para ser atualizado pelo AdamW
        meta_opt.step()
        
        if (step + 1) % 50 == 0:
            log(f'  [META] Outer Step {step+1}/{cfg.meta_warmup_steps} | '
                f'Task: {task} | Inner Loss: {loss.item():.4f}', 'META')
    
    log('[META] Meta-aprendizagem completa. Inicialização base pronta.', 'META')

def train_task_epoch(model, ema, optimizer, scaler, loader, awp, cfg,
                     epoch, gstep, lr_sched, base_opt, ewc,
                     replay_buffer, unk_id, task):
    model.train()
    is_reg = cfg.GLUE_TASKS[task]['type'] == 'reg'
    n = len(loader); t0 = time.time()

    progress = (epoch - 1) / max(cfg.epochs_per_task - 1, 1)
    mixp = cfg.mixup_prob_base * (1.0 - progress) + cfg.mixup_prob_min * progress
    ls   = cfg.label_smooth_base * (1.0 - progress) + cfg.label_smooth_min * progress

    st = {'loss': 0.0, 'ce': 0.0, 'ewc': 0.0, 'ortho': 0.0, 'cor': 0, 'tot': 0}
    optimizer.zero_grad(set_to_none=True)

    for step, (ids, fm, lbl) in enumerate(loader):
        ids = ids.to(cfg.device, non_blocking=True)
        fm  = fm.to(cfg.device,  non_blocking=True)
        lbl = lbl.to(cfg.device, non_blocking=True)

        ids = token_dropout(ids, fm, unk_id, cfg.token_dropout_prob,
                            cfg.token_dropout_apply)
        use_slerp = (random.random() < mixp) and (not is_reg)

        with autocast(device_type=cfg.device.type, dtype=cfg.amp_dtype):
            if use_slerp:
                emb  = model.embed_drop(model.embed(ids) * model.embed_scale)
                lam  = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                idx2 = torch.randperm(emb.size(0), device=cfg.device)
                emb_mix = slerp_mixup(emb, emb[idx2], lam)
                logits  = model.heads[task](model.encode(emb_mix, fm, task))
                lp = F.log_softmax(logits, -1)
                C  = cfg.GLUE_TASKS[task]['classes']
                t1 = torch.full_like(lp, ls / max(C - 1, 1))
                t2 = torch.full_like(lp, ls / max(C - 1, 1))
                t1.scatter_(-1, lbl.unsqueeze(-1), 1.0 - ls)
                t2.scatter_(-1, lbl[idx2].unsqueeze(-1), 1.0 - ls)
                ce = (lam * (-(t1 * lp).sum(-1).mean()) +
                      (1.0 - lam) * (-(t2 * lp).sum(-1).mean()))
            else:
                logits = model(ids, fm, task)
                if is_reg:
                    ce = regression_loss(logits.squeeze(-1), lbl.float())
                else:
                    ce = focal_ce(logits, lbl, cfg.focal_gamma, ls)

            ewc_pen   = ewc.penalty(model, cfg.ewc_lambda)
            sph_div   = (model.sphere_nav.anchor_diversity_loss() * cfg.sphere_anchor_div_w)
            div       = model.system2.diversity_loss() * cfg.div_weight
            ortho_pen = model.task_prompts.get_orthogonal_loss() * cfg.prompt_ortho_weight
            
            loss      = ce + div + ewc_pen + sph_div + ortho_pen

        scaler.scale(loss / cfg.grad_accum).backward()

        if (step + 1) % cfg.grad_accum == 0:
            scaler.unscale_(base_opt)

            if epoch >= cfg.awp_start_ep:
                awp.perturb()
                with autocast(device_type=cfg.device.type, dtype=cfg.amp_dtype):
                    l2    = model(ids, fm, task)
                    ce_awp = (regression_loss(l2.squeeze(-1), lbl.float()) if is_reg
                              else focal_ce(l2, lbl, cfg.focal_gamma, ls))
                (ce_awp / cfg.grad_accum).backward()
                awp.restore()

            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)

            scaler.step(base_opt)
            optimizer._sync()
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            ema.update(model)
            gstep += 1
            lr_sched.step_batch()

        st['loss']  += float(loss.item())
        st['ce']    += float(ce.item())
        st['ewc']   += float(ewc_pen.item())
        st['ortho'] += float(ortho_pen.item())
        if not use_slerp and not is_reg:
            st['cor'] += (logits.argmax(-1) == lbl).sum().item()
        st['tot'] += lbl.size(0)

        if step % cfg.attn_log_freq == 0 or step == n - 1:
            ela = time.time() - t0
            eta = ela / (step + 1) * (n - step - 1)
            acc_str = (f'tr_acc={st["cor"]/max(st["tot"],1)*100:.2f}%'
                       if not is_reg else 'reg')
            log(f'[{task}] ep={epoch:02d} step={step:04d}/{n} '
                f'lr={lr_sched.get_lr():.2e} loss={loss.item():.5f} '
                f'ce={ce.item():.5f} ewc={ewc_pen.item():.5f} '
                f'ortho={ortho_pen.item():.4f} {acc_str} '
                f'{ela:.1f}s ETA={eta:.1f}s', 'ATTN')

    avg_acc = (st['cor'] / max(st['tot'], 1)) if not is_reg else (-st['loss'] / n)
    return {'loss': st['loss'] / n, 'acc': avg_acc}, gstep

@torch.no_grad()
def evaluate_task(model, loader, cfg: CFG, task: str) -> dict:
    model.eval()
    is_reg     = cfg.GLUE_TASKS[task]['type'] == 'reg'
    all_preds  = []
    all_labels = []
    total_loss = 0.0
    tot        = 0

    for ids, fm, lbl in loader:
        ids = ids.to(cfg.device, non_blocking=True)
        fm  = fm.to(cfg.device,  non_blocking=True)
        lbl = lbl.to(cfg.device, non_blocking=True)
        with autocast(device_type=cfg.device.type, dtype=cfg.amp_dtype):
            logits = model(ids, fm, task)
        if is_reg:
            preds = logits.squeeze(-1).float().clamp(0.0, 5.0)
            total_loss += F.mse_loss(preds, lbl.float()).item() * lbl.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(lbl.cpu().tolist())
        else:
            total_loss += F.cross_entropy(logits, lbl).item() * lbl.size(0)
            probs = F.softmax(logits, dim=-1)[:, 1] if logits.size(-1) == 2 else logits.argmax(-1).float()
            all_preds.extend((logits.argmax(-1)).cpu().tolist())
            all_labels.extend(lbl.cpu().tolist())
        tot += lbl.size(0)

    metric = compute_task_metric(all_preds, all_labels, task)
    return {'metric': metric, 'acc': metric, 'loss': total_loss / max(tot, 1)}

@torch.no_grad()
def generate_task_submission(model, test_ds: GlueTestDataset, cfg: CFG, filename: str):
    model.eval()
    loader   = DataLoader(test_ds, batch_size=256, shuffle=False,
                          num_workers=4, pin_memory=True)
    task     = test_ds.task
    is_reg   = cfg.GLUE_TASKS[task]['type'] == 'reg'
    out_path = os.path.join(cfg.checkpoint_dir, filename)
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['index', 'prediction'])
        for ids, fm, idxs in loader:
            ids = ids.to(cfg.device, non_blocking=True)
            fm  = fm.to(cfg.device,  non_blocking=True)
            with autocast(device_type=cfg.device.type, dtype=cfg.amp_dtype):
                logits = model(ids, fm, task)
            if is_reg:
                preds = logits.clamp(0.0, 5.0).cpu().tolist()
            else:
                preds_idx = logits.argmax(-1).cpu().tolist()
                if task in LABEL_STR_MAP:
                    mapping = LABEL_STR_MAP[task]
                    preds   = [mapping[p] for p in preds_idx]
                else:
                    preds = preds_idx
            for i, p in zip(idxs.tolist(), preds):
                w.writerow([i, p])
    log(f'[LB] Gerado {filename} ({len(test_ds)} previsões).', 'LB')

# ═══════════════════════════════════════════════════════════════════════
# MAIN (OTIMIZADO COM LAYER-WISE LEARNING RATE DECAY)
# ═══════════════════════════════════════════════════════════════════════
def main():
    cfg = CFG()
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    if HAS_RICH:
        console.rule(f'[bold green]GrafoPropagation {cfg.VERSION} · GLUE Multi-Task · Run {RUN_ID}[/bold green]')
    log(f'Treino sequencial: {" → ".join(cfg.TASK_ORDER)}')

    tok    = build_global_tokenizer(cfg)
    unk_id = tok.token_to_id(cfg.unk_token)

    model  = GrafoPropagationMT(cfg, tok).to(cfg.device)
    ema    = EMA(model, cfg.ema_decay)
    ewc    = ElasticWeightConsolidation(model, cfg.device)
    replay = ExperienceReplayBuffer(cfg.replay_buffer_size)

    total = sum(p.numel() for p in model.parameters())
    sph_p = sum(p.numel() for p in model.sphere_nav.parameters())
    log(f'Parâmetros totais: {total:,} ({total/1e6:.3f} M) [SphereNavigator: {sph_p:,}]', 'SYS2')

    TSV_MAPPING = {
        'cola': 'CoLA.tsv', 'sst2': 'SST-2.tsv', 'mrpc': 'MRPC.tsv',
        'qqp':  'QQP.tsv',  'stsb': 'STS-B.tsv', 'qnli': 'QNLI.tsv',
        'rte':  'RTE.tsv',  'wnli': 'WNLI.tsv',
    }

    tr_dataloaders = {}
    val_dataloaders = {}
    raw_datasets = {}
    for task in cfg.TASK_ORDER:
        raw = load_dataset('glue', task)
        raw_datasets[task] = raw
        tr_ds = GlueDataset(raw['train'], tok, cfg, task)
        val_split = 'validation_matched' if task == 'mnli' else 'validation'
        va_ds = GlueDataset(raw[val_split], tok, cfg, task)
        tr_dataloaders[task] = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        val_dataloaders[task] = DataLoader(va_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    # Executar Meta-Warmup com AdamW e Stratified Sampling
    meta_learning_warmup(model, tr_dataloaders, cfg)
    ema.update(model)

    for task in cfg.TASK_ORDER:
        log('=' * 80, 'LB')
        log(f'Task: {task.upper()}', 'LB')

        tr_ld = tr_dataloaders[task]
        va_ld = val_dataloaders[task]
        raw   = raw_datasets[task]

        # ─── Implementação de Layer-Wise Learning Rate Decay (LLRD) ────────
        param_groups = []
        
        embed_params = [p for n, p in model.named_parameters() if 'embed' in n and p.requires_grad]
        head_params  = [p for n, p in model.named_parameters() if 'heads' in n and p.requires_grad]
        nas_params   = [p for n, p in model.named_parameters() if 'arch_alphas' in n and p.requires_grad]
        
        if embed_params: param_groups.append({'params': embed_params, 'lr': cfg.lr_embeddings, 'layer_type': 'embeddings'})
        if head_params:  param_groups.append({'params': head_params,  'lr': cfg.lr_heads,      'layer_type': 'heads'})
        if nas_params:   param_groups.append({'params': nas_params,   'lr': cfg.nas_lr,        'layer_type': 'nas_alphas'})

        # LLRD distribuído pelas layers do backbone Transformer
        for l in range(cfg.n_layers):
            # FIX: excluir 'arch_alphas' para evitar duplicação com o grupo nas_params
            layer_params = [p for n, p in model.named_parameters() if f'blocks.{l}.' in n and 'arch_alphas' not in n and p.requires_grad]
            if layer_params:
                # Layers próximas da output recebem LR mais alto. Layers de input são amortecidas pelo decay.
                decay_factor = cfg.llrd_decay ** (cfg.n_layers - l - 1)
                param_groups.append({'params': layer_params, 'lr': cfg.lr_backbone * decay_factor, 'layer_type': f'block_{l}'})
                
        # Resto do backbone que não faz parte das layers centrais
        other_backbone = [p for n, p in model.named_parameters() if 
                          not any(k in n for k in ['embed', 'heads', 'arch_alphas', 'blocks.']) and p.requires_grad]
        if other_backbone:
            param_groups.append({'params': other_backbone, 'lr': cfg.lr_backbone, 'layer_type': 'other_backbone'})
        # ───────────────────────────────────────────────────────────────────

        base_opt  = torch.optim.AdamW(param_groups, lr=0.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=cfg.wd)
        optimizer = Lookahead(base_opt, cfg.la_k, cfg.la_alpha)
        scaler    = GradScaler('cuda', enabled=(cfg.amp_dtype == torch.float16))
        awp       = AWP(model, cfg.awp_eps, cfg.awp_lr)

        steps_per_epoch = len(tr_ld) // cfg.grad_accum
        lr_sched = LaboratoryLRScheduler(base_opt, cfg, steps_per_epoch)

        best_metric = -float('inf')
        gstep       = 0
        
        soup = DynamicGreedySoup(k=cfg.soup_k)

        for epoch in range(1, cfg.epochs_per_task + 1):
            tr_s, gstep = train_task_epoch(
                model, ema, optimizer, scaler, tr_ld, awp, cfg,
                epoch, gstep, lr_sched, base_opt, ewc, replay, unk_id, task)

            va_s = evaluate_task(ema.shadow, va_ld, cfg, task)
            lr_sched.step_epoch(metric=va_s['metric'])

            metric_name = {'cola': 'MCC', 'stsb': 'Pearson', 'mrpc': 'F1+Acc', 'qqp': 'F1+Acc'}.get(task, 'Acc')

            log(f'[{task}] EP{epoch:02d} tr_loss={tr_s["loss"]:.4f} val_{metric_name}={va_s["metric"]:.4f} lr={lr_sched.get_lr():.2e}')

            soup.add(va_s['metric'], ema.shadow.state_dict())

            if va_s['metric'] > best_metric:
                best_metric = va_s['metric']
                torch.save(
                    {'ema': ema.shadow.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'epoch': epoch,
                     'metric': best_metric},
                    os.path.join(cfg.checkpoint_dir, f'best_{task}.pt'))
                log(f'[{task}] Novo melhor {metric_name}={best_metric:.4f} guardado.', 'METRIC')

        log(f'[ULTRA] Fundindo Top-{cfg.soup_k} Modelos da Sopa...', 'ULTRA')
        merged_state = soup.merge()
        if merged_state:
            ema.shadow.load_state_dict(merged_state)
            soup_eval = evaluate_task(ema.shadow, va_ld, cfg, task)
            if soup_eval['metric'] > best_metric:
                log(f'[ULTRA] Sopa vitoriosa! Metric aumentou de {best_metric:.4f} para {soup_eval["metric"]:.4f}', 'METRIC')
                torch.save({'ema': merged_state, 'metric': soup_eval['metric']}, os.path.join(cfg.checkpoint_dir, f'best_{task}_soup.pt'))
            else:
                log(f'[ULTRA] Sopa falhou (deu {soup_eval["metric"]:.4f}). Revertendo para o melhor single checkpoint.', 'WARN')
                best_ckpt = torch.load(os.path.join(cfg.checkpoint_dir, f'best_{task}.pt'), weights_only=True)
                ema.shadow.load_state_dict(best_ckpt['ema'])

        log(f'[EWC] Guardando knowledge de {task}…', 'LAB')
        ewc.compute_fisher(tr_ld, task, num_samples=cfg.ewc_fisher_samples)
        replay.add_examples(task, tr_ld)

        if task == 'mnli':
            t_m  = GlueTestDataset(raw['test_matched'],    tok, cfg, task)
            t_mm = GlueTestDataset(raw['test_mismatched'], tok, cfg, task)
            generate_task_submission(ema.shadow, t_m,  cfg, 'MNLI-m.tsv')
            generate_task_submission(ema.shadow, t_mm, cfg, 'MNLI-mm.tsv')
            log('Gerando AX.tsv (diagnóstico MNLI)…', 'LB')
            ds_ax = load_dataset('glue', 'ax')
            t_ax  = GlueTestDataset(ds_ax['test'], tok, cfg, 'mnli')
            generate_task_submission(ema.shadow, t_ax, cfg, 'AX.tsv')
        else:
            test_split = 'test' if 'test' in raw else val_split
            t_ds = GlueTestDataset(raw[test_split], tok, cfg, task)
            generate_task_submission(ema.shadow, t_ds, cfg, TSV_MAPPING[task])

    log('=' * 80, 'LB')
    zip_path = os.path.join(cfg.checkpoint_dir, 'submission_glue_v28_ultra.zip')
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for tsv in list(TSV_MAPPING.values()) + ['MNLI-m.tsv', 'MNLI-mm.tsv', 'AX.tsv']:
            tsv_path = os.path.join(cfg.checkpoint_dir, tsv)
            if os.path.exists(tsv_path):
                zf.write(tsv_path, arcname=tsv)
                log(f'  [ZIP] {tsv}', 'INFO')
            else:
                log(f'  [ERRO] Faltou {tsv}', 'ERROR')

    log(f'[GLUE] ULTRA concluído. Submissão: {zip_path}', 'LB')

    log_path = os.path.join(cfg.checkpoint_dir, f'run_{RUN_ID}.json')
    with open(log_path, 'w') as f:
        json.dump(_LOG_BUF, f, indent=2)
    log(f'Log guardado → {log_path}')

if __name__ == '__main__':
    main()