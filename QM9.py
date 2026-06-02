#!/usr/bin/env python3
"""
GrafoPropagation v27-GEO-APEX-QM9-RAD · HOMO-LUMO Gap Regression
=================================================================
"Rich Atomic Descriptors" — representação atómica ultra-rica que substitui
a CoulombMatrix por features fisicamente significativas + encoding geométrico.

REVOLUÇÃO vs v26 (CoulombMatrix):
  ✗ v26: CoulombMatrix 29×29 → cada linha = 1 átomo "token" → Linear(29, 192)
         Cada átomo só vê UMA linha da matriz → perde geometria 3D!
  ✓ v27: 47+16 features por átomo:
         • 27 propriedades físicas (EN×4, IE, EA, raios, polar., config. electr.)
         • 20 features moleculares (posição 3D, estatísticas distância, vizinhos,
           contexto electrostático, tamanho mol.)
         • 16-dim learnable Z embedding (como PaiNN/SchNet/DimeNet++)
         • DistanceAttentionBias com Bessel RBF + Gaussian RBF (DimeNet++ style)

Arquitectura IDÊNTICA (mantida conforme pedido do utilizador):
  d=192, L=4, H=8, hd=24, d_ff=896
  ✓ AcceleratedRoPERotator (gradiente activo)
  ✓ XorAttentionBias (O(1) memória)
  ✓ TopologicalMERAScore (MERA-inspired)
  ✓ CyclicTemporalBias (wrap-around)
  ✓ ConditionalValueGate
  ✓ XORSpatialFusion
  ✓ LocalConvMix + SwiGLU FFN
  ✓ RMSNorm, Stochastic Depth
  ✓ EMA, Lookahead, AWP, GradCentralization

NOVO: DistanceAttentionBias
  ✓ Bessel RBF (8 bases) — orthogonal, info-dense (DimeNet++)
  ✓ Gaussian RBF (16 bases) — smooth, broad (SchNet/PaiNN)
  ✓ Polynomial cutoff envelope p=5 (DimeNet++)
  ✓ Projeção Linear → per-head attention bias

Input Pipeline:
  Cache V4: z_indices + positions + fmask + targets_raw + n_atoms (RAW)
  Features computadas APÓS split → normalização train-only (sem batota!)
  Per-atom: [fixed_phys(27) | mol_dependent(20) | z_embed(16)] = 63 → Linear(63,192)

Compliance: train-only normalization, proper splits, sem data leakage.
"""

import math, random, os, sys, time, datetime, json, warnings, copy, subprocess, bz2
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    console = Console(width=200, force_terminal=True)
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    class _FC:
        def print(self, *a, **kw): print(*a)
        def rule(self, *a, **kw): print('─' * 120)
    console = _FC()

# ════════════════════════════════════ CONFIG ═══════════════════════════════════
class CFG:
    VERSION   = 'v27-GEO-APEX-QM9-RAD'
    seed      = 42
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_dtype = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                 else torch.float32)

    # QM9 Dataset
    max_atoms    = 29
    # Feature dims: fixed_phys(27) + mol_dependent(20) + z_embed(16) = 63
    n_fixed_feats   = 27
    n_mol_feats     = 20
    n_z_embed       = 16
    feature_dim     = n_fixed_feats + n_mol_feats + n_z_embed  # = 63
    gap_idx         = 4
    qm9_root        = os.environ.get('QM9_ROOT', '/tmp/QM9')
    cache_path      = os.environ.get('QM9_CACHE', '/tmp/qm9_rad_cache_v4.pt')
    if not os.access(os.path.dirname(cache_path) or '.', os.W_OK):
        cache_path = './qm9_rad_cache_v4.pt'
    if not os.access(os.path.dirname(qm9_root) or '.', os.W_OK):
        qm9_root = './QM9'
    normalize_y  = True

    # DistanceAttentionBias
    dist_n_rbf_gauss  = 16        # Gaussian RBF basis functions
    dist_n_rbf_bessel = 8         # Bessel RBF basis functions
    dist_cutoff       = 5.0       # Å — cutoff distance
    use_distance_bias = True

    # Arquitectura (~2.83M) — IDÊNTICO ao v26
    d_model     = 192
    n_layers    = 4
    n_heads     = 8
    head_dim    = 24
    d_ff        = 896
    dropout     = 0.10
    stoch_depth = 0.08
    conv_kernel = 3

    # RoPE
    use_accelerated_rope = True
    rope_base   = 10000.0
    rope_max_seq = 64

    # Attention biases
    use_cyclic_mask       = True
    use_xor_bias          = True
    use_topological_bias  = True
    use_conditional_gate  = True

    # Treino — MESMO do v26
    epochs       = 60
    batch_size   = 128
    grad_accum   = 1
    wd           = 1e-4
    clip_grad    = 1.0
    base_lr_max  = 5e-4
    warmup_frac  = 0.05
    min_lr_frac  = 0.05
    ema_decay    = 0.999
    awp_eps      = 0.003
    awp_lr       = 0.005
    awp_start_ep = 15
    la_k         = 6
    la_alpha     = 0.50
    huber_delta  = 1.0
    mixup_alpha  = 0.20
    mixup_prob   = 0.10

    # Logging
    attn_log_freq    = 20
    checkpoint_every = 25
    checkpoint_dir   = './ckpt_qm9_v27_rad'
    history_path     = './ckpt_qm9_v27_rad/history.json'

# ══════════════════════════════════ UTILITÁRIOS ═══════════════════════════════
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
set_seed(CFG.seed)

RUN_ID = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
_LOG_BUF = []

def _ts():
    return datetime.datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]

def log(msg, level='INFO'):
    ts = _ts()
    _LOG_BUF.append({'ts': ts, 'lvl': level, 'msg': msg})
    style_map = {
        'INFO': 'dim', 'WARN': 'yellow', 'ERROR': 'bold red',
        'METRIC': 'bold cyan', 'DATA': 'bold green', 'TRAIN': 'magenta',
        'EVAL': 'bold blue',
    }
    s = style_map.get(level, '')
    if HAS_RICH and s:
        console.print(f'[{s}][[{ts}]] [{level}] {msg}[/{s}]')
    else:
        print(f'[[{ts}]] [{level}] {msg}', flush=True)

def log_separator(title='', char='═', width=120):
    if HAS_RICH:
        console.rule(f'[bold]{title}[/bold]', style='bright_blue', characters=char)
    else:
        print(f'\n{char * width}')
        if title: print(f'  {title}')
        print(f'{char * width}\n')

# ═════════════════════════ QUANTUM ATOMIC PROPERTIES ═════════════════════════════
# Comprehensive lookup table of physically meaningful atomic properties.
# All values from NIST/PubChem/IUPAC primary sources.
# Only H, C, N, O, F appear in QM9 (max 29 atoms, up to 9 heavy atoms + H).

ATOMIC_PROPS = {
    # Z: (symbol, pauling_en, allen_en, allred_rochow_en,
    #     ie_eV, ea_eV, cov_rad_pm, vdw_rad_pm, polariz_A3,
    #     valence_e, val_s, val_p, unpaired_e, lone_pairs,
    #     period, group, max_valence, z_eff_slater, core_e,
    #     self_energy, nuc_charge_density, elec_density_proxy)
    1: ('H',  2.20, 2.300, 2.20,
        13.59844, 0.75420, 31, 110, 0.667,
        1, 1, 0, 1, 0,
        1, 1, 1, 1.0, 0,
        0.5, 0.0337, 0.0337),
    6: ('C',  2.55, 2.544, 2.50,
        11.26030, 1.26212, 76, 170, 1.760,
        4, 2, 2, 2, 0,
        2, 14, 4, 3.25, 2,
        50.2, 0.0273, 0.0109),
    7: ('N',  3.04, 3.066, 3.07,
        14.53414, -0.07, 71, 155, 1.100,
        5, 2, 3, 3, 1,
        2, 15, 3, 3.90, 2,
        79.3, 0.0278, 0.0139),
    8: ('O',  3.44, 3.610, 3.50,
        13.61806, 1.46111, 66, 152, 0.802,
        6, 2, 4, 2, 2,
        2, 16, 2, 4.55, 2,
        115.3, 0.0278, 0.0185),
    9: ('F',  3.98, 4.193, 4.10,
        17.42282, 3.40119, 57, 147, 0.557,
        7, 2, 5, 1, 3,
        2, 17, 1, 5.20, 2,
        162.5, 0.0280, 0.0271),
}

# Indices into ATOMIC_PROPS tuples for building feature vectors
_APROP_SYMBOL    = 0
_APROP_PAULING   = 1
_APROP_ALLEN     = 2
_APROP_ALLRED    = 3
_APROP_IE        = 4
_APROP_EA        = 5
_APROP_COV_R     = 6
_APROP_VDW_R     = 7
_APROP_POLARIZ   = 8
_APROP_VAL_E     = 9
_APROP_VAL_S     = 10
_APROP_VAL_P     = 11
_APROP_UNPAIRED  = 12
_APROP_LONE_P    = 13
_APROP_PERIOD    = 14
_APROP_GROUP     = 15
_APROP_MAX_VAL   = 16
_APROP_ZEFF      = 17
_APROP_CORE_E    = 18
_APROP_SELF_E    = 19
_APROP_NUC_DENS  = 20
_APROP_ELEC_DENS = 21

# Per-Z fixed feature vector builder
# Returns 27-dim vector: [one_hot(5), Z, pauling, allen, allred, IE, EA,
#   cov_rad, vdw_rad, polariz, val_e, val_s, val_p, unpaired, lone_pairs,
#   period, group, max_val, z_eff, core_e, self_energy, nuc_dens, elec_dens]
def _build_fixed_feature_vector(z):
    """Build 27-dim fixed physical property vector for atomic number z."""
    if z not in ATOMIC_PROPS:
        z = 6  # Default to carbon if unknown (shouldn't happen for QM9)
    p = ATOMIC_PROPS[z]
    # One-hot atom type
    one_hot = [1.0 if z == zn else 0.0 for zn in [1, 6, 7, 8, 9]]
    return np.array(one_hot + [
        float(z),                      # Z
        p[_APROP_PAULING],             # Pauling EN
        p[_APROP_ALLEN],               # Allen EN
        p[_APROP_ALLRED],              # Allred-Rochow EN
        p[_APROP_IE],                  # Ionization Energy (eV)
        p[_APROP_EA],                  # Electron Affinity (eV)
        p[_APROP_COV_R] / 100.0,       # Covalent radius (Å)
        p[_APROP_VDW_R] / 100.0,       # vdW radius (Å)
        p[_APROP_POLARIZ],             # Polarizability (ų)
        float(p[_APROP_VAL_E]),        # Valence electrons
        float(p[_APROP_VAL_S]),        # Valence s electrons
        float(p[_APROP_VAL_P]),        # Valence p electrons
        float(p[_APROP_UNPAIRED]),     # Unpaired electrons
        float(p[_APROP_LONE_P]),       # Lone pairs
        float(p[_APROP_PERIOD]),       # Period
        float(p[_APROP_GROUP]),        # Group
        float(p[_APROP_MAX_VAL]),      # Max valence
        p[_APROP_ZEFF],                # Effective nuclear charge (Slater)
        float(p[_APROP_CORE_E]),       # Core electrons
        p[_APROP_SELF_E],              # 0.5 * Z^2.4 (Coulomb self-energy)
        p[_APROP_NUC_DENS],            # Nuclear charge density
        p[_APROP_ELEC_DENS],           # Electron density proxy
    ], dtype=np.float64)

# Pre-build lookup table (10 entries, indexed by Z)
FIXED_FEAT_TABLE = np.zeros((10, CFG.n_fixed_feats), dtype=np.float32)
for _z in [1, 6, 7, 8, 9]:
    FIXED_FEAT_TABLE[_z] = _build_fixed_feature_vector(_z).astype(np.float32)


def compute_mol_dependent_features(z_arr, pos, max_atoms):
    """
    Compute molecule-dependent per-atom features (20 dims).

    z_arr: (n,) atomic numbers (float64)
    pos:   (n, 3) positions (float64, Angstrom)
    max_atoms: 29

    Returns: (max_atoms, 20) Float64 array (padded with 0)
    """
    M = max_atoms
    n = min(len(z_arr), M)
    feats = np.zeros((M, 20), dtype=np.float64)

    if n == 0:
        return feats

    z = z_arr[:n].astype(np.float64)
    p = pos[:n].astype(np.float64)

    # Centroid
    centroid = p.mean(axis=0)
    p_centered = p - centroid  # (n, 3)

    # Molecular radius
    dists_to_centroid = np.sqrt((p_centered ** 2).sum(axis=1))
    mol_radius = max(dists_to_centroid.max(), 1e-6)
    mol_size = n / M  # normalized

    # Pairwise distances
    diff = p[:, None, :] - p[None, :, :]  # (n, n, 3)
    dist_mat = np.sqrt((diff ** 2).sum(axis=-1))  # (n, n)
    # NOTE: We do NOT fill diagonal with 1e10 here — instead, we exclude
    # the diagonal when computing per-atom statistics below.

    # Pairwise Coulomb interactions: Z_i * Z_j / r_ij
    dist_mat_safe = dist_mat.copy()
    np.fill_diagonal(dist_mat_safe, 1.0)  # avoid div/0
    zz = z[:, None] * z[None, :]  # (n, n)
    coulomb = zz / dist_mat_safe  # (n, n)
    np.fill_diagonal(coulomb, 0.0)

    for i in range(n):
        d_i = dist_mat[i].copy()  # (n,) distances from atom i to all others
        d_i[i] = np.inf  # exclude self-distance (diagonal = 0)

        # Sort distances (for nearest neighbor distances)
        sorted_d = np.sort(d_i[d_i < np.inf])  # only real distances

        # Distance statistics (excluding self)
        n_others = len(sorted_d)
        mean_dist = sorted_d.mean() if n_others > 0 else 0.0
        std_dist = sorted_d.std() if n_others > 0 else 0.0
        min_dist = sorted_d[0] if n_others > 0 else 0.0  # 1st NN
        nn2_dist = sorted_d[1] if n_others > 1 else min_dist  # 2nd NN
        nn3_dist = sorted_d[2] if n_others > 2 else nn2_dist  # 3rd NN
        max_dist = sorted_d[-1] if n_others > 0 else 0.0

        # Neighbor counts (Gaussian-smoothed, excluding self)
        d_others = d_i[d_i < np.inf]
        sigma_smooth = 0.3
        nb_1p8 = np.exp(-0.5 * ((d_others - 1.8) / sigma_smooth) ** 2).sum()  # ~bonded
        nb_2p5 = np.exp(-0.5 * ((d_others - 2.5) / sigma_smooth) ** 2).sum()  # ~second shell
        nb_3p5 = np.exp(-0.5 * ((d_others - 3.5) / sigma_smooth) ** 2).sum()  # ~third shell

        # Electrostatic context
        c_i = coulomb[i]
        sum_coulomb = c_i.sum()
        # exp-weighted Z of neighbors (exclude self)
        z_others = np.delete(z, i)
        local_elec_dens = (z_others * np.exp(-d_others / 2.0)).sum()  # exp-weighted Z

        # Position features
        x_c, y_c, z_c = p_centered[i]
        xn = x_c / mol_radius
        yn = y_c / mol_radius
        zn = z_c / mol_radius

        # Distance to centroid
        d_centroid = dists_to_centroid[i]

        feats[i] = [
            x_c, y_c, z_c,            # Centered position (3)
            xn, yn, zn,               # Normalized position (3)
            mean_dist,                 # Mean distance (1)
            std_dist,                  # Std distance (1)
            min_dist,                  # 1st NN distance (1)
            nn2_dist,                  # 2nd NN distance (1)
            nn3_dist,                  # 3rd NN distance (1)
            max_dist,                  # Max distance (1)
            nb_1p8,                   # Neighbor count at 1.8Å (1)
            nb_2p5,                   # Neighbor count at 2.5Å (1)
            nb_3p5,                   # Neighbor count at 3.5Å (1)
            sum_coulomb,              # Sum Coulomb interactions (1)
            local_elec_dens,          # Local electron density (1)
            d_centroid,               # Distance to centroid (1)
            mol_size,                 # Molecular size (1)
            mol_radius,               # Molecular radius (1)
        ]

    return feats


# ═════════════════════════ INICIALIZAÇÕES ESPECIAIS ═══════════════════════════════
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))
    def forward(self, x):
        xf = x.float()
        rms = xf.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (xf * rms * self.scale.float()).to(x.dtype)

# ═════════════════════════ RoPE ACELERADO (GRADIENTE ATIVO) ═════════════════════════
class AcceleratedRoPERotator(nn.Module):
    """RoPE de segunda ordem com α por cabeça e gradiente fluindo."""
    def __init__(self, head_dim, n_heads, max_len=512, base=10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        self.hd = head_dim
        self.n_heads = n_heads
        half = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
        self.register_buffer('inv_freq', inv_freq)
        self.alpha = nn.Parameter(torch.zeros(n_heads, half))

    @staticmethod
    def _rot(x):
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], -1)

    def forward(self, mu):
        B, T, H, D = mu.shape
        t = torch.arange(T, dtype=torch.float32, device=mu.device)
        theta_linear = t[:, None] * self.inv_freq[None, :]
        poly = t * (t - 1) / 2.0
        theta = theta_linear[None, :, :] + poly[None, :, None] * self.alpha[:, None, :]
        emb = torch.cat([theta, theta], dim=-1)
        c = emb.cos().unsqueeze(0).permute(0, 2, 1, 3).to(mu.dtype)
        s = emb.sin().unsqueeze(0).permute(0, 2, 1, 3).to(mu.dtype)
        return F.normalize(mu * c + self._rot(mu) * s, p=2, dim=-1, eps=1e-8)

# ══════════════════════ MÓDULOS GEOMÉTRICOS DA ATENÇÃO ════════════════════════════════
class XorAttentionBias(nn.Module):
    """XOR algébrico: O(1) em memória via q_sum + k_sum - 2·qk_prod."""
    def __init__(self, head_dim):
        super().__init__()
        self.proj_q = nn.Linear(head_dim, head_dim, bias=False)
        self.proj_k = nn.Linear(head_dim, head_dim, bias=False)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        self.hd = head_dim
        nn.init.xavier_uniform_(self.proj_q.weight, gain=0.5)
        nn.init.xavier_uniform_(self.proj_k.weight, gain=0.5)

    def forward(self, mu_q, mu_k):
        q_bin = torch.sigmoid(self.proj_q(mu_q))
        k_bin = torch.sigmoid(self.proj_k(mu_k))
        q_sum = q_bin.sum(dim=-1, keepdim=True).permute(0, 2, 1, 3)
        k_sum = k_bin.sum(dim=-1, keepdim=True).permute(0, 2, 3, 1)
        q_bin_h = q_bin.permute(0, 2, 1, 3)
        k_bin_h = k_bin.permute(0, 2, 1, 3)
        qk_prod = torch.matmul(q_bin_h, k_bin_h.transpose(-2, -1))
        xor_dist = q_sum + k_sum - 2.0 * qk_prod
        xor_sim = (self.hd - xor_dist) / self.hd
        return xor_sim * self.scale

class TopologicalMERAScore(nn.Module):
    """Invariante topológico via divergência de isometrias convolucionais (MERA-inspired)."""
    def __init__(self, head_dim, heads):
        super().__init__()
        self.heads = heads
        self.isometry = nn.Conv1d(head_dim, head_dim, kernel_size=3, padding=1, groups=head_dim, bias=False)
        nn.init.xavier_uniform_(self.isometry.weight, gain=0.5)
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, mu_q):
        B, T, H, D = mu_q.shape
        x = mu_q.permute(0, 2, 3, 1).reshape(B * H, D, T)
        local_flow = self.isometry(x)
        div = 1.0 - F.cosine_similarity(x, local_flow, dim=1)
        score = div.view(B, H, T).mean(dim=-1, keepdim=True).unsqueeze(-1)
        return score * self.scale

class CyclicTemporalBias(nn.Module):
    """Máscara circular com decaimento exponencial."""
    def __init__(self):
        super().__init__()
        self.decay = nn.Parameter(torch.tensor(2.0))
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, T, device, dtype):
        idx = torch.arange(T, device=device, dtype=torch.float32)
        dist = torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0))
        circ_dist = torch.min(dist, T - dist)
        bias = -self.scale.abs() * torch.exp(-self.decay.abs() * circ_dist)
        return bias.unsqueeze(0).unsqueeze(0).to(dtype)

class ConditionalValueGate(nn.Module):
    """Porta condicionada por hiperplano: inverte se a query está no outro lado."""
    def __init__(self, d_model, n_heads, head_dim):
        super().__init__()
        self.W_gate = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.w = nn.Parameter(torch.randn(head_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.W_gate.weight, gain=0.1)

    def forward(self, x, mu_q):
        B, T, D = x.shape
        H = mu_q.shape[2]
        HD = mu_q.shape[3]
        gate = torch.sigmoid(self.W_gate(x)).view(B, T, H, HD)
        condition = torch.sigmoid(torch.einsum('bthd,d->bth', mu_q, self.w) + self.b)
        cond = condition.unsqueeze(-1)
        return cond * gate + (1 - cond) * (1 - gate)

# ══════════════════════ DISTANCE ATTENTION BIAS (NOVO!) ════════════════════════════
class DistanceAttentionBias(nn.Module):
    """
    Multi-scale distance-based attention bias combining Bessel RBF + Gaussian RBF.
    Inspired by DimeNet++ (Bessel) and SchNet/PaiNN (Gaussian RBF).
    Directly injects 3D geometric information into the attention mechanism.
    """
    def __init__(self, n_rbf_gauss=16, n_rbf_bessel=8, cutoff=5.0, n_heads=8):
        super().__init__()
        self.n_rbf_gauss = n_rbf_gauss
        self.n_rbf_bessel = n_rbf_bessel
        self.cutoff = cutoff
        total_rbf = n_rbf_gauss + n_rbf_bessel

        # Gaussian RBF centers (evenly spaced, like SchNet)
        self.register_buffer('gauss_centers', torch.linspace(0, cutoff, n_rbf_gauss))
        self.register_buffer('gauss_gamma', torch.tensor(10.0))  # Width

        # Bessel RBF: precompute n values (like DimeNet++)
        self.register_buffer('bessel_n', torch.arange(1, n_rbf_bessel + 1, dtype=torch.float32))

        # Project combined RBF features to per-head attention bias
        self.proj = nn.Linear(total_rbf, n_heads, bias=True)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.05)
        nn.init.zeros_(self.proj.bias)

    @staticmethod
    def polynomial_envelope(d, cutoff, p=5):
        """Polynomial cutoff envelope (DimeNet++ style): 1 - 6x^5 + 15x^4 - 10x^3"""
        x = (d / cutoff).clamp(max=1.0)
        env = 1.0 - ((p + 1) * (p + 2) / 2.0) * x**p + p * (p + 2) * x**(p+1) - (p * (p + 1) / 2.0) * x**(p+2)
        return env.clamp(min=0.0)

    def forward(self, distances, fmask=None):
        """
        distances: (B, T, T) pairwise distances (Å)
        fmask: (B, T) padding mask (-10000 for pad, 0 for real)
        Returns: (B, H, T, T) attention bias
        """
        B, T, _ = distances.shape
        d = distances.unsqueeze(-1)  # (B, T, T, 1)

        # --- Gaussian RBF ---
        gauss = torch.exp(-self.gauss_gamma * (d - self.gauss_centers) ** 2)  # (B, T, T, n_gauss)

        # --- Bessel RBF ---
        # RBF_n(d) = sqrt(2/c) * sin(n*pi*d/c) / d
        d_safe = distances.unsqueeze(-1).clamp(min=1e-6)  # (B, T, T, 1)
        bessel = (math.sqrt(2.0 / self.cutoff) *
                  torch.sin(self.bessel_n * math.pi * d_safe / self.cutoff) / d_safe)
        # (B, T, T, n_bessel)

        # --- Combine ---
        rbf = torch.cat([gauss, bessel], dim=-1)  # (B, T, T, total_rbf)

        # --- Polynomial envelope ---
        env = self.polynomial_envelope(distances, self.cutoff)  # (B, T, T)
        rbf = rbf * env.unsqueeze(-1)  # Zero beyond cutoff

        # --- Project to attention bias ---
        bias = self.proj(rbf)  # (B, T, T, H)
        bias = bias.permute(0, 3, 1, 2)  # (B, H, T, T)

        # --- Padding mask ---
        if fmask is not None:
            pad = (fmask < -9000.0)  # (B, T)
            # Mask out rows AND columns involving padding atoms
            pad_row = pad.unsqueeze(1).unsqueeze(3)  # (B, 1, T, 1)
            pad_col = pad.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            bias = bias.masked_fill(pad_row, 0.0)
            bias = bias.masked_fill(pad_col, 0.0)

        return bias


# ══════════════════════ ATENÇÃO GEOMÉTRICA COMPLETA ═══════════════════════════════════
class GeometricAttention(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, dropout=0.1, cfg=None):
        super().__init__()
        self.n_heads = n_heads
        self.hd = head_dim
        self.dp = dropout
        self._sc = 1.0 / math.sqrt(head_dim)

        self.rope = AcceleratedRoPERotator(head_dim, n_heads, cfg.rope_max_seq, cfg.rope_base)
        self.W_mu = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.W_kappa = nn.Linear(d_model, n_heads, bias=True)
        self.Wv = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.Wo = nn.Linear(n_heads * head_dim, d_model, bias=False)

        self.cond_gate = ConditionalValueGate(d_model, n_heads, head_dim) if cfg.use_conditional_gate else None
        if not cfg.use_conditional_gate:
            self.W_gate = nn.Linear(d_model, n_heads * head_dim, bias=False)
            nn.init.xavier_uniform_(self.W_gate.weight, gain=0.1)

        self.tau = nn.Parameter(torch.ones(n_heads) * 2.0)
        self.bias_q = nn.Parameter(torch.zeros(n_heads))

        self.use_xor = cfg.use_xor_bias
        if self.use_xor:
            self.xor_bias = XorAttentionBias(head_dim)
        self.use_topo = cfg.use_topological_bias
        if self.use_topo:
            self.topo_bias = TopologicalMERAScore(head_dim, n_heads)
        self.use_cyclic = cfg.use_cyclic_mask
        if self.use_cyclic:
            self.cyclic_bias = CyclicTemporalBias()

        # NOVO: DistanceAttentionBias
        self.use_distance_bias = cfg.use_distance_bias
        if self.use_distance_bias:
            self.dist_bias = DistanceAttentionBias(
                cfg.dist_n_rbf_gauss, cfg.dist_n_rbf_bessel,
                cfg.dist_cutoff, n_heads
            )

        g = 1.0 / math.sqrt(2)
        for w in [self.W_mu, self.Wv, self.Wo]:
            nn.init.xavier_uniform_(w.weight, gain=g)
        nn.init.xavier_uniform_(self.W_kappa.weight, gain=0.1)
        nn.init.constant_(self.W_kappa.bias, math.log(max(4.0 - 1.0, 1e-4)))

    def get_kappa(self, x):
        return torch.clamp(F.softplus(self.W_kappa(x)) + 1e-4, max=30.0)

    def forward(self, x, fmask=None, tmask=None, distances=None):
        B, T, D = x.shape
        mu = F.normalize(self.W_mu(x).view(B, T, self.n_heads, self.hd), p=2, dim=-1, eps=1e-8)
        mu = self.rope(mu)
        kappa = self.get_kappa(x)

        mu_t = mu.permute(0, 2, 1, 3)
        S_cos = torch.matmul(mu_t, mu_t.transpose(-2, -1))
        kh = kappa.permute(0, 2, 1)
        S_cos = torch.sqrt(kh.unsqueeze(-1) * kh.unsqueeze(-2) + 1e-8) * S_cos
        scores = self.tau.view(1, self.n_heads, 1, 1) * S_cos * self._sc + self.bias_q.view(1, self.n_heads, 1, 1)

        if self.use_xor:
            scores = scores + self.xor_bias(mu, mu)
        if self.use_topo:
            scores = scores + self.topo_bias(mu)
        if self.use_cyclic:
            scores = scores + self.cyclic_bias(T, x.device, x.dtype)
        elif tmask is not None:
            scores = scores + tmask.unsqueeze(1)

        # NOVO: Distance-based attention bias
        if self.use_distance_bias and distances is not None:
            scores = scores + self.dist_bias(distances, fmask)

        # Padding mask
        if fmask is not None:
            pad_bool = (fmask < -9000.0)
            scores = scores.masked_fill(pad_bool.unsqueeze(1).unsqueeze(2), -1e4)

        attn = F.dropout(F.softmax(scores, dim=-1), p=self.dp if self.training else 0.0, training=self.training)

        v = self.Wv(x).view(B, T, self.n_heads, self.hd).permute(0, 2, 1, 3)
        av = attn @ v
        av = av.permute(0, 2, 1, 3)

        if self.cond_gate is not None:
            gate = self.cond_gate(x, mu)
        else:
            gate = torch.sigmoid(self.W_gate(x).view(B, T, self.n_heads, self.hd))

        out = (gate * av).reshape(B, T, self.n_heads * self.hd)
        return self.Wo(out)

# ══════════════════════════ TRANSFORMER BLOCK ═══════════════════════════════════════════
class LocalConvMix(nn.Module):
    def __init__(self, d, k=3, dp=0.1):
        super().__init__()
        self.norm = RMSNorm(d)
        self.dw = nn.Conv1d(d, d, k, padding=(k-1)//2, groups=d, bias=False)
        self.pw = nn.Conv1d(d, d, 1, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.drop = nn.Dropout(dp)
        nn.init.kaiming_normal_(self.dw.weight, nonlinearity='linear')
        nn.init.xavier_uniform_(self.pw.weight)

    def forward(self, x):
        h = self.norm(x).transpose(1, 2).contiguous()
        return x + self.drop(self.act(self.pw(self.dw(h))).transpose(1, 2).contiguous())

class SwiGLU(nn.Module):
    def __init__(self, d, dff, dp=0.1):
        super().__init__()
        self.W_gu = nn.Linear(d, 2 * dff, bias=False)
        self.Wd = nn.Linear(dff, d, bias=False)
        self.drop = nn.Dropout(dp)
        nn.init.kaiming_normal_(self.W_gu.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.Wd.weight, gain=1.0 / math.sqrt(12))

    def forward(self, x):
        g, u = self.W_gu(x).chunk(2, dim=-1)
        return self.drop(self.Wd(F.silu(g) * u))

class TransformerBlock(nn.Module):
    def __init__(self, d, nh, hd, dff, dp, sd, cfg):
        super().__init__()
        self.sd = sd
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.attn = GeometricAttention(d, nh, hd, dp, cfg)
        self.ffn = SwiGLU(d, dff, dp)

    def _drop(self, r):
        if not self.training or self.sd == 0.0:
            return r
        keep = (torch.rand(r.shape[0], 1, 1, device=r.device) > self.sd).float()
        return r * keep / (1.0 - self.sd)

    def forward(self, x, fmask=None, tmask=None, distances=None):
        x = x + self._drop(self.attn(self.norm1(x), fmask, tmask, distances))
        return x + self._drop(self.ffn(self.norm2(x)))

# ════════════════════════════ FUSÃO XOR ESPACIAL ════════════════════════════════════════
class XORSpatialFusion(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj_think = nn.Linear(d, d, bias=False)
        self.proj_seq   = nn.Linear(d, d, bias=False)
        nn.init.xavier_uniform_(self.proj_think.weight, gain=0.1)
        nn.init.xavier_uniform_(self.proj_seq.weight, gain=0.1)

    def forward(self, think, seq_avg):
        x = torch.sigmoid(self.proj_think(think))
        y = torch.sigmoid(self.proj_seq(seq_avg))
        return x * (1 - y) + (1 - x) * y

# ════════════════════════════ CABEÇA DE REGRESSÃO ══════════════════════════════════
class RegressionHead(nn.Module):
    """MultiSampleDropout para regressão: k predictions averaged na inferência."""
    def __init__(self, d, dp=0.1, k=5):
        super().__init__()
        self.k = k
        self.dp = dp
        self.fc1 = nn.Linear(d, d)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(d, 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.02)
        nn.init.zeros_(self.fc2.bias)

    def _once(self, x):
        return self.fc2(F.dropout(self.act(self.fc1(x)), p=self.dp, training=True))

    def forward(self, x):
        if self.training:
            return self._once(x).squeeze(-1)
        else:
            return torch.stack([self._once(x) for _ in range(self.k)]).mean(0).squeeze(-1)

# ════════════════════════ GRAFOPROPAGATION v27 — QM9 RAD VERSION ═════════════════════════
class GrafoPropagationGeoQM9(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, L = cfg.d_model, cfg.n_layers

        # --- Input projection ---
        # Input: [fixed_phys(27) | mol_dependent(20) | z_embed(16)] = 63 → d_model
        self.input_proj = nn.Linear(cfg.feature_dim, d, bias=True)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        self.embed_scale = d ** 0.5
        self.embed_drop = nn.Dropout(cfg.dropout)

        # --- Learnable Z embedding (like PaiNN/SchNet/DimeNet++) ---
        self.z_embed = nn.Embedding(10, cfg.n_z_embed)  # Z=0..9
        nn.init.normal_(self.z_embed.weight, std=0.02)

        # --- Fixed feature lookup table (not learned, just stored) ---
        self.register_buffer('fixed_feat_table', torch.from_numpy(FIXED_FEAT_TABLE))

        self.conv_mix = LocalConvMix(d, cfg.conv_kernel, cfg.dropout)

        sd_list = [cfg.stoch_depth * i / max(L - 1, 1) for i in range(L)]
        self.blocks = nn.ModuleList([
            TransformerBlock(d, cfg.n_heads, cfg.head_dim, cfg.d_ff,
                             cfg.dropout, sd_list[i], cfg)
            for i in range(L)
        ])

        self.final_norm = RMSNorm(d)
        self.fusion = XORSpatialFusion(d)
        self.head = RegressionHead(d, cfg.dropout, k=5)

    def _compute_pairwise_distances(self, positions, fmask=None):
        """
        Compute pairwise distances from 3D positions.
        positions: (B, T, 3)
        fmask: (B, T) padding mask
        Returns: (B, T, T) pairwise distances
        """
        # positions for padding atoms are 0 → distance to padding = 0 (misleading)
        # Fix: set padding positions far away
        if fmask is not None:
            pad = (fmask < -9000.0)  # (B, T)
            # Move padding atoms to (100, 100, 100) so distances are large
            pos_fixed = positions.clone()
            pos_fixed[pad] = pos_fixed[pad] + 100.0
        else:
            pos_fixed = positions

        diff = pos_fixed.unsqueeze(2) - pos_fixed.unsqueeze(1)  # (B, T, T, 3)
        dists = (diff ** 2).sum(dim=-1).sqrt()  # (B, T, T)
        return dists

    def encode(self, emb, fmask, distances=None):
        B, T, D = emb.shape
        x = emb
        x = self.conv_mix(x)
        for block in self.blocks:
            x = block(x, fmask, distances=distances)
        x = self.final_norm(x)
        valid = (~(fmask < -9000.0)).float().unsqueeze(-1)  # (B, T, 1)
        seq_avg = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        x_masked = x.clone()
        x_masked[~valid.expand(-1, -1, D).bool()] = -1e9
        think = x_masked.max(dim=1).values
        fused = self.fusion(think, seq_avg)
        return fused

    def forward(self, atom_features, fmask, z_indices, positions):
        """
        atom_features: (B, T, 47) pre-normalized per-atom features
                       = [fixed_phys(27) | mol_dependent(20)]
        fmask: (B, T) padding mask
        z_indices: (B, T) LongTensor atomic numbers (for learnable embedding)
        positions: (B, T, 3) 3D positions (for distance-based attention bias)
        """
        # Learnable Z embedding (like PaiNN/SchNet/DimeNet++)
        z_emb = self.z_embed(z_indices)  # (B, T, 16)

        # Combine: normalized features (47) + learnable embedding (16) = 63
        combined = torch.cat([atom_features, z_emb], dim=-1)  # (B, T, 63)

        # Project to d_model
        emb = self.embed_drop(self.input_proj(combined) * self.embed_scale)

        # Compute pairwise distances for attention bias
        dists = self._compute_pairwise_distances(positions, fmask)

        # Encode
        fused = self.encode(emb, fmask, distances=dists)
        return self.head(fused)


# ═══════════════════════════ DATA LOADING ═══════════════════════════════════════════
def precompute_qm9_data(cfg):
    """
    Precompute raw QM9 data (z_indices, positions, targets) with caching.
    Cache v4: stores RAW data only — NO normalization, NO CoulombMatrix.
    Features are computed AFTER the train/val/test split in main().
    """
    cache_path = cfg.cache_path
    if os.path.exists(cache_path):
        log(f'Loading cached QM9 data from {cache_path}…', 'DATA')
        data = torch.load(cache_path, map_location='cpu', weights_only=False)
        if 'z_indices' not in data:
            log('  Cache antigo detectado — a recalcular com formato v4 (raw z+pos)…', 'WARN')
            os.remove(cache_path)
        else:
            log(f'  Cache loaded: {data["z_indices"].shape[0]} molecules', 'DATA')
            return data

    log('Computing raw QM9 data (z_indices + positions + targets)…', 'DATA')

    # ── Detect backends ──
    USE_PYG = False
    USE_DIRECT = True

    try:
        from torch_geometric.datasets import QM9
        USE_PYG = True
        log('  Backend disponível: torch_geometric (PyG)', 'DATA')
    except ImportError:
        log('  torch_geometric não encontrado, a tentar pip install…', 'WARN')
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch_geometric', '-q'],
                         capture_output=True, timeout=120)
            try:
                from torch_geometric.datasets import QM9
                USE_PYG = True
                log('  Backend disponível: torch_geometric (PyG) — instalado via pip', 'DATA')
            except ImportError:
                pass
        except Exception:
            pass

    log(f'  Backends activos: PyG={USE_PYG}, Direct={USE_DIRECT}', 'DATA')

    M = cfg.max_atoms

    # ═══════════════ MÉTODO 1: PyG ═══════════════
    all_z_indices = None
    all_positions = None
    all_targets = None
    all_n_atoms = None

    if USE_PYG:
        try:
            log('  A carregar QM9 via torch_geometric…', 'DATA')

            import builtins
            _orig_import = builtins.__import__
            _rdkit_hidden = False

            try:
                import rdkit  # noqa
                _rdkit_hidden = True
                log('  rdkit detectado — a esconder do PyG (FIX v12)', 'DATA')
            except ImportError:
                pass

            if _rdkit_hidden:
                def _custom_import(name, globals_dict=None, locals_dict=None, fromlist=(), level=0):
                    if name == 'rdkit' or name.startswith('rdkit.'):
                        raise ImportError('rdkit temporarily hidden')
                    return _orig_import(name, globals_dict, locals_dict, fromlist, level)
                builtins.__import__ = _custom_import

            try:
                import shutil
                processed_dir = os.path.join(cfg.qm9_root, 'processed')
                if os.path.exists(processed_dir):
                    try:
                        shutil.rmtree(processed_dir)
                    except Exception:
                        pass

                from torch_geometric.datasets import QM9
                dataset = QM9(root=cfg.qm9_root)
                n_total = len(dataset)
                log(f'  PyG QM9 carregado: {n_total} moléculas', 'DATA')

                all_z_indices = np.zeros((n_total, M), dtype=np.int64)
                all_positions = np.zeros((n_total, M, 3), dtype=np.float32)
                all_n_atoms = np.zeros(n_total, dtype=np.int32)
                all_targets = np.zeros(n_total, dtype=np.float32)

                t0 = time.time()
                for idx in range(n_total):
                    data = dataset[idx]
                    z = data.z.numpy()  # (n,) Long
                    pos = data.pos.numpy()  # (n, 3)
                    n = len(z)
                    all_n_atoms[idx] = n
                    all_targets[idx] = data.y[0, cfg.gap_idx].item()
                    all_z_indices[idx, :n] = z
                    all_positions[idx, :n] = pos

                    if (idx + 1) % 20000 == 0 or idx == n_total - 1:
                        elapsed = time.time() - t0
                        rate = (idx + 1) / elapsed
                        eta = (n_total - idx - 1) / rate
                        log(f'  Raw data: {idx+1}/{n_total} ({rate:.0f} mol/s, ETA {eta:.0f}s)', 'DATA')
            finally:
                if _rdkit_hidden:
                    builtins.__import__ = _orig_import

        except Exception as e_pyg:
            log(f'  PyG QM9 falhou: {e_pyg}', 'WARN')
            all_z_indices = None
            USE_PYG = False

    # ═══════════════ MÉTODO 2: Download directo ═══════════════
    if all_z_indices is None and USE_DIRECT:
        import zipfile
        import urllib.request
        import io

        log('  Método 2: Download directo / parse manual de dados QM9', 'DATA')

        ATOMIC_NUM = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
                      'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
                      'S': 16, 'Cl': 17, 'Ar': 18}
        RAW_GAP_IDX = 9

        all_z_list = []
        all_pos_list = []
        all_natoms_list = []
        all_tgt_list = []
        parsed_ok = False
        skipped = 0

        raw_dir = os.path.join(cfg.qm9_root, 'raw')
        os.makedirs(raw_dir, exist_ok=True)

        existing_xyz = os.path.join(raw_dir, 'dsgdb9nsd.xyz')
        existing_tarbz2 = os.path.join(raw_dir, 'dsgdb9nsd.xyz.tar.bz2')
        existing_bz2 = os.path.join(raw_dir, 'dsgdb9nsd.xyz.bz2')

        def _parse_dsgdb9nsd_text(raw_text):
            lines_all = raw_text.strip().split('\n')
            line_ptr = 0
            t0 = time.time()
            count = 0
            while line_ptr < len(lines_all):
                line = lines_all[line_ptr].strip()
                if not line:
                    line_ptr += 1
                    continue
                try:
                    n_atoms = int(line.split()[0])
                    if n_atoms < 1 or n_atoms > 100:
                        line_ptr += 1
                        continue
                except (ValueError, IndexError):
                    line_ptr += 1
                    continue
                if line_ptr + 1 >= len(lines_all):
                    break
                props = lines_all[line_ptr + 1].strip().split()
                if len(props) < 10:
                    line_ptr += 1
                    continue
                try:
                    gap_val = float(props[RAW_GAP_IDX])
                except (ValueError, IndexError):
                    line_ptr += 1
                    continue
                z_list = []
                pos_list = []
                valid = True
                for i in range(n_atoms):
                    atom_ptr = line_ptr + 2 + i
                    if atom_ptr >= len(lines_all):
                        valid = False
                        break
                    parts = lines_all[atom_ptr].strip().split()
                    if len(parts) < 4:
                        valid = False
                        break
                    sym = parts[0]
                    if sym not in ATOMIC_NUM:
                        valid = False
                        break
                    z_list.append(ATOMIC_NUM[sym])
                    try:
                        pos_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except (ValueError, IndexError):
                        valid = False
                        break
                if valid and len(z_list) == n_atoms:
                    all_z_list.append(z_list)
                    all_pos_list.append(pos_list)
                    all_natoms_list.append(n_atoms)
                    all_tgt_list.append(gap_val)
                    count += 1
                    if count % 20000 == 0:
                        elapsed = time.time() - t0
                        log(f'  Parsed: {count} molecules ({count/elapsed:.0f} mol/s)', 'DATA')
                line_ptr += 2 + n_atoms
            return count

        # Try existing dsgdb9nsd.xyz
        if os.path.exists(existing_xyz):
            try:
                with open(existing_xyz, 'r') as f:
                    raw_text = f.read()
                cnt = _parse_dsgdb9nsd_text(raw_text)
                if cnt > 0:
                    parsed_ok = True
                log(f'  Parsed {cnt} molecules from existing dsgdb9nsd.xyz', 'DATA')
            except Exception as e:
                log(f'  Falhou: {e}', 'WARN')

        # Try .tar.bz2
        if not parsed_ok and os.path.exists(existing_tarbz2):
            import tarfile
            try:
                with tarfile.open(existing_tarbz2, 'r:bz2') as tar:
                    for member in tar.getmembers():
                        if member.name.endswith('.xyz'):
                            f = tar.extractfile(member)
                            if f:
                                raw_text = f.read().decode('ascii')
                                cnt = _parse_dsgdb9nsd_text(raw_text)
                                if cnt > 0:
                                    parsed_ok = True
                            break
            except Exception as e:
                log(f'  tar.bz2 falhou: {e}', 'WARN')

        # Try .bz2
        if not parsed_ok and os.path.exists(existing_bz2):
            try:
                with open(existing_bz2, 'rb') as f:
                    raw_text = bz2.decompress(f.read()).decode('ascii')
                cnt = _parse_dsgdb9nsd_text(raw_text)
                if cnt > 0:
                    parsed_ok = True
            except Exception as e:
                log(f'  .bz2 falhou: {e}', 'WARN')

        # Download from PyG
        if not parsed_ok:
            qm9_url = 'https://data.pyg.org/datasets/qm9_v3.zip'
            raw_file = os.path.join(raw_dir, 'qm9_v3.zip')
            if not os.path.exists(raw_file):
                log(f'  Downloading {qm9_url}…', 'DATA')
                try:
                    urllib.request.urlretrieve(qm9_url, raw_file)
                    log(f'  Download complete', 'DATA')
                except Exception as e:
                    log(f'  Download failed: {e}', 'WARN')
                    raw_file = None

            if raw_file is not None:
                try:
                    with zipfile.ZipFile(raw_file, 'r') as zf:
                        pt_files = [f for f in zf.namelist() if f.endswith('.pt')]
                        if pt_files:
                            for pt_name in pt_files:
                                try:
                                    raw_data = torch.load(io.BytesIO(zf.open(pt_name).read()),
                                                         map_location='cpu', weights_only=False)
                                    data_list = None

                                    if isinstance(raw_data, dict) and 'data' in raw_data and 'slices' in raw_data:
                                        collated_data = raw_data['data']
                                        slices = raw_data['slices']
                                        n_total = len(slices.get('z', slices.get(list(slices.keys())[0], []))) - 1
                                        try:
                                            from torch_geometric.data import InMemoryDataset, Data
                                            import tempfile, shutil as _shutil
                                            tmp_root = tempfile.mkdtemp(prefix='qm9_pt_')
                                            tmp_proc = os.path.join(tmp_root, 'processed')
                                            os.makedirs(tmp_proc, exist_ok=True)
                                            torch.save(raw_data, os.path.join(tmp_proc, 'data_v3.pt'))
                                            class _TmpQM9(InMemoryDataset):
                                                def __init__(self, root):
                                                    super().__init__(root)
                                                    self.load(self.processed_paths[0])
                                                @property
                                                def raw_file_names(self): return []
                                                @property
                                                def processed_file_names(self): return ['data_v3.pt']
                                                def process(self): pass
                                            ds = _TmpQM9(tmp_root)
                                            data_list = [ds[i] for i in range(len(ds))]
                                            _shutil.rmtree(tmp_root, ignore_errors=True)
                                        except Exception:
                                            from torch_geometric.data import Data as _Data
                                            data_list = []
                                            for i in range(n_total):
                                                d = _Data()
                                                for key in collated_data.keys():
                                                    s = slices.get(key, None)
                                                    if s is not None and i + 1 < len(s):
                                                        start, end = int(s[i]), int(s[i + 1])
                                                        d[key] = collated_data[key][start:end]
                                                    else:
                                                        d[key] = collated_data[key]
                                                data_list.append(d)

                                    elif isinstance(raw_data, tuple) and len(raw_data) >= 2:
                                        collated_data, slices = raw_data[0], raw_data[1]
                                        n_total = len(slices.get('z', slices.get(list(slices.keys())[0], []))) - 1
                                        from torch_geometric.data import InMemoryDataset, Data
                                        import tempfile, shutil as _shutil
                                        tmp_root = tempfile.mkdtemp(prefix='qm9_pt_')
                                        tmp_proc = os.path.join(tmp_root, 'processed')
                                        os.makedirs(tmp_proc, exist_ok=True)
                                        torch.save({'data': collated_data, 'slices': slices},
                                                  os.path.join(tmp_proc, 'data_v3.pt'))
                                        class _TmpQM9(InMemoryDataset):
                                            def __init__(self, root):
                                                super().__init__(root)
                                                self.load(self.processed_paths[0])
                                            @property
                                            def raw_file_names(self): return []
                                            @property
                                            def processed_file_names(self): return ['data_v3.pt']
                                            def process(self): pass
                                        ds = _TmpQM9(tmp_root)
                                        data_list = [ds[i] for i in range(len(ds))]
                                        _shutil.rmtree(tmp_root, ignore_errors=True)

                                    elif isinstance(raw_data, list):
                                        data_list = raw_data

                                    if data_list is not None:
                                        t0 = time.time()
                                        for idx in range(len(data_list)):
                                            d = data_list[idx]
                                            if isinstance(d, dict):
                                                z_val = d.get('z', d.get('atomic_numbers'))
                                                pos_val = d.get('pos', d.get('positions'))
                                                y_val = d.get('y')
                                            elif hasattr(d, 'z'):
                                                z_val = d.z
                                                pos_val = d.pos if hasattr(d, 'pos') else None
                                                y_val = d.y if hasattr(d, 'y') else None
                                            else:
                                                continue
                                            if z_val is None or pos_val is None or y_val is None:
                                                continue
                                            z_np = z_val.numpy() if isinstance(z_val, torch.Tensor) else np.array(z_val)
                                            pos_np = pos_val.numpy() if isinstance(pos_val, torch.Tensor) else np.array(pos_val)
                                            y_np = y_val.numpy() if isinstance(y_val, torch.Tensor) else np.array(y_val)
                                            n = len(z_np)
                                            all_z_list.append(z_np.tolist())
                                            all_pos_list.append(pos_np.tolist())
                                            all_natoms_list.append(n)
                                            gap_val = float(y_np[0, cfg.gap_idx]) if y_np.ndim >= 2 else float(y_np[cfg.gap_idx])
                                            all_tgt_list.append(gap_val)
                                            if (idx + 1) % 20000 == 0:
                                                elapsed = time.time() - t0
                                                log(f'  .pt: {idx+1}/{len(data_list)} ({(idx+1)/elapsed:.0f} mol/s)', 'DATA')
                                        parsed_ok = len(all_z_list) > 0
                                        break
                                except Exception as e:
                                    log(f'  Falhou carregar {pt_name}: {e}', 'WARN')

                        # Also try .xyz files and .bz2 in ZIP
                        if not parsed_ok:
                            for member_name in zf.namelist():
                                if member_name.endswith('.bz2'):
                                    try:
                                        raw_text = bz2.decompress(zf.open(member_name).read()).decode('ascii')
                                        cnt = _parse_dsgdb9nsd_text(raw_text)
                                        if cnt > 0:
                                            parsed_ok = True
                                        break
                                    except Exception:
                                        pass

                except Exception as e_zip:
                    log(f'  Erro ao processar ZIP: {e_zip}', 'WARN')

        # Figshare fallback
        if not parsed_ok:
            log('  A tentar figshare para dsgdb9nsd.xyz.tar.bz2…', 'DATA')
            figshare_url = 'https://ndownloader.figshare.com/files/3195389'
            tarbz2_file = os.path.join(raw_dir, 'dsgdb9nsd.xyz.tar.bz2')
            if not os.path.exists(tarbz2_file):
                try:
                    req = urllib.request.Request(figshare_url, headers={'User-Agent': 'Mozilla/5.0'})
                    resp = urllib.request.urlopen(req, timeout=120)
                    for _ in range(5):
                        if resp.status in (301, 302, 303, 307, 308):
                            redirect_url = resp.headers.get('Location')
                            if redirect_url:
                                req = urllib.request.Request(redirect_url, headers={'User-Agent': 'Mozilla/5.0'})
                                resp = urllib.request.urlopen(req, timeout=120)
                            else:
                                break
                        else:
                            break
                    data_bytes = resp.read()
                    with open(tarbz2_file, 'wb') as f:
                        f.write(data_bytes)
                except Exception as e:
                    log(f'  figshare download failed: {e}', 'ERROR')

            if os.path.exists(tarbz2_file):
                import tarfile
                extracted_xyz = os.path.join(raw_dir, 'dsgdb9nsd.xyz')
                if not os.path.exists(extracted_xyz):
                    try:
                        with tarfile.open(tarbz2_file, 'r:bz2') as tar:
                            for member in tar.getmembers():
                                if member.name.endswith('.xyz'):
                                    tar.extract(member, raw_dir)
                                    extracted_path = os.path.join(raw_dir, member.name)
                                    if extracted_path != extracted_xyz:
                                        os.rename(extracted_path, extracted_xyz)
                                    break
                    except Exception as e:
                        log(f'  tar.bz2 extraction failed: {e}', 'ERROR')

                if os.path.exists(extracted_xyz):
                    try:
                        with open(extracted_xyz, 'r') as f:
                            raw_text = f.read()
                        cnt = _parse_dsgdb9nsd_text(raw_text)
                        if cnt > 0:
                            parsed_ok = True
                    except Exception as e:
                        log(f'  Falhou ao processar: {e}', 'ERROR')

        if not parsed_ok:
            raise RuntimeError('Could not parse QM9 data from any source!')

        n_total = len(all_z_list)
        log(f'  Parsed {n_total} molecules total', 'DATA')

        # Convert lists to arrays
        all_z_indices = np.zeros((n_total, M), dtype=np.int64)
        all_positions = np.zeros((n_total, M, 3), dtype=np.float32)
        all_n_atoms = np.zeros(n_total, dtype=np.int32)
        all_targets = np.zeros(n_total, dtype=np.float32)

        for idx in range(n_total):
            z_list = all_z_list[idx]
            pos_list = all_pos_list[idx]
            n = min(len(z_list), M)
            all_z_indices[idx, :n] = z_list[:n]
            all_positions[idx, :n] = pos_list[:n]
            all_n_atoms[idx] = n
            all_targets[idx] = all_tgt_list[idx]

    # ── Final check ──
    if all_z_indices is None:
        raise RuntimeError('Could not load QM9 data from any backend!')

    n_total = len(all_z_indices)
    log(f'  Loaded {n_total} molecules successfully', 'DATA')

    # Build padding mask
    all_fmask = np.zeros((n_total, M), dtype=np.float32)
    for idx in range(n_total):
        n = int(all_n_atoms[idx])
        if n < M:
            all_fmask[idx, n:] = -10000.0

    # Center positions by molecule centroid (IMPORTANT for symmetry)
    for idx in range(n_total):
        n = int(all_n_atoms[idx])
        if n > 0:
            centroid = all_positions[idx, :n].mean(axis=0)
            all_positions[idx, :n] = all_positions[idx, :n] - centroid

    result = {
        'z_indices': torch.from_numpy(all_z_indices),
        'positions': torch.from_numpy(all_positions),
        'fmask': torch.from_numpy(all_fmask),
        'targets_raw': torch.from_numpy(all_targets),
        'n_atoms': torch.from_numpy(all_n_atoms),
    }

    try:
        torch.save(result, cache_path)
        log(f'  Cache saved to {cache_path}', 'DATA')
    except Exception as e:
        log(f'  Warning: could not save cache: {e}', 'WARN')

    return result


def compute_all_mol_features(z_indices, positions, n_atoms, max_atoms, batch_size=5000):
    """
    Compute molecule-dependent per-atom features for ALL molecules.
    Called AFTER the train/val/test split, so normalization can be done properly.

    z_indices: (N, M) LongTensor
    positions: (N, M, 3) FloatTensor
    n_atoms: (N,) IntTensor
    max_atoms: 29

    Returns: (N, M, 20) FloatTensor of molecule-dependent features
    """
    N = z_indices.shape[0]
    M = max_atoms
    all_feats = np.zeros((N, M, CFG.n_mol_feats), dtype=np.float32)

    log(f'Computing molecule-dependent features for {N} molecules…', 'DATA')
    t0 = time.time()

    z_np = z_indices.numpy()
    pos_np = positions.numpy()
    nat_np = n_atoms.numpy()

    for idx in range(N):
        n = int(nat_np[idx])
        if n == 0:
            continue
        z = z_np[idx, :n].astype(np.float64)
        pos = pos_np[idx, :n].astype(np.float64)
        feats = compute_mol_dependent_features(z, pos, M)
        all_feats[idx] = feats.astype(np.float32)

        if (idx + 1) % batch_size == 0 or idx == N - 1:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (N - idx - 1) / rate
            log(f'  Features: {idx+1}/{N} ({rate:.0f} mol/s, ETA {eta:.0f}s)', 'DATA')

    return torch.from_numpy(all_feats)


# ═══════════════════════════ QM9 DATASET ═══════════════════════════════════════════
class QM9Dataset(Dataset):
    """Dataset with per-atom features, z_indices, positions, and targets."""
    def __init__(self, mol_features, fmask, targets, z_indices, positions, indices=None):
        if indices is not None:
            self.mol_features = mol_features[indices]
            self.fmask = fmask[indices]
            self.targets = targets[indices]
            self.z_indices = z_indices[indices]
            self.positions = positions[indices]
        else:
            self.mol_features = mol_features
            self.fmask = fmask
            self.targets = targets
            self.z_indices = z_indices
            self.positions = positions

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return (self.mol_features[i], self.fmask[i], self.targets[i],
                self.z_indices[i], self.positions[i])

# ═══════════════════════════ OPTIMIZADORES & SCHEDULER ═══════════════════════════════
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.lerp_(m.float(), 1.0 - self.decay)
        for sb, mb in zip(self.shadow.buffers(), model.buffers()):
            sb.copy_(mb)

class Lookahead(torch.optim.Optimizer):
    def __init__(self, base, k=6, alpha=0.5):
        self._b = base
        self.k = k
        self.alpha = alpha
        self._steps = 0
        self._slow = {}
        self.param_groups = base.param_groups
        self.defaults = getattr(base, 'defaults', {})

    @property
    def state(self):
        return self._b.state

    def zero_grad(self, set_to_none=True):
        self._b.zero_grad(set_to_none=set_to_none)

    def _ensure(self):
        if self._slow:
            return
        for g in self.param_groups:
            for p in g['params']:
                self._slow[id(p)] = p.data.clone().detach()

    def step(self, closure=None):
        loss = self._b.step(closure)
        self._steps += 1
        self._ensure()
        if self._steps % self.k == 0:
            for g in self.param_groups:
                for p in g['params']:
                    s = self._slow[id(p)]
                    s.add_(self.alpha * (p.data - s))
                    p.data.copy_(s)
        return loss

class AWP:
    def __init__(self, model, scaler, eps=0.003, lr=0.005):
        self.model = model
        self.scaler = scaler
        self.eps = eps
        self.lr = lr
        self._bk = {}
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
                    self._bk[n] = p.data.clone()
                    p.data.add_((self.lr * g / (gn + 1e-8)).clamp_(-self.eps, self.eps).to(p.dtype))
        self._on = True

    def restore(self):
        for n, p in self.model.named_parameters():
            if n in self._bk:
                p.data.copy_(self._bk[n])
        self._bk.clear()
        self._on = False

def _gc_hook(g):
    return g - g.mean(tuple(range(1, g.dim())), keepdim=True) if g.dim() > 1 else g

def register_gc(model):
    return [p.register_hook(_gc_hook) for n, p in model.named_parameters()
            if p.requires_grad and p.dim() > 1 and 'input_proj' not in n and 'z_embed' not in n]

class WarmupCosineLR:
    def __init__(self, total, wf, mf):
        self.T = max(total, 1)
        self.W = max(int(wf * total), 1)
        self.mf = mf

    def factor(self, step):
        if step < self.W:
            return step / self.W
        p = min(max((step - self.W) / max(self.T - self.W, 1), 0.0), 1.0)
        return self.mf + (1.0 - self.mf) * 0.5 * (1.0 + math.cos(math.pi * p))

# ═══════════════════════════ MÉTRICAS DE REGRESSÃO ═══════════════════════════════
def compute_regression_metrics(pred, target, y_mean=0.0, y_std=1.0):
    pred_orig = pred * y_std + y_mean
    target_orig = target * y_std + y_mean
    diff = pred_orig - target_orig
    abs_diff = diff.abs()
    mae = abs_diff.mean().item()
    rmse = diff.pow(2).mean().sqrt().item()
    ss_res = diff.pow(2).sum().item()
    ss_tot = (target_orig - target_orig.mean()).pow(2).sum().item()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    non_zero = target_orig.abs() > 0.01
    mape = (abs_diff[non_zero] / target_orig[non_zero].abs()).mean().item() * 100.0 if non_zero.sum() > 0 else 0.0
    median_ae = abs_diff.median().item()
    max_ae = abs_diff.max().item()
    pred_m = pred_orig - pred_orig.mean()
    tgt_m = target_orig - target_orig.mean()
    pearson_r = ((pred_m * tgt_m).sum() / (pred_m.norm() * tgt_m.norm()).clamp(min=1e-8)).item()
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape, 'median_ae': median_ae,
            'max_ae': max_ae, 'pearson_r': pearson_r}

# ═══════════════════════════ TREINO E AVALIAÇÃO ═══════════════════════════════════════
def train_epoch(model, ema, optimizer, scaler, loader, awp, cfg, epoch, gstep, lr_sched, base_opt):
    model.train()
    n = len(loader)
    t0 = time.time()
    total_loss = 0.0
    total_mae_norm = 0.0
    total_mae_ev = 0.0
    total_samples = 0
    grad_norm_accum = 0.0
    grad_norm_count = 0
    optimizer.zero_grad(set_to_none=True)

    for step, (feat, fm, target, z_idx, pos) in enumerate(loader):
        feat = feat.to(cfg.device, non_blocking=True)
        fm = fm.to(cfg.device, non_blocking=True)
        target = target.to(cfg.device, non_blocking=True)
        z_idx = z_idx.to(cfg.device, non_blocking=True)
        pos = pos.to(cfg.device, non_blocking=True)

        use_mx = random.random() < cfg.mixup_prob
        with autocast(device_type=cfg.device.type, dtype=cfg.amp_dtype, enabled=(cfg.device.type == 'cuda')):
            if use_mx:
                lam = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                idx2 = torch.randperm(feat.size(0), device=cfg.device)
                mixed_feat = lam * feat + (1 - lam) * feat[idx2]
                mixed_pos = lam * pos + (1 - lam) * pos[idx2]
                mixed_target = lam * target + (1 - lam) * target[idx2]
                # z_indices: use the dominant atom's z (from lam)
                mixed_z = z_idx if lam >= 0.5 else z_idx[idx2]
                pred = model(mixed_feat, fm, mixed_z, mixed_pos)
                loss = F.smooth_l1_loss(pred, mixed_target, beta=cfg.huber_delta)
            else:
                pred = model(feat, fm, z_idx, pos)
                loss = F.smooth_l1_loss(pred, target, beta=cfg.huber_delta)

        scaler.scale(loss / cfg.grad_accum).backward()

        if (step + 1) % cfg.grad_accum == 0:
            scaler.unscale_(optimizer)
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            grad_norm_accum += min(total_norm, 1000.0)
            grad_norm_count += 1

            if epoch >= cfg.awp_start_ep:
                awp.perturb()
                with autocast(device_type=cfg.device.type, dtype=cfg.amp_dtype, enabled=(cfg.device.type == 'cuda')):
                    scaler.scale(
                        F.smooth_l1_loss(model(feat, fm, z_idx, pos), target, beta=cfg.huber_delta) / cfg.grad_accum
                    ).backward()
                awp.restore()

            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            gstep += 1
            base_opt.param_groups[0]['lr'] = cfg.base_lr_max * lr_sched.factor(gstep)

        bs = target.size(0)
        total_loss += loss.item() * bs
        total_mae_norm += (pred - target).abs().sum().item()
        total_mae_ev += ((pred - target) * cfg._y_std).abs().sum().item()
        total_samples += bs

        if step % cfg.attn_log_freq == 0 or step == n - 1:
            ela = time.time() - t0
            lr_now = base_opt.param_groups[0]['lr']
            avg_loss = total_loss / max(total_samples, 1)
            avg_mae_ev = total_mae_ev / max(total_samples, 1)
            avg_mae_norm = total_mae_norm / max(total_samples, 1)
            gn = grad_norm_accum / max(grad_norm_count, 1) if grad_norm_count > 0 else 0.0
            log(f'ep={epoch:3d} │ step={step:04d}/{n} │ lr={lr_now:.6f} │ '
                f'loss={loss.item():.5f} │ mae(norm)={avg_mae_norm:.4f} │ '
                f'mae(eV)={avg_mae_ev:.4f} │ gnorm={gn:.3f} │ {ela:.1f}s', 'TRAIN')

    avg_loss = total_loss / max(total_samples, 1)
    avg_mae_norm = total_mae_norm / max(total_samples, 1)
    avg_mae_ev = total_mae_ev / max(total_samples, 1)
    avg_gnorm = grad_norm_accum / max(grad_norm_count, 1) if grad_norm_count > 0 else 0.0
    elapsed = round(time.time() - t0, 1)

    return {'loss': avg_loss, 'mae_norm': avg_mae_norm, 'mae_ev': avg_mae_ev,
            'grad_norm': avg_gnorm, 'lr': base_opt.param_groups[0]['lr'], 'time_s': elapsed}, gstep


@torch.no_grad()
def evaluate(model, loader, cfg, y_mean=0.0, y_std=1.0):
    model.eval()
    all_pred = []
    all_target = []
    total_loss = 0.0
    total_samples = 0
    t0 = time.time()

    for feat, fm, target, z_idx, pos in loader:
        feat = feat.to(cfg.device)
        fm = fm.to(cfg.device)
        target = target.to(cfg.device)
        z_idx = z_idx.to(cfg.device)
        pos = pos.to(cfg.device)

        with autocast(device_type=cfg.device.type, dtype=cfg.amp_dtype, enabled=(cfg.device.type == 'cuda')):
            pred = model(feat, fm, z_idx, pos)
            loss = F.smooth_l1_loss(pred, target, beta=1.0)

        total_loss += loss.item() * target.size(0)
        total_samples += target.size(0)
        all_pred.append(pred.cpu())
        all_target.append(target.cpu())

    all_pred = torch.cat(all_pred)
    all_target = torch.cat(all_target)
    metrics = compute_regression_metrics(all_pred, all_target, y_mean, y_std)
    metrics['time_s'] = round(time.time() - t0, 1)
    metrics['loss'] = total_loss / max(total_samples, 1)

    pred_orig = all_pred * y_std + y_mean
    tgt_orig = all_target * y_std + y_mean
    metrics['pred_mean'] = pred_orig.mean().item()
    metrics['pred_std'] = pred_orig.std().item()
    metrics['target_mean'] = tgt_orig.mean().item()
    metrics['target_std'] = tgt_orig.std().item()

    return metrics


def print_epoch_table(epoch, tr_s, va_s, best_mae, cfg):
    if not HAS_RICH:
        log(f'EPOCH {epoch:3d} │ tr_loss={tr_s["loss"]:.6f} │ tr_mae={tr_s["mae_ev"]:.4f} eV │ '
            f'va_mae={va_s["mae"]:.4f} eV │ va_rmse={va_s["rmse"]:.4f} eV │ '
            f'va_R2={va_s["r2"]:.4f} │ best={best_mae:.4f} eV │ {tr_s["time_s"]}s', 'METRIC')
        return
    t = Table(title=f'Epoch {epoch:3d}/{cfg.epochs} · GrafoPropagation {cfg.VERSION}',
              box=box.HEAVY_EDGE, show_lines=True, title_style='bold cyan')
    t.add_column('Split', style='bold', width=6)
    t.add_column('MAE(eV)', style='bold green', width=10, justify='right')
    t.add_column('RMSE(eV)', style='green', width=10, justify='right')
    t.add_column('R²', style='green', width=8, justify='right')
    t.add_column('MAPE%', style='green', width=7, justify='right')
    t.add_column('MedAE', style='green', width=8, justify='right')
    t.add_column('MaxAE', style='green', width=8, justify='right')
    t.add_column('Pearson', style='green', width=8, justify='right')
    t.add_column('Loss', style='yellow', width=10, justify='right')
    t.add_column('LR', style='dim', width=10, justify='right')
    t.add_column('GNorm', style='dim', width=7, justify='right')
    t.add_column('Time', style='dim', width=6, justify='right')

    t.add_row('Train', f'{tr_s["mae_ev"]:.4f}', '—', '—', '—', '—', '—', '—',
              f'{tr_s["loss"]:.6f}', f'{tr_s["lr"]:.2e}', f'{tr_s["grad_norm"]:.3f}', f'{tr_s["time_s"]}s')

    is_best = va_s['mae'] <= best_mae
    r2_style = 'bold green' if va_s['r2'] > 0.9 else ('yellow' if va_s['r2'] > 0.7 else 'red')
    mae_style = 'bold green' if va_s['mae'] < 0.1 else ('yellow' if va_s['mae'] < 0.2 else 'red')
    split_label = '[bold blue]Val★[/bold blue]' if is_best else 'Val'

    t.add_row(split_label, f'[{mae_style}]{va_s["mae"]:.4f}[/{mae_style}]',
              f'{va_s["rmse"]:.4f}', f'[{r2_style}]{va_s["r2"]:.4f}[/{r2_style}]',
              f'{va_s["mape"]:.2f}', f'{va_s["median_ae"]:.4f}', f'{va_s["max_ae"]:.4f}',
              f'{va_s["pearson_r"]:.4f}', f'{va_s.get("loss", 0):.6f}' if "loss" in va_s else '—',
              '—', '—', f'{va_s["time_s"]}s')
    console.print(t)


def print_leaderboard_table(test_s, best_epoch, best_mae, total_params, cfg):
    if not HAS_RICH:
        log(f'═══ FINAL TEST ═══ MAE={test_s["mae"]:.4f} eV │ RMSE={test_s["rmse"]:.4f} eV │ '
            f'R2={test_s["r2"]:.4f} │ MAPE={test_s["mape"]:.2f}%', 'METRIC')
        return
    t = Table(title=f'LEADERBOARD · GrafoPropagation {cfg.VERSION} · QM9 HOMO-LUMO Gap',
              box=box.DOUBLE_EDGE, show_lines=True, title_style='bold yellow on blue')
    t.add_column('Metric', style='bold', width=20)
    t.add_column('Value', style='bold green', width=20, justify='right')
    t.add_column('Unit', style='dim', width=10)
    t.add_row('Test MAE', f'{test_s["mae"]:.4f}', 'eV')
    t.add_row('Test RMSE', f'{test_s["rmse"]:.4f}', 'eV')
    t.add_row('Test R²', f'{test_s["r2"]:.6f}', '')
    t.add_row('Test MAPE', f'{test_s["mape"]:.2f}', '%')
    t.add_row('Test MedianAE', f'{test_s["median_ae"]:.4f}', 'eV')
    t.add_row('Test MaxAE', f'{test_s["max_ae"]:.4f}', 'eV')
    t.add_row('Test Pearson r', f'{test_s["pearson_r"]:.6f}', '')
    t.add_row('Best Val MAE', f'{best_mae:.4f}', 'eV')
    t.add_row('Best Epoch', f'{best_epoch}', '')
    t.add_row('Parameters', f'{total_params:,}', f'({total_params/1e6:.3f}M)')
    t.add_row('Architecture', f'd={cfg.d_model} L={cfg.n_layers} H={cfg.n_heads} hd={cfg.head_dim}', '')
    t.add_row('Features', f'fixed={cfg.n_fixed_feats} mol={cfg.n_mol_feats} z_emb={cfg.n_z_embed}', '')
    t.add_row('DistBias', f'gauss={cfg.dist_n_rbf_gauss} bessel={cfg.dist_n_rbf_bessel} cut={cfg.dist_cutoff}Å', '')
    t.add_row('Run ID', RUN_ID, '')
    t.add_row('Pred Mean', f'{test_s["pred_mean"]:.4f}', 'eV')
    t.add_row('Pred Std', f'{test_s["pred_std"]:.4f}', 'eV')
    t.add_row('Target Mean', f'{test_s["target_mean"]:.4f}', 'eV')
    t.add_row('Target Std', f'{test_s["target_std"]:.4f}', 'eV')
    console.print(t)


def print_config_table(cfg):
    if not HAS_RICH:
        return
    t = Table(title='Configuration', box=box.SIMPLE, show_lines=False, title_style='bold')
    t.add_column('Parameter', style='bold', width=30)
    t.add_column('Value', style='cyan', width=40)
    items = [
        ('Version', cfg.VERSION),
        ('Device', str(cfg.device)),
        ('AMP dtype', str(cfg.amp_dtype)),
        ('Seed', str(cfg.seed)),
        ('Max atoms', str(cfg.max_atoms)),
        ('Feature dim', f'{cfg.feature_dim} (fixed={cfg.n_fixed_feats} mol={cfg.n_mol_feats} z_emb={cfg.n_z_embed})'),
        ('d_model', str(cfg.d_model)),
        ('n_layers', str(cfg.n_layers)),
        ('n_heads', str(cfg.n_heads)),
        ('head_dim', str(cfg.head_dim)),
        ('d_ff', str(cfg.d_ff)),
        ('Dropout', str(cfg.dropout)),
        ('Stochastic depth', str(cfg.stoch_depth)),
        ('Distance bias', f'gauss={cfg.dist_n_rbf_gauss} bessel={cfg.dist_n_rbf_bessel} cut={cfg.dist_cutoff}Å'),
        ('Epochs', str(cfg.epochs)),
        ('Batch size', str(cfg.batch_size)),
        ('Base LR', str(cfg.base_lr_max)),
        ('EMA decay', str(cfg.ema_decay)),
        ('AWP eps/lr/start', f'{cfg.awp_eps}/{cfg.awp_lr}/ep{cfg.awp_start_ep}'),
        ('Lookahead k/α', f'{cfg.la_k}/{cfg.la_alpha}'),
        ('Huber delta', str(cfg.huber_delta)),
        ('Mixup α/prob', f'{cfg.mixup_alpha}/{cfg.mixup_prob}'),
        ('Normalize y', str(cfg.normalize_y)),
    ]
    for k, v in items:
        t.add_row(k, v)
    console.print(t)


def print_data_stats(data, cfg, train_ds, val_ds, test_ds, y_mean=0.0, y_std=1.0):
    if not HAS_RICH:
        return
    targets_raw = data['targets_raw']
    n_atoms = data['n_atoms']
    t = Table(title='QM9 Dataset Statistics', box=box.SIMPLE_HEAVY, show_lines=True, title_style='bold green')
    t.add_column('Statistic', style='bold', width=30)
    t.add_column('Value', style='green', width=30)
    t.add_row('Total molecules', f'{len(data["z_indices"]):,}')
    t.add_row('Train / Val / Test', f'{len(train_ds):,} / {len(val_ds):,} / {len(test_ds):,}')
    t.add_row('Target (gap) mean', f'{targets_raw.mean():.4f} eV')
    t.add_row('Target (gap) std', f'{targets_raw.std():.4f} eV')
    t.add_row('Target (gap) min', f'{targets_raw.min():.4f} eV')
    t.add_row('Target (gap) max', f'{targets_raw.max():.4f} eV')
    t.add_row('Atoms mean', f'{n_atoms.float().mean():.1f}')
    t.add_row('Atoms min / max', f'{n_atoms.min()} / {n_atoms.max()}')
    t.add_row('z_indices shape', f'{data["z_indices"].shape}')
    t.add_row('positions shape', f'{data["positions"].shape}')
    t.add_row('y_mean (train-only norm)', f'{y_mean:.4f}')
    t.add_row('y_std (train-only norm)', f'{y_std:.4f}')
    console.print(t)


def print_feature_stats_table(feat_mean, feat_std, cfg):
    """Print feature normalization statistics."""
    if not HAS_RICH:
        return
    t = Table(title='Feature Normalization (Train-Only Stats)', box=box.SIMPLE,
              show_lines=False, title_style='bold cyan')
    t.add_column('Feature Group', style='bold', width=25)
    t.add_column('Dims', style='cyan', width=8)
    t.add_column('Mean Range', style='green', width=25)
    t.add_column('Std Range', style='green', width=25)
    # Fixed features (0:27)
    t.add_row('Fixed Physical', '0-26',
              f'[{feat_mean[:27].min():.3f}, {feat_mean[:27].max():.3f}]',
              f'[{feat_std[:27].min():.3f}, {feat_std[:27].max():.3f}]')
    # Mol-dependent features (27:47)
    t.add_row('Mol-Dependent', '27-46',
              f'[{feat_mean[27:47].min():.3f}, {feat_mean[27:47].max():.3f}]',
              f'[{feat_std[27:47].min():.3f}, {feat_std[27:47].max():.3f}]')
    console.print(t)


# ═══════════════════════════ MAIN ════════════════════════════════════════════════
def main():
    cfg = CFG()
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    log_separator(f'GrafoPropagation {cfg.VERSION} · Run {RUN_ID}')
    print_config_table(cfg)

    log(f'device={cfg.device}  amp={cfg.amp_dtype}  max_atoms={cfg.max_atoms}  feature_dim={cfg.feature_dim}')

    # ── Load raw QM9 data (z_indices, positions, fmask, targets) ──
    raw_data = precompute_qm9_data(cfg)

    z_indices_all = raw_data['z_indices']    # (N, 29) Long
    positions_all = raw_data['positions']    # (N, 29, 3) Float
    fmask_all = raw_data['fmask']            # (N, 29)
    targets_raw = raw_data['targets_raw']    # (N,)
    n_atoms_all = raw_data['n_atoms']        # (N,)

    # ── Split 80/10/10 (determinístico) ──
    torch.manual_seed(cfg.seed)
    n_total = len(z_indices_all)
    indices = torch.randperm(n_total)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # ══ NORMALIZAÇÃO: APENAS em dados de treino (sem batota) ══
    # Target normalization
    y_mean = float(targets_raw[train_idx].mean())
    y_std = float(targets_raw[train_idx].std())
    if cfg.normalize_y:
        targets_all = (targets_raw - y_mean) / max(y_std, 1e-8)
    else:
        targets_all = targets_raw.clone()
        y_mean = 0.0
        y_std = 1.0
    cfg._y_std = y_std
    log(f'Target stats (train-only): y_mean={y_mean:.4f} eV, y_std={y_std:.4f} eV', 'DATA')

    # ── Compute molecule-dependent per-atom features ──
    mol_features_all = compute_all_mol_features(
        z_indices_all, positions_all, n_atoms_all, cfg.max_atoms
    )  # (N, 29, 20)

    # ── Feature normalization (train-only) ──
    M = cfg.max_atoms
    train_fmask = fmask_all[train_idx]  # (n_train, 29)
    train_mol_feats = mol_features_all[train_idx]  # (n_train, 29, 20)
    real_mask_train = (train_fmask > -9000.0)  # (n_train, 29) bool

    # Also normalize the fixed features using train-only atom frequency
    # Compute combined features for normalization stats
    train_z = z_indices_all[train_idx]  # (n_train, 29) Long
    train_fixed = torch.from_numpy(FIXED_FEAT_TABLE)[train_z]  # (n_train, 29, 27)

    # Combined: (n_train, 29, 47) = fixed(27) + mol(20)
    train_combined = torch.cat([train_fixed, train_mol_feats], dim=-1)  # (n_train, 29, 47)

    # Compute mean/std per feature dimension across all real atoms in training set
    n_total_feats = cfg.n_fixed_feats + cfg.n_mol_feats  # 47
    feat_mean = torch.zeros(n_total_feats, dtype=torch.float32)
    feat_std = torch.ones(n_total_feats, dtype=torch.float32)

    for j in range(n_total_feats):
        col_vals = train_combined[:, :, j]  # (n_train, 29)
        real_vals = col_vals[real_mask_train]
        if len(real_vals) > 0:
            feat_mean[j] = real_vals.mean()
            feat_std[j] = max(real_vals.std(), 1e-6)

    log(f'Feature stats (train-only): feat_mean range=[{feat_mean.min():.3f}, {feat_mean.max():.3f}]', 'DATA')
    log(f'Feature stats (train-only): feat_std  range=[{feat_std.min():.4f}, {feat_std.max():.4f}]', 'DATA')
    print_feature_stats_table(feat_mean, feat_std, cfg)

    # Apply normalization to ALL data
    all_fixed = torch.from_numpy(FIXED_FEAT_TABLE)[z_indices_all]  # (N, 29, 27)
    all_combined = torch.cat([all_fixed, mol_features_all], dim=-1)  # (N, 29, 47)

    # Normalize
    real_mask_all = (fmask_all > -9000.0)  # (N, 29) bool
    for idx in range(n_total):
        real_atoms = real_mask_all[idx]
        all_combined[idx] = (all_combined[idx] - feat_mean) / feat_std
        all_combined[idx, ~real_atoms] = 0.0

    # The Dataset will receive:
    # - mol_features: already normalized (N, 29, 20)  — we extract just the mol-dependent part
    #   But wait, we normalized ALL 47 features together. The fixed features are also normalized now.
    #   We need to split them back apart.
    #   Actually, since the model looks up fixed features from the buffer and then normalizes them,
    #   we should normalize the fixed features INSIDE the model, not here.
    #   Let me rethink...

    # Actually, the simplest approach: normalize ALL 47 features (fixed + mol-dependent) together,
    # and pass them all as mol_features. Then in the model, DON'T use the fixed_feat_table buffer;
    # just use the already-normalized features directly.
    #
    # But this means we lose the elegant lookup approach. Let me think...
    #
    # Better approach: split the normalized combined back into fixed and mol-dependent parts.
    # - Normalized fixed features: (N, 29, 27) — store these as part of mol_features
    # - Normalized mol-dependent features: (N, 29, 20) — store these as part of mol_features
    # - In the model, concatenate them with z_embed
    #
    # This means the model doesn't need the fixed_feat_table buffer at all.
    # The normalization is done here, and the model just receives the full feature vector.
    #
    # Wait, but then the model needs to know which features are which. Actually, it doesn't!
    # The model just gets a (B, T, 47+16=63) tensor and projects it through nn.Linear(63, 192).
    #
    # So the simplest approach: store the full 47-dim normalized features as mol_features.
    # The model concatenates z_embed to get 63 dims total.
    #
    # Let me do this. It's cleaner.

    # mol_features_all now contains the NORMALIZED combined features (N, 29, 47)
    mol_features_normalized = all_combined  # (N, 29, 47)

    # Update feature_dim to reflect the actual input (47 normalized + 16 z_embed = 63)
    # Already set in CFG: feature_dim = 27 + 20 + 16 = 63
    # But now mol_features has 47 dims instead of 20, so total = 47 + 16 = 63
    # That's the same! Because 27 (fixed) + 20 (mol) = 47, plus 16 (z_embed) = 63.
    # So the input to the model is still 63 dims. 

    train_ds = QM9Dataset(mol_features_normalized, fmask_all, targets_all,
                          z_indices_all, positions_all, train_idx)
    val_ds = QM9Dataset(mol_features_normalized, fmask_all, targets_all,
                         z_indices_all, positions_all, val_idx)
    test_ds = QM9Dataset(mol_features_normalized, fmask_all, targets_all,
                          z_indices_all, positions_all, test_idx)

    print_data_stats(raw_data, cfg, train_ds, val_ds, test_ds, y_mean, y_std)

    tr_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=0, pin_memory=False, drop_last=True)
    va_ld = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
    te_ld = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)

    log(f'DataLoaders: train={len(tr_ld)} batches (bs={cfg.batch_size})  '
        f'val={len(va_ld)} batches  test={len(te_ld)} batches')

    # ── Build Model ──
    log_separator('Building Model')
    model = GrafoPropagationGeoQM9(cfg).to(cfg.device)

    ema = EMA(model, cfg.ema_decay)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'Total parameters: {total_params:,} ({total_params/1e6:.3f} M)')
    log(f'Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.3f} M)')

    # ── Quick forward test ──
    fb_feat, fb_fm, fb_tgt, fb_z, fb_pos = next(iter(tr_ld))
    fb_feat = fb_feat.to(cfg.device)
    fb_fm = fb_fm.to(cfg.device)
    fb_tgt = fb_tgt.to(cfg.device)
    fb_z = fb_z.to(cfg.device)
    fb_pos = fb_pos.to(cfg.device)
    test_pred = model(fb_feat, fb_fm, fb_z, fb_pos)
    log(f'Forward test: feat={fb_feat.shape}, fmask={fb_fm.shape}, z={fb_z.shape}, pos={fb_pos.shape}')
    log(f'  pred shape={test_pred.shape}, range=[{test_pred.min().item():.4f}, {test_pred.max().item():.4f}]')
    log(f'  target range=[{fb_tgt.min().item():.4f}, {fb_tgt.max().item():.4f}]')
    log(f'  Initial MAE (normalized): {(test_pred - fb_tgt).abs().mean().item():.4f}')
    log(f'  Initial MAE (eV): {((test_pred - fb_tgt).abs() * y_std).mean().item():.4f} eV')
    del fb_feat, fb_fm, fb_tgt, fb_z, fb_pos, test_pred

    # ── Optimizer ──
    base_opt = torch.optim.AdamW(model.parameters(), lr=0.0, betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=cfg.wd)
    optimizer = Lookahead(base_opt, cfg.la_k, cfg.la_alpha)
    use_amp = (cfg.device.type == 'cuda')
    scaler = GradScaler('cuda', enabled=False)
    awp = AWP(model, scaler, cfg.awp_eps, cfg.awp_lr)
    gc_h = register_gc(model)

    total_steps = cfg.epochs * len(tr_ld)
    lr_sched = WarmupCosineLR(total_steps, cfg.warmup_frac, cfg.min_lr_frac)
    log(f'LR schedule: total_steps={total_steps}, warmup_steps={int(cfg.warmup_frac * total_steps)}')

    best_mae = float('inf')
    best_epoch = 0
    best_r2 = 0.0
    gstep = 0
    history = []

    log_separator('TRAINING START')

    # ── Training Loop ──
    for epoch in range(1, cfg.epochs + 1):
        tr_s, gstep = train_epoch(model, ema, optimizer, scaler, tr_ld, awp,
                                   cfg, epoch, gstep, lr_sched, base_opt)
        va_s = evaluate(ema.shadow, va_ld, cfg, y_mean, y_std)

        is_best = va_s['mae'] < best_mae
        if is_best:
            best_mae = va_s['mae']
            best_epoch = epoch
            best_r2 = va_s['r2']

        print_epoch_table(epoch, tr_s, va_s, best_mae, cfg)

        log(f'EPOCH {epoch:3d}/{cfg.epochs} │ '
            f'tr_loss={tr_s["loss"]:.6f} │ tr_mae={tr_s["mae_ev"]:.4f} eV │ '
            f'va_mae={va_s["mae"]:.4f} eV │ va_rmse={va_s["rmse"]:.4f} eV │ '
            f'va_R2={va_s["r2"]:.4f} │ va_MAPE={va_s["mape"]:.2f}% │ '
            f'va_Pearson={va_s["pearson_r"]:.4f} │ '
            f'best_mae={best_mae:.4f} eV (ep{best_epoch}) │ '
            f'gnorm={tr_s["grad_norm"]:.3f} │ {tr_s["time_s"]}s', 'METRIC')

        hist_entry = {
            'epoch': epoch,
            'train_loss': tr_s['loss'],
            'train_mae_ev': tr_s['mae_ev'],
            'train_grad_norm': tr_s['grad_norm'],
            'train_lr': tr_s['lr'],
            'train_time_s': tr_s['time_s'],
            'val_loss': va_s['loss'],
            'val_mae': va_s['mae'],
            'val_rmse': va_s['rmse'],
            'val_r2': va_s['r2'],
            'val_mape': va_s['mape'],
            'val_median_ae': va_s['median_ae'],
            'val_max_ae': va_s['max_ae'],
            'val_pearson_r': va_s['pearson_r'],
            'best_mae': best_mae,
            'best_epoch': best_epoch,
        }
        history.append(hist_entry)

        if is_best:
            torch.save({
                'model': model.state_dict(),
                'ema': ema.shadow.state_dict(),
                'epoch': epoch,
                'mae': best_mae,
                'r2': best_r2,
                'history': history,
                'y_mean': y_mean,
                'y_std': y_std,
                'feat_mean': feat_mean,
                'feat_std': feat_std,
                'cfg': {k: v for k, v in vars(cfg).items() if not k.startswith('_')},
            }, os.path.join(cfg.checkpoint_dir, 'best_model.pt'))
            log(f'  ★ New best MAE: {best_mae:.4f} eV  R²={best_r2:.4f}  (epoch {epoch})', 'METRIC')

        if epoch % cfg.checkpoint_every == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'ema': ema.shadow.state_dict(),
                'best_mae': best_mae,
            }, os.path.join(cfg.checkpoint_dir, f'ep{epoch:03d}.pt'))

        try:
            with open(cfg.history_path, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception:
            pass

    # ── Final Test Evaluation ──
    log_separator('FINAL TEST EVALUATION')

    ckpt = torch.load(os.path.join(cfg.checkpoint_dir, 'best_model.pt'),
                       map_location=cfg.device, weights_only=False)
    ema.shadow.load_state_dict(ckpt['ema'])
    te_s = evaluate(ema.shadow, te_ld, cfg, ckpt['y_mean'], ckpt['y_std'])

    print_leaderboard_table(te_s, best_epoch, best_mae, total_params, cfg)

    log_separator('RESULTS')
    log(f'Best Val MAE: {best_mae:.4f} eV (epoch {best_epoch})', 'METRIC')
    log(f'Test MAE:     {te_s["mae"]:.4f} eV', 'METRIC')
    log(f'Test RMSE:    {te_s["rmse"]:.4f} eV', 'METRIC')
    log(f'Test R²:      {te_s["r2"]:.6f}', 'METRIC')
    log(f'Test MAPE:    {te_s["mape"]:.2f}%', 'METRIC')
    log(f'Test Pearson: {te_s["pearson_r"]:.6f}', 'METRIC')
    log(f'Test MedianAE:{te_s["median_ae"]:.4f} eV', 'METRIC')
    log(f'Test MaxAE:   {te_s["max_ae"]:.4f} eV', 'METRIC')
    log(f'Parameters:   {total_params:,} ({total_params/1e6:.3f} M)', 'METRIC')

    for h in gc_h:
        h.remove()

    log(f'DONE · GrafoPropagation {cfg.VERSION} · Run {RUN_ID}', 'METRIC')


if __name__ == '__main__':
    main()