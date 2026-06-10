#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GrafoPropagation v29-DEEPMIND-BYOL-ASYMMETRIC · HOMO-LUMO Gap Regression
=====================================================================

Sem E(3), sem S(3), sem rotação das moléculas.

Novidades principais vs v28:
  1) Pre-pretreino substituído por DeepMind BYOL (Bootstrap Your Own Latent):
     - Fluxo de gradiente assimétrico (Student prevê Teacher, Teacher não prevê Student).
     - Resolve o colapso de dimensionalidade e a "preguiça" do projetor VICReg.
     - Teacher atualizado por Exponential Moving Average (EMA).

  2) Spherical Shock Terrain Assimétrico ("O que eu sinto todos sentem"):
     - "Podes encaixar mas não ser encaixado": Competição Softmax dos átomos pelo terreno.
     - Global Sphere State: O terreno soma o contacto da molécula e partilha a forma
       global de volta com todos os pontos antes da projeção.

  3) CÁLCULO ALIENÍGENA (RESSURGÊNCIA DE ÉCALLE):
     - Fim do Gradient Clipping artificial que destrói a geometria das singularidades.
     - Implementação da Transformada de Borel-Laplace e derivação não-perturbativa
       para absorver e navegar gradientes infinitos (BorelLaplaceResurgentOptimizer).

  4) CORREÇÃO CRÍTICA DE LEAKAGE:
     - O pré-treino não supervisionado (BYOL) usa ESTRITAMENTE o train set.
     - Zero exposição à distribuição do test/val set para garantir integridade.
"""

import os, sys, math, time, json, copy, bz2, tarfile, random, datetime, warnings, urllib.request
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    console = Console(width=180, force_terminal=True)
    HAS_RICH = True
except Exception:
    HAS_RICH = False
    class _Console:
        def print(self, *a, **kw): print(*a)
        def rule(self, *a, **kw): print("─" * 120)
    console = _Console()


# ═══════════════════════════════════ CONFIG ═══════════════════════════════════

class CFG:
    VERSION = "v29-DEEPMIND-BYOL-ASYMMETRIC"

    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    amp_enabled = bool(torch.cuda.is_available() and amp_dtype != torch.float32)

    qm9_root = os.environ.get("QM9_ROOT", "/tmp/QM9")
    cache_path = os.environ.get("QM9_CACHE", "/tmp/qm9_rad_cache_v5_terrain.pt")

    max_atoms = 29
    gap_idx = 4
    normalize_y = True

    # Features: fixed physical 27 + molecule-dependent 20 + Z embedding 16.
    n_fixed_feats = 27
    n_mol_feats = 20
    atom_feature_dim = n_fixed_feats + n_mol_feats  # 47
    n_z_embed = 16
    feature_dim = atom_feature_dim + n_z_embed      # 63

    # Standard QM9 split used by many papers when N permits.
    split_mode = "standard_110k"
    standard_train = 110000
    standard_val = 10000

    # Shock Terrain Sphere (Asymmetric & Global).
    use_shock_terrain = True
    terrain_dirs = 16
    terrain_shells = 4
    terrain_radius = 4.8
    terrain_sigma = 0.55
    terrain_rotate_train = True
    terrain_eval_views = 4
    terrain_dropout = 0.03

    # Distance attention bias.
    use_distance_bias = True
    dist_n_rbf_gauss = 16
    dist_n_rbf_bessel = 8
    dist_cutoff = 5.0

    # Architecture.
    d_model = 192
    n_layers = 4
    n_heads = 8
    head_dim = 24
    d_ff = 896
    dropout = 0.10
    stoch_depth = 0.08
    conv_kernel = 3

    # RoPE/order mechanisms. Cyclic mask is false by default for molecules.
    use_accelerated_rope = True
    rope_base = 10000.0
    rope_max_seq = 64

    use_cyclic_mask = False
    use_xor_bias = True
    use_topological_bias = True
    use_conditional_gate = True

    # ── PRE-PRETREINO DEEPMIND BYOL ───────────────────────────────────────────
    use_prepretrain = bool(int(os.environ.get("USE_PREPRETRAIN", "1")))
    prepretrain_epochs = int(os.environ.get("PREPRETRAIN_EPOCHS", "12"))
    prepretrain_batch_size = 128
    prepretrain_lr = 5e-4
    prepretrain_wd = 1.5e-4
    prepretrain_warmup_frac = 0.05
    prepretrain_min_lr_frac = 0.05
    # Clipping limits removidos em favor do BorelLaplaceResurgentOptimizer
    prepretrain_use_all_data = False # NUNCA usar test data, mesmo em unsupervised
    prepretrain_projector_hidden = 256
    prepretrain_projector_out = 128
    prepretrain_byol_momentum = 0.99
    
    # Curriculum da rotação do terreno.
    prepretrain_curriculum = True
    prepretrain_rot_start = 0.15
    prepretrain_rot_end = 1.0

    # Pretraining principal.
    pretrain_epochs = int(os.environ.get("PRETRAIN_EPOCHS", "8"))
    pretrain_batch_size = 128
    pretrain_lr = 4e-4
    pretrain_wd = 1e-4
    pretrain_mask_rate = 0.15
    pretrain_aux_weight = 1.00
    pretrain_atom_weight = 0.25
    pretrain_vicreg_weight = 0.0 

    # Finetuning.
    epochs = int(os.environ.get("EPOCHS", "60"))
    batch_size = 128
    grad_accum = 1
    wd = 1e-4
    base_lr_max = 1e-3                # <── aumentado de 5e-4 para 1e-3
    warmup_frac = 0.05
    min_lr_frac = 0.05
    ema_decay = 0.999
    awp_eps = 0.003
    awp_lr = 0.005
    awp_start_ep = 15
    la_k = 6
    la_alpha = 0.50
    huber_delta = 1.0

    # Desligado: mixup entre moléculas é geometria falsa.
    mixup_prob = 0.0
    feature_noise_std = 0.002

    # Loader/logging.
    num_workers = int(os.environ.get("NUM_WORKERS", "2"))
    pin_memory = bool(torch.cuda.is_available())
    attn_log_freq = 30
    checkpoint_every = 25
    checkpoint_dir = "./ckpt_qm9_v29_terrain"
    history_path = "./ckpt_qm9_v29_terrain/history.json"


# ═════════════════════════════════ UTIL ═══════════════════════════════════════

RUN_ID = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def _ts():
    return datetime.datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]

def log(msg, level="INFO"):
    style = {
        "INFO": "dim",
        "WARN": "yellow",
        "ERROR": "bold red",
        "DATA": "bold green",
        "TRAIN": "magenta",
        "PRE": "bold magenta",
        "PREPRE": "bold yellow",
        "METRIC": "bold cyan",
        "EVAL": "bold blue",
    }.get(level, "")
    if HAS_RICH and style:
        console.print(f"[{style}][[{_ts()}]] [{level}] {msg}[/{style}]")
    else:
        print(f"[[{_ts()}]] [{level}] {msg}", flush=True)

def log_separator(title=""):
    if HAS_RICH:
        console.rule(f"[bold]{title}[/bold]", style="bright_blue", characters="═")
    else:
        print("\n" + "═" * 120)
        print(title)
        print("═" * 120)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cfg_to_dict(cfg):
    out = {}
    for k in dir(cfg):
        if k.startswith("_"):
            continue
        v = getattr(cfg, k)
        if callable(v):
            continue
        if isinstance(v, torch.device):
            out[k] = str(v)
        elif isinstance(v, torch.dtype):
            out[k] = str(v)
        elif isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
    return out

def ensure_paths(cfg):
    for attr, fallback in [
        ("qm9_root", "./QM9"),
        ("cache_path", "./qm9_rad_cache_v5_terrain.pt"),
    ]:
        p = getattr(cfg, attr)
        d = os.path.dirname(p) if attr == "cache_path" else p
        try:
            os.makedirs(d or ".", exist_ok=True)
        except Exception:
            setattr(cfg, attr, fallback)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

if hasattr(torch, "set_float32_matmul_precision"):
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ═════════════════════════ ATOMIC PHYSICAL FEATURES ══════════════════════════

ATOMIC_PROPS = {
    1: ("H", 2.20, 2.300, 2.20, 13.59844, 0.75420, 31, 110, 0.667,
        1, 1, 0, 1, 0, 1, 1, 1, 1.0, 0, 0.5, 0.0337, 0.0337),
    6: ("C", 2.55, 2.544, 2.50, 11.26030, 1.26212, 76, 170, 1.760,
        4, 2, 2, 2, 0, 2, 14, 4, 3.25, 2, 50.2, 0.0273, 0.0109),
    7: ("N", 3.04, 3.066, 3.07, 14.53414, -0.07, 71, 155, 1.100,
        5, 2, 3, 3, 1, 2, 15, 3, 3.90, 2, 79.3, 0.0278, 0.0139),
    8: ("O", 3.44, 3.610, 3.50, 13.61806, 1.46111, 66, 152, 0.802,
        6, 2, 4, 2, 2, 2, 16, 2, 4.55, 2, 115.3, 0.0278, 0.0185),
    9: ("F", 3.98, 4.193, 4.10, 17.42282, 3.40119, 57, 147, 0.557,
        7, 2, 5, 1, 3, 2, 17, 1, 5.20, 2, 162.5, 0.0280, 0.0271),
}

def _build_fixed_feature_vector(z):
    if z not in ATOMIC_PROPS:
        z = 6
    p = ATOMIC_PROPS[z]
    one_hot = [1.0 if z == zn else 0.0 for zn in [1, 6, 7, 8, 9]]
    return np.array(one_hot + [
        float(z),
        p[1], p[2], p[3],
        p[4], p[5],
        p[6] / 100.0,
        p[7] / 100.0,
        p[8],
        float(p[9]), float(p[10]), float(p[11]), float(p[12]), float(p[13]),
        float(p[14]), float(p[15]), float(p[16]),
        p[17], float(p[18]), p[19], p[20], p[21],
    ], dtype=np.float32)

FIXED_FEAT_TABLE = np.zeros((10, CFG.n_fixed_feats), dtype=np.float32)
for _z in [1, 6, 7, 8, 9]:
    FIXED_FEAT_TABLE[_z] = _build_fixed_feature_vector(_z)


def compute_mol_dependent_features(z_arr, pos, max_atoms):
    M = max_atoms
    n = min(len(z_arr), M)
    feats = np.zeros((M, 20), dtype=np.float64)
    if n == 0:
        return feats

    z = z_arr[:n].astype(np.float64)
    p = pos[:n].astype(np.float64)

    centroid = p.mean(axis=0)
    pc = p - centroid
    d_cent = np.sqrt((pc ** 2).sum(axis=1))
    mol_radius = max(float(d_cent.max()), 1e-6)
    mol_size = n / M

    diff = p[:, None, :] - p[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))

    dist_safe = dist.copy()
    np.fill_diagonal(dist_safe, 1.0)
    coul = (z[:, None] * z[None, :]) / dist_safe
    np.fill_diagonal(coul, 0.0)

    for i in range(n):
        d_i = dist[i].copy()
        d_i[i] = np.inf
        sorted_d = np.sort(d_i[d_i < np.inf])
        d_others = d_i[d_i < np.inf]
        n_others = len(sorted_d)

        mean_dist = sorted_d.mean() if n_others else 0.0
        std_dist = sorted_d.std() if n_others else 0.0
        min_dist = sorted_d[0] if n_others else 0.0
        nn2_dist = sorted_d[1] if n_others > 1 else min_dist
        nn3_dist = sorted_d[2] if n_others > 2 else nn2_dist
        max_dist = sorted_d[-1] if n_others else 0.0

        sigma = 0.3
        nb_1p8 = np.exp(-0.5 * ((d_others - 1.8) / sigma) ** 2).sum()
        nb_2p5 = np.exp(-0.5 * ((d_others - 2.5) / sigma) ** 2).sum()
        nb_3p5 = np.exp(-0.5 * ((d_others - 3.5) / sigma) ** 2).sum()

        c_i = coul[i]
        sum_coulomb = c_i.sum()
        z_others = np.delete(z, i)
        local_elec_dens = (z_others * np.exp(-d_others / 2.0)).sum()

        x_c, y_c, z_c = pc[i]
        xn, yn, zn = x_c / mol_radius, y_c / mol_radius, z_c / mol_radius

        feats[i] = [
            x_c, y_c, z_c,
            xn, yn, zn,
            mean_dist, std_dist,
            min_dist, nn2_dist, nn3_dist, max_dist,
            nb_1p8, nb_2p5, nb_3p5,
            sum_coulomb,
            local_elec_dens,
            d_cent[i],
            mol_size,
            mol_radius,
        ]
    return feats


# ═════════════════════════ DATA LOADING / CACHE ══════════════════════════════

HAR2EV = 27.211386245988

def _convert_compact12_to_pyg_units(y):
    y = np.array(y, dtype=np.float32)
    for idx in [2, 3, 4, 6, 7, 8, 9, 10]:
        y[idx] *= HAR2EV
    return y

def _parse_xyz_text_block(text, append_fn):
    lines = text.splitlines()
    ptr = 0
    while ptr < len(lines):
        line = lines[ptr].strip()
        try:
            n_atoms = int(line.split()[0])
            if n_atoms < 1 or n_atoms > 100:
                ptr += 1
                continue
        except Exception:
            ptr += 1
            continue

        if ptr + 1 + n_atoms >= len(lines):
            break

        props = lines[ptr + 1].strip().split()
        if len(props) < 17:
            ptr += 1
            continue

        try:
            y = _convert_compact12_to_pyg_units([float(props[i]) for i in range(5, 17)])
            z_list, pos_list = [], []
            atomic_num = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
            valid = True
            for i in range(n_atoms):
                parts = lines[ptr + 2 + i].strip().split()
                if len(parts) < 4 or parts[0] not in atomic_num:
                    valid = False
                    break
                z_list.append(atomic_num[parts[0]])
                pos_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if valid:
                append_fn(z_list, pos_list, y)
        except Exception:
            pass

        ptr += 2 + n_atoms

def precompute_qm9_data(cfg):
    cache_path = cfg.cache_path
    if os.path.exists(cache_path):
        try:
            data = torch.load(cache_path, map_location="cpu", weights_only=False)
            required = {"z_indices", "positions", "fmask", "targets_raw", "targets_all", "n_atoms"}
            if required.issubset(set(data.keys())):
                log(f"Loading cache v5: {cache_path} · N={len(data['z_indices']):,}", "DATA")
                return data
            log("Cache antigo/incompleto; recalculando.", "WARN")
            os.remove(cache_path)
        except Exception:
            log("Cache corrupto; recalculando.", "WARN")
            try: os.remove(cache_path)
            except Exception: pass

    M = cfg.max_atoms
    all_z_indices = None
    all_positions = None
    all_targets_all = None
    all_n_atoms = None

    try:
        from torch_geometric.datasets import QM9
        log("A carregar QM9 via torch_geometric.", "DATA")
        ds = QM9(root=cfg.qm9_root)
        N = len(ds)
        K = int(ds[0].y.view(-1).numel())
        all_z_indices = np.zeros((N, M), dtype=np.int64)
        all_positions = np.zeros((N, M, 3), dtype=np.float32)
        all_targets_all = np.full((N, K), np.nan, dtype=np.float32)
        all_n_atoms = np.zeros(N, dtype=np.int32)

        t0 = time.time()
        for idx in range(N):
            d = ds[idx]
            z = d.z.cpu().numpy().astype(np.int64)
            pos = d.pos.cpu().numpy().astype(np.float32)
            y = d.y.view(-1).cpu().numpy().astype(np.float32)
            n = min(len(z), M)
            all_z_indices[idx, :n] = z[:n]
            all_positions[idx, :n] = pos[:n]
            all_targets_all[idx, :len(y)] = y
            all_n_atoms[idx] = n
            if (idx + 1) % 20000 == 0 or idx + 1 == N:
                rate = (idx + 1) / max(time.time() - t0, 1e-9)
                log(f"PyG parse: {idx+1:,}/{N:,} · {rate:.0f} mol/s", "DATA")
    except Exception as e:
        log(f"PyG falhou: {repr(e)}", "WARN")

    if all_z_indices is None:
        log("Fallback direto: figshare dsgdb9nsd.xyz.tar.bz2", "DATA")
        raw_dir = os.path.join(cfg.qm9_root, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        tar_path = os.path.join(raw_dir, "dsgdb9nsd.xyz.tar.bz2")
        xyz_path = os.path.join(raw_dir, "dsgdb9nsd.xyz")

        if not os.path.exists(tar_path) and not os.path.exists(xyz_path):
            url = "https://ndownloader.figshare.com/files/3195389"
            log(f"Downloading {url}", "DATA")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r:
                with open(tar_path, "wb") as f:
                    f.write(r.read())

        z_list_all, pos_list_all, y_list_all, n_list_all = [], [], [], []
        def append_fn(zl, pl, y):
            z_list_all.append(zl)
            pos_list_all.append(pl)
            y_list_all.append(y)
            n_list_all.append(len(zl))
            if len(z_list_all) % 20000 == 0:
                log(f"Direct parse: {len(z_list_all):,} molecules", "DATA")

        if os.path.exists(xyz_path):
            with open(xyz_path, "r", encoding="ascii", errors="ignore") as f:
                _parse_xyz_text_block(f.read(), append_fn)
        elif os.path.exists(tar_path):
            with tarfile.open(tar_path, "r:bz2") as tar:
                for member in tar:
                    if not member.isfile() or not member.name.endswith(".xyz"):
                        continue
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    txt = f.read().decode("ascii", errors="ignore")
                    _parse_xyz_text_block(txt, append_fn)

        if len(z_list_all) == 0:
            raise RuntimeError("Não foi possível carregar QM9.")

        N = len(z_list_all)
        K = 12
        all_z_indices = np.zeros((N, M), dtype=np.int64)
        all_positions = np.zeros((N, M, 3), dtype=np.float32)
        all_targets_all = np.full((N, K), np.nan, dtype=np.float32)
        all_n_atoms = np.zeros(N, dtype=np.int32)

        for idx, (zl, pl, y) in enumerate(zip(z_list_all, pos_list_all, y_list_all)):
            n = min(len(zl), M)
            all_z_indices[idx, :n] = np.array(zl[:n], dtype=np.int64)
            all_positions[idx, :n] = np.array(pl[:n], dtype=np.float32)
            all_targets_all[idx, :len(y)] = y
            all_n_atoms[idx] = n

    N = len(all_z_indices)
    for idx in range(N):
        n = int(all_n_atoms[idx])
        if n > 0:
            c = all_positions[idx, :n].mean(axis=0)
            all_positions[idx, :n] -= c

    fmask = np.zeros((N, M), dtype=np.float32)
    for idx in range(N):
        n = int(all_n_atoms[idx])
        if n < M:
            fmask[idx, n:] = -10000.0

    targets_raw = all_targets_all[:, cfg.gap_idx].astype(np.float32)

    result = {
        "z_indices": torch.from_numpy(all_z_indices),
        "positions": torch.from_numpy(all_positions),
        "fmask": torch.from_numpy(fmask),
        "targets_raw": torch.from_numpy(targets_raw),
        "targets_all": torch.from_numpy(all_targets_all),
        "n_atoms": torch.from_numpy(all_n_atoms),
    }
    try: torch.save(result, cache_path)
    except Exception: pass
    return result


def compute_all_mol_features(z_indices, positions, n_atoms, max_atoms, batch_log=5000):
    N = z_indices.shape[0]
    out = np.zeros((N, max_atoms, CFG.n_mol_feats), dtype=np.float32)
    z_np = z_indices.numpy()
    p_np = positions.numpy()
    n_np = n_atoms.numpy()

    t0 = time.time()
    for i in range(N):
        n = int(n_np[i])
        if n > 0:
            out[i] = compute_mol_dependent_features(
                z_np[i, :n].astype(np.float64), p_np[i, :n].astype(np.float64), max_atoms
            ).astype(np.float32)
    return torch.from_numpy(out)


def build_atom_features_train_only(raw_data, train_idx, cfg):
    z_all = raw_data["z_indices"]
    fm_all = raw_data["fmask"]
    pos_all = raw_data["positions"]
    n_atoms = raw_data["n_atoms"]
    N, M = z_all.shape

    mol_feats = compute_all_mol_features(z_all, pos_all, n_atoms, M)
    fixed_table = torch.from_numpy(FIXED_FEAT_TABLE).float()

    feat_sum = torch.zeros(cfg.atom_feature_dim)
    feat_sumsq = torch.zeros(cfg.atom_feature_dim)
    count = 0

    chunk = 4096
    train_idx_cpu = train_idx.cpu()
    for s in range(0, len(train_idx_cpu), chunk):
        ids = train_idx_cpu[s:s+chunk]
        fixed = fixed_table[z_all[ids].clamp(0, 9)]
        comb = torch.cat([fixed, mol_feats[ids]], dim=-1)
        mask = fm_all[ids] > -9000.0
        vals = comb[mask]
        if vals.numel() > 0:
            feat_sum += vals.sum(dim=0)
            feat_sumsq += (vals ** 2).sum(dim=0)
            count += vals.shape[0]

    mean = feat_sum / max(count, 1)
    var = (feat_sumsq / max(count, 1) - mean ** 2).clamp_min(1e-12)
    std = var.sqrt().clamp_min(1e-6)

    atom_features = torch.empty((N, M, cfg.atom_feature_dim), dtype=torch.float32)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        fixed = fixed_table[z_all[s:e].clamp(0, 9)]
        comb = torch.cat([fixed, mol_feats[s:e]], dim=-1)
        comb = (comb - mean.view(1, 1, -1)) / std.view(1, 1, -1)
        mask = fm_all[s:e] > -9000.0
        comb[~mask] = 0.0
        atom_features[s:e] = comb

    del mol_feats
    return atom_features, mean, std


def build_aux_targets_train_only(targets_all, train_idx, gap_idx):
    if targets_all.ndim != 2 or targets_all.shape[1] <= 1:
        return torch.empty((len(targets_all), 0)), [], torch.empty(0), torch.empty(0)

    K = targets_all.shape[1]
    cols = []
    for k in range(K):
        if k == gap_idx: continue
        if int(torch.isfinite(targets_all[train_idx, k]).sum()) > 100: cols.append(k)

    if not cols: return torch.empty((len(targets_all), 0)), [], torch.empty(0), torch.empty(0)

    aux_raw = targets_all[:, cols].float()
    aux_norm = torch.full_like(aux_raw, float("nan"))
    means, stds = [], []

    for j in range(len(cols)):
        tr = aux_raw[train_idx, j]
        finite = torch.isfinite(tr)
        m = tr[finite].mean()
        sd = tr[finite].std().clamp_min(1e-6)
        means.append(m)
        stds.append(sd)
        aux_norm[:, j] = (aux_raw[:, j] - m) / sd

    return aux_norm, cols, torch.stack(means), torch.stack(stds)


def make_splits(N, cfg):
    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(N, generator=g)
    if cfg.split_mode == "standard_110k" and N >= cfg.standard_train + cfg.standard_val + 1:
        n_train, n_val = cfg.standard_train, cfg.standard_val
    else:
        n_train, n_val = int(0.8 * N), int(0.1 * N)
    return perm[:n_train], perm[n_train:n_train+n_val], perm[n_train+n_val:]


# ═════════════════════════ MODEL BLOCKS ══════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))
    def forward(self, x):
        xf = x.float()
        rms = xf.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (xf * rms * self.scale.float()).to(x.dtype)


class AcceleratedRoPERotator(nn.Module):
    def __init__(self, head_dim, n_heads, max_len=512, base=10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        half = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
        self.register_buffer("inv_freq", inv_freq)
        self.alpha = nn.Parameter(torch.zeros(n_heads, half))
        
    @staticmethod
    def _rot(x):
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def forward(self, mu):
        B, T, H, D = mu.shape
        t = torch.arange(T, dtype=torch.float32, device=mu.device)
        theta = t[:, None] * self.inv_freq[None, :] + (t * (t - 1) / 2.0)[None, :, None] * self.alpha[:, None, :]
        emb = torch.cat([theta, theta], dim=-1)
        c, s = emb.cos().unsqueeze(0).permute(0, 2, 1, 3).to(mu.dtype), emb.sin().unsqueeze(0).permute(0, 2, 1, 3).to(mu.dtype)
        return F.normalize(mu * c + self._rot(mu) * s, p=2, dim=-1, eps=1e-8)


class IdentityRotator(nn.Module):
    def forward(self, mu): return mu


class XorAttentionBias(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.proj_q = nn.Linear(head_dim, head_dim, bias=False)
        self.proj_k = nn.Linear(head_dim, head_dim, bias=False)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        self.hd = head_dim
        nn.init.xavier_uniform_(self.proj_q.weight, gain=0.5)
        nn.init.xavier_uniform_(self.proj_k.weight, gain=0.5)

    def forward(self, mu_q, mu_k):
        q_bin, k_bin = torch.sigmoid(self.proj_q(mu_q)), torch.sigmoid(self.proj_k(mu_k))
        q_sum = q_bin.sum(dim=-1, keepdim=True).permute(0, 2, 1, 3)
        k_sum = k_bin.sum(dim=-1, keepdim=True).permute(0, 2, 3, 1)
        qk = torch.matmul(q_bin.permute(0, 2, 1, 3), k_bin.permute(0, 2, 1, 3).transpose(-2, -1))
        return ((self.hd - (q_sum + k_sum - 2.0 * qk)) / self.hd) * self.scale


class TopologicalMERAScore(nn.Module):
    def __init__(self, head_dim, heads):
        super().__init__()
        self.isometry = nn.Conv1d(head_dim, head_dim, 3, padding=1, groups=head_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.02))
        nn.init.xavier_uniform_(self.isometry.weight, gain=0.5)

    def forward(self, mu_q):
        B, T, H, D = mu_q.shape
        x = mu_q.permute(0, 2, 3, 1).reshape(B * H, D, T)
        div = 1.0 - F.cosine_similarity(x, self.isometry(x), dim=1)
        div = div.view(B, H, T)
        return -torch.abs(div.unsqueeze(-1) - div.unsqueeze(-2)) * self.scale.abs()


class DistanceAttentionBias(nn.Module):
    def __init__(self, n_rbf_gauss=16, n_rbf_bessel=8, cutoff=5.0, n_heads=8):
        super().__init__()
        self.cutoff = float(cutoff)
        self.register_buffer("gauss_centers", torch.linspace(0, cutoff, n_rbf_gauss))
        self.register_buffer("gauss_gamma", torch.tensor(10.0))
        self.register_buffer("bessel_n", torch.arange(1, n_rbf_bessel + 1, dtype=torch.float32))
        self.proj = nn.Linear(n_rbf_gauss + n_rbf_bessel, n_heads, bias=True)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.05)
        nn.init.zeros_(self.proj.bias)

    def forward(self, distances, fmask=None):
        d = distances.unsqueeze(-1)
        gauss = torch.exp(-self.gauss_gamma * (d - self.gauss_centers) ** 2)
        d_safe = d.clamp(min=1e-6)
        bessel = math.sqrt(2.0 / self.cutoff) * torch.sin(self.bessel_n * math.pi * d_safe / self.cutoff) / d_safe
        rbf = torch.cat([gauss, bessel], dim=-1)
        
        x = (distances / self.cutoff).clamp(max=1.0)
        p = 5
        env = 1.0 - ((p + 1) * (p + 2) / 2.0) * x**p + p * (p + 2) * x**(p + 1) - (p * (p + 1) / 2.0) * x**(p + 2)
        
        bias = self.proj(torch.nan_to_num(rbf * env.unsqueeze(-1).clamp(min=0.0), nan=0.0)).permute(0, 3, 1, 2)
        if fmask is not None:
            pad = fmask < -9000.0
            bias = bias.masked_fill(pad.unsqueeze(1).unsqueeze(3), 0.0).masked_fill(pad.unsqueeze(1).unsqueeze(2), 0.0)
        return bias


def fibonacci_sphere(n):
    pts = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (i / max(n - 1, 1)) * 2.0
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        pts.append([math.cos(theta) * r, y, math.sin(theta) * r])
    return torch.tensor(pts, dtype=torch.float32)

def random_rotation_matrix(batch, device, dtype, max_angle=None):
    if max_angle is None or max_angle >= math.pi - 1e-6:
        u1, u2, u3 = torch.rand(batch, device=device, dtype=dtype), torch.rand(batch, device=device, dtype=dtype), torch.rand(batch, device=device, dtype=dtype)
        qx = torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2)
        qy = torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2)
        qz = torch.sqrt(u1) * torch.sin(2 * math.pi * u3)
        qw = torch.sqrt(u1) * torch.cos(2 * math.pi * u3)
    else:
        axis = F.normalize(torch.randn(batch, 3, device=device, dtype=dtype), dim=-1)
        angle = torch.rand(batch, device=device, dtype=dtype) * max_angle
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        qx_, qy_, qz_ = axis[:, 0] * sin_a, axis[:, 1] * sin_a, axis[:, 2] * sin_a
        norm = torch.sqrt(cos_a**2 + qx_**2 + qy_**2 + qz_**2).clamp(min=1e-8)
        qw, qx, qy, qz = cos_a/norm, qx_/norm, qy_/norm, qz_/norm

    R = torch.empty(batch, 3, 3, device=device, dtype=dtype)
    R[:, 0, 0] = 1 - 2 * (qy*qy + qz*qz)
    R[:, 0, 1] = 2 * (qx*qy - qz*qw)
    R[:, 0, 2] = 2 * (qx*qz + qy*qw)
    R[:, 1, 0] = 2 * (qx*qy + qz*qw)
    R[:, 1, 1] = 1 - 2 * (qx*qx + qz*qz)
    R[:, 1, 2] = 2 * (qy*qz - qx*qw)
    R[:, 2, 0] = 2 * (qx*qz - qy*qw)
    R[:, 2, 1] = 2 * (qy*qz + qx*qw)
    R[:, 2, 2] = 1 - 2 * (qx*qx + qy*qy)
    return R

def fixed_eval_rotations(n):
    rng = np.random.default_rng(12345)
    rots = [np.eye(3, dtype=np.float32)]
    for _ in range(1, max(n, 1)):
        u1, u2, u3 = rng.random(3)
        qx, qy = math.sqrt(1-u1) * math.sin(2*math.pi*u2), math.sqrt(1-u1) * math.cos(2*math.pi*u2)
        qz, qw = math.sqrt(u1) * math.sin(2*math.pi*u3), math.sqrt(u1) * math.cos(2*math.pi*u3)
        rots.append(np.array([
            [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]
        ], dtype=np.float32))
    return torch.from_numpy(np.stack(rots, axis=0))


class SphericalShockTerrain(nn.Module):
    """
    O Terreno Mente-Coletiva e Assimétrico.
    "O que eu sinto todos sentem, podes encaixar mas não ser encaixado."
    """
    def __init__(self, cfg):
        super().__init__()
        dirs = fibonacci_sphere(cfg.terrain_dirs)
        radii = torch.linspace(cfg.terrain_radius / cfg.terrain_shells, cfg.terrain_radius, cfg.terrain_shells, dtype=torch.float32)
        anchors = (dirs[:, None, :] * radii[None, :, None]).reshape(-1, 3).contiguous()
        self.register_buffer("anchors", anchors)
        self.n_points = anchors.shape[0]
        self.rotate_train = cfg.terrain_rotate_train
        self.dropout = cfg.terrain_dropout
        self.register_buffer("eval_rots", fixed_eval_rotations(max(cfg.terrain_eval_views, 1)))
        self.log_sigma = nn.Parameter(torch.full((self.n_points,), math.log(cfg.terrain_sigma)))
        self.gain = nn.Parameter(torch.ones(self.n_points))

    def forward(self, positions, fmask=None, view=None, rot_strength=None):
        B, T, _ = positions.shape
        dtype = positions.dtype
        base = self.anchors.to(device=positions.device, dtype=dtype)

        if (self.training and self.rotate_train) or view == "random":
            R = random_rotation_matrix(B, positions.device, dtype, max_angle=None if rot_strength is None else float(rot_strength) * math.pi)
            anchors = torch.einsum("bij,pj->bpi", R, base)
        elif isinstance(view, int):
            R = self.eval_rots[view % self.eval_rots.shape[0]].to(device=positions.device, dtype=dtype)
            anchors = torch.einsum("ij,pj->pi", R, base).unsqueeze(0).expand(B, -1, -1)
        else:
            anchors = base.unsqueeze(0).expand(B, -1, -1)

        diff = positions.unsqueeze(2) - anchors.unsqueeze(1)
        d2 = diff.pow(2).sum(dim=-1)
        sigma = self.log_sigma.exp().clamp(0.05, 3.0).to(dtype).view(1, 1, -1)
        
        # Ativação raw do choque
        shock_raw = torch.exp(-0.5 * d2 / (sigma * sigma)) * self.gain.to(dtype).view(1, 1, -1)
        
        if fmask is not None:
            valid = (fmask > -9000.0).to(dtype).unsqueeze(-1)
            shock_raw = shock_raw * valid

        # 1. "Podes encaixar mas não ser encaixado": Competição topológica (Softmax).
        # Os átomos (dim=1) competem para se ligar aos pontos do terreno (dim=2).
        fit_prob = F.softmax(shock_raw / 0.1, dim=-1)

        # 2. "O que eu sinto todos sentem": O estado global da esfera.
        # A esfera perceciona o impacto agregado da molécula.
        global_feel = fit_prob.sum(dim=1, keepdim=True)
        global_feel = F.normalize(global_feel, p=2, dim=-1)

        # A interação final é a combinação do encaixe competitivo com o sentimento coletivo
        shock = (shock_raw * fit_prob) + (global_feel * 0.1)

        if self.training and self.dropout > 0:
            shock = F.dropout(shock, p=self.dropout, training=True)
            
        return shock


class GeometricAttention(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, dropout, cfg):
        super().__init__()
        self.n_heads, self.hd, self.dp, self._sc = n_heads, head_dim, dropout, 1.0 / math.sqrt(head_dim)
        self.rope = AcceleratedRoPERotator(head_dim, n_heads, cfg.rope_max_seq, cfg.rope_base) if cfg.use_accelerated_rope else IdentityRotator()
        self.W_mu = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.W_kappa = nn.Linear(d_model, n_heads, bias=True)
        self.Wv = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.Wo = nn.Linear(n_heads * head_dim, d_model, bias=False)

        self.W_gate = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.tau = nn.Parameter(torch.ones(n_heads) * 2.0)
        self.bias_q = nn.Parameter(torch.zeros(n_heads))
        self.xor_bias = XorAttentionBias(head_dim) if cfg.use_xor_bias else None
        self.topo_bias = TopologicalMERAScore(head_dim, n_heads) if cfg.use_topological_bias else None
        self.dist_bias = DistanceAttentionBias(cfg.dist_n_rbf_gauss, cfg.dist_n_rbf_bessel, cfg.dist_cutoff, n_heads) if cfg.use_distance_bias else None

        g = 1.0 / math.sqrt(2)
        for w in [self.W_mu, self.Wv, self.Wo]: nn.init.xavier_uniform_(w.weight, gain=g)
        nn.init.xavier_uniform_(self.W_kappa.weight, gain=0.1)
        nn.init.constant_(self.W_kappa.bias, math.log(3.0))

    def forward(self, x, fmask=None, distances=None):
        B, T, _ = x.shape
        mu = self.rope(F.normalize(self.W_mu(x).view(B, T, self.n_heads, self.hd), p=2, dim=-1, eps=1e-8))
        kappa = torch.clamp(F.softplus(self.W_kappa(x)) + 1e-4, max=30.0)

        mt = mu.permute(0, 2, 1, 3)
        scores = torch.matmul(mt, mt.transpose(-2, -1))
        kh = kappa.permute(0, 2, 1)
        scores = torch.sqrt(kh.unsqueeze(-1) * kh.unsqueeze(-2) + 1e-8) * scores
        scores = self.tau.view(1, self.n_heads, 1, 1) * scores * self._sc + self.bias_q.view(1, self.n_heads, 1, 1)

        if self.xor_bias is not None: scores = scores + self.xor_bias(mu, mu)
        if self.topo_bias is not None: scores = scores + self.topo_bias(mu)
        if self.dist_bias is not None and distances is not None: scores = scores + self.dist_bias(distances, fmask)

        if fmask is not None:
            pad = fmask < -9000.0
            scores = scores.masked_fill(pad.unsqueeze(1).unsqueeze(2), -1e4)

        attn = F.dropout(F.softmax(scores, dim=-1), p=self.dp if self.training else 0.0, training=self.training)
        v = self.Wv(x).view(B, T, self.n_heads, self.hd).permute(0, 2, 1, 3)
        out = (torch.sigmoid(self.W_gate(x).view(B, T, self.n_heads, self.hd)) * (attn @ v).permute(0, 2, 1, 3)).reshape(B, T, self.n_heads * self.hd)
        return self.Wo(out)


class LocalConvMix(nn.Module):
    def __init__(self, d, k=3, dp=0.1):
        super().__init__()
        self.norm = RMSNorm(d)
        self.dw = nn.Conv1d(d, d, k, padding=(k-1)//2, groups=d, bias=False)
        self.pw = nn.Conv1d(d, d, 1, bias=False)
        self.drop = nn.Dropout(dp)

    def forward(self, x):
        h = self.norm(x).transpose(1, 2).contiguous()
        return x + self.drop(F.gelu(self.pw(self.dw(h)), approximate="tanh").transpose(1, 2).contiguous())


class SwiGLU(nn.Module):
    def __init__(self, d, dff, dp=0.1):
        super().__init__()
        self.W_gu = nn.Linear(d, 2 * dff, bias=False)
        self.Wd = nn.Linear(dff, d, bias=False)
        self.drop = nn.Dropout(dp)

    def forward(self, x):
        g, u = self.W_gu(x).chunk(2, dim=-1)
        return self.drop(self.Wd(F.silu(g) * u))


class TransformerBlock(nn.Module):
    def __init__(self, d, nh, hd, dff, dp, sd, cfg):
        super().__init__()
        self.sd = sd
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.attn = GeometricAttention(d, nh, hd, dp, cfg)
        self.ffn = SwiGLU(d, dff, dp)

    def _drop(self, r):
        if not self.training or self.sd == 0.0: return r
        keep = (torch.rand(r.shape[0], 1, 1, device=r.device) > self.sd).float()
        return r * keep / (1.0 - self.sd)

    def forward(self, x, fmask=None, distances=None):
        x = x + self._drop(self.attn(self.norm1(x), fmask=fmask, distances=distances))
        return x + self._drop(self.ffn(self.norm2(x)))


class XORSpatialFusion(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj_think = nn.Linear(d, d, bias=False)
        self.proj_seq = nn.Linear(d, d, bias=False)

    def forward(self, think, seq_avg):
        x, y = torch.sigmoid(self.proj_think(think)), torch.sigmoid(self.proj_seq(seq_avg))
        return x * (1 - y) + (1 - x) * y


class RegressionHead(nn.Module):
    def __init__(self, d, dp=0.1, k=5):
        super().__init__()
        self.k, self.dp = k, dp
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, 1)

    def _once(self, x):
        return self.fc2(F.dropout(F.gelu(self.fc1(x), approximate="tanh"), p=self.dp, training=True)).squeeze(-1)

    def forward(self, x):
        if self.training: return self._once(x)
        return torch.stack([self._once(x) for _ in range(self.k)], dim=0).mean(0)


class GrafoPropagationGeoQM9(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.input_proj = nn.Linear(cfg.atom_feature_dim + cfg.n_z_embed, d, bias=True)
        self.embed_scale = d ** 0.5
        self.embed_drop = nn.Dropout(cfg.dropout)
        self.z_embed = nn.Embedding(10, cfg.n_z_embed, padding_idx=0)
        with torch.no_grad(): self.z_embed.weight[0].zero_()

        self.terrain = SphericalShockTerrain(cfg) if cfg.use_shock_terrain else None
        if self.terrain is not None:
            self.terrain_proj = nn.Linear(self.terrain.n_points, d, bias=False)

        self.conv_mix = LocalConvMix(d, cfg.conv_kernel, cfg.dropout)
        sd_list = [cfg.stoch_depth * i / max(cfg.n_layers - 1, 1) for i in range(cfg.n_layers)]
        self.blocks = nn.ModuleList([TransformerBlock(d, cfg.n_heads, cfg.head_dim, cfg.d_ff, cfg.dropout, sd_list[i], cfg) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(d)
        self.fusion = XORSpatialFusion(d)
        self.head = RegressionHead(d, cfg.dropout, k=5)

    def _pairwise_distances(self, positions, fmask=None):
        pos = positions.clone()
        if fmask is not None: pos[fmask < -9000.0] += 100.0
        return (pos.unsqueeze(2) - pos.unsqueeze(1)).pow(2).sum(dim=-1).sqrt()

    def encode_tokens(self, atom_features, fmask, z_indices, positions, terrain_view=None, terrain_rot_strength=None):
        valid = (fmask > -9000.0).to(atom_features.dtype).unsqueeze(-1)
        z_emb = self.z_embed(z_indices.clamp(0, 9)) * valid
        emb = self.input_proj(torch.cat([atom_features, z_emb], dim=-1)) * self.embed_scale

        if self.terrain is not None:
            shocks = self.terrain(positions, fmask=fmask, view=terrain_view, rot_strength=terrain_rot_strength)
            emb = emb + self.terrain_proj(shocks)

        emb = self.embed_drop(emb) * valid
        dists = self._pairwise_distances(positions, fmask)
        x = self.conv_mix(emb) * valid
        for blk in self.blocks:
            x = blk(x, fmask=fmask, distances=dists) * valid
        return self.final_norm(x) * valid

    def pool(self, tokens, fmask):
        valid = (fmask > -9000.0).float().unsqueeze(-1)
        seq_avg = (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        think = tokens.masked_fill(~valid.bool(), -1e9).max(dim=1).values
        return self.fusion(think, seq_avg)

    def forward(self, atom_features, fmask, z_indices, positions, return_rep=False, return_tokens=False, terrain_view=None, terrain_rot_strength=None, make_pred=True):
        tok = self.encode_tokens(atom_features, fmask, z_indices, positions, terrain_view, terrain_rot_strength)
        rep = self.pool(tok, fmask)
        pred = self.head(rep) if make_pred else None
        
        if return_rep and return_tokens: return pred, rep, tok
        if return_rep: return pred, rep
        if return_tokens: return pred, tok
        return pred


# ═════════════════════════ DEEPMIND BYOL (PRE-PRETRAIN) ══════════════════════

class DeepMindBYOLWrapper(nn.Module):
    """
    Bootstrap Your Own Latent (DeepMind).
    Fluxo assimétrico: o Student tenta aprender a geometria invariante rodando
    pelo espaço latente construído pelo seu próprio reflexo passado (o Teacher).
    O Teacher nunca tenta encaixar-se no Student ("Podes encaixar mas não ser encaixado").
    Isto resolve a falha do VICReg e estabiliza estruturalmente as representações.
    """
    def __init__(self, backbone, cfg):
        super().__init__()
        self.cfg = cfg
        # Student
        self.student_backbone = backbone
        
        # Teacher (EMA) - sem gradientes
        self.teacher_backbone = copy.deepcopy(backbone)
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False

        d = cfg.d_model
        h = cfg.prepretrain_projector_hidden
        out_dim = cfg.prepretrain_projector_out

        # Student Networks (Projector -> Predictor)
        self.student_proj = nn.Sequential(
            nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Linear(h, out_dim)
        )
        self.student_pred = nn.Sequential(
            nn.Linear(out_dim, h), nn.BatchNorm1d(h), nn.GELU(), nn.Linear(h, out_dim)
        )

        # Teacher Network (apenas Projector, igual ao original BYOL)
        self.teacher_proj = copy.deepcopy(self.student_proj)
        for p in self.teacher_proj.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self, momentum):
        for s, t in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            t.data.mul_(momentum).add_(s.data, alpha=1 - momentum)
        for s, t in zip(self.student_proj.parameters(), self.teacher_proj.parameters()):
            t.data.mul_(momentum).add_(s.data, alpha=1 - momentum)

    def forward(self, feat, fm, z, pos, rot_student, rot_teacher):
        # 1. View do Student (Encaixa na representação do Teacher)
        _, rep_s, _ = self.student_backbone(
            feat, fm, z, pos, return_rep=True, return_tokens=True,
            terrain_view="random", terrain_rot_strength=rot_student, make_pred=False
        )
        proj_s = self.student_proj(rep_s)
        pred_s = self.student_pred(proj_s)

        # 2. View do Teacher (Apenas gera o target, sem gradiente)
        with torch.no_grad():
            _, rep_t, _ = self.teacher_backbone(
                feat, fm, z, pos, return_rep=True, return_tokens=True,
                terrain_view="random", terrain_rot_strength=rot_teacher, make_pred=False
            )
            proj_t = self.teacher_proj(rep_t)

        # 3. Loss BYOL: L2 normalizado negativo ou Cosine Distance
        pred_s_norm = F.normalize(pred_s, dim=-1)
        proj_t_norm = F.normalize(proj_t.detach(), dim=-1)
        
        # O MSE de vectores normalizados é equivalente a 2 - 2 * cosine_similarity
        loss = 2 - 2 * (pred_s_norm * proj_t_norm).sum(dim=-1).mean()
        return loss


def pre_pretrain_epoch(wrapper, opt, loader, cfg, epoch, sched, gstep):
    wrapper.train()
    t0 = time.time()
    total_loss = 0.0
    samples = 0

    if cfg.prepretrain_curriculum:
        progress = (epoch - 1) / max(cfg.prepretrain_epochs - 1, 1)
        rot_strength = cfg.prepretrain_rot_start + (cfg.prepretrain_rot_end - cfg.prepretrain_rot_start) * (progress ** 0.5)
    else:
        rot_strength = cfg.prepretrain_rot_end

    opt.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        feat, fm, target, z, pos, aux = to_device(batch, cfg)
        lr = cfg.prepretrain_lr * sched.factor(gstep + 1)
        
        # Atualiza LR do optimizador original guardado no Resurgent Wrapper
        for pg in opt.optimizer.param_groups: pg["lr"] = lr

        with autocast(device_type=cfg.device.type, dtype=cfg.amp_dtype, enabled=cfg.amp_enabled):
            loss = wrapper(feat, fm, z, pos, rot_strength, rot_strength)

        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue

        loss.backward()
        
        # ==============================================================================
        # CÁLCULO ALIENÍGENA:
        # Removido o clip_grad_norm artificial. A Transformada de Borel-Laplace nativa do
        # Otimizador absorve e ressurge o gradiente, mesmo se tender a infinito.
        # ==============================================================================
        opt.step()
        opt.zero_grad(set_to_none=True)
        
        # Atualização do EMA do Teacher
        wrapper.update_teacher(cfg.prepretrain_byol_momentum)
        
        gstep += 1
        bs = feat.size(0)
        samples += bs
        total_loss += float(loss.detach().cpu()) * bs

        if step % cfg.attn_log_freq == 0 or step + 1 == len(loader):
            log(
                f"prepre_ep={epoch:03d} step={step:04d}/{len(loader)} lr={lr:.2e} "
                f"rot={rot_strength:.2f}π loss={total_loss/max(samples,1):.4f} "
                f"{time.time()-t0:.1f}s",
                "PREPRE",
            )

    return {"loss": total_loss / max(samples, 1), "time_s": round(time.time() - t0, 1), "rot_strength": rot_strength}, gstep


# ═════════════════════════ PRETRAINING MASKED ════════════════════════════════

class PretrainWrapper(nn.Module):
    def __init__(self, backbone, cfg, n_aux):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        self.mask_feat = nn.Parameter(torch.zeros(cfg.atom_feature_dim))
        d = cfg.d_model
        self.feat_head = nn.Sequential(RMSNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg.atom_feature_dim))
        self.z_head = nn.Sequential(RMSNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 5))
        self.aux_head = nn.Sequential(RMSNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_aux)) if n_aux > 0 else None
        self.register_buffer("z_to_class", torch.tensor([[-1, 0, -1, -1, -1, -1, 1, 2, 3, 4][z] for z in range(10)], dtype=torch.long))

    def forward(self, feat, fmask, z_idx, pos, aux_y):
        valid = fmask > -9000.0
        atom_mask = (torch.rand(valid.shape, device=feat.device) < self.cfg.pretrain_mask_rate) & valid
        if atom_mask.sum() == 0: atom_mask[torch.arange(feat.shape[0], device=feat.device), valid.float().argmax(dim=1)] = True

        feat_in, z_in = feat.clone(), z_idx.masked_fill(atom_mask, 0)
        feat_in[atom_mask] = self.mask_feat.to(feat.dtype)

        _, rep, tok = self.backbone(feat_in, fmask, z_in, pos, return_rep=True, return_tokens=True, terrain_view="random", make_pred=False)
        zero = rep.sum() * 0.0
        loss_z = loss_feat = loss_aux = loss_vic = zero

        if atom_mask.any():
            target_cls = self.z_to_class[z_idx.clamp(0, 9)][atom_mask]
            if (ok := target_cls >= 0).any(): loss_z = F.cross_entropy(self.z_head(tok[atom_mask])[ok].float(), target_cls[ok])
            loss_feat = F.smooth_l1_loss(self.feat_head(tok[atom_mask]).float(), feat[atom_mask].float(), beta=1.0)

        if self.aux_head is not None and aux_y is not None and (m := torch.isfinite(aux_y)).any():
            loss_aux = F.smooth_l1_loss(self.aux_head(rep)[m].float(), aux_y[m].float(), beta=1.0)

        # VICReg weight é 0.0 na config agora, não destruindo o BYOL space.
        if self.cfg.pretrain_vicreg_weight > 0:
            _, r1 = self.backbone(feat, fmask, z_idx, pos, return_rep=True, terrain_view="random", make_pred=False)
            _, r2 = self.backbone(feat, fmask, z_idx, pos, return_rep=True, terrain_view="random", make_pred=False)
            loss_vic = F.mse_loss(r1, r2) 

        loss = self.cfg.pretrain_aux_weight * loss_aux + self.cfg.pretrain_atom_weight * (loss_z + loss_feat) + self.cfg.pretrain_vicreg_weight * loss_vic
        return loss, {"loss": float(loss.detach()), "aux": float(loss_aux.detach()), "z": float(loss_z.detach()), "feat": float(loss_feat.detach()), "vic": float(loss_vic.detach())}


def pretrain_epoch(wrapper, opt, loader, cfg, epoch, sched, gstep):
    wrapper.train()
    t0, totals, samples = time.time(), {"loss": 0.0, "aux": 0.0, "z": 0.0, "feat": 0.0, "vic": 0.0}, 0
    opt.zero_grad(set_to_none=True)
    
    for step, batch in enumerate(loader):
        feat, fm, target, z, pos, aux = to_device(batch, cfg)
        lr = cfg.pretrain_lr * sched.factor(gstep + 1)
        for pg in opt.optimizer.param_groups: pg["lr"] = lr

        with autocast(device_type=cfg.device.type, dtype=cfg.amp_dtype, enabled=cfg.amp_enabled):
            loss, logs = wrapper(feat, fm, z, pos, aux)

        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue

        loss.backward()
        
        # CÁLCULO ALIENÍGENA: O gradiente flui analiticamente no plano de Borel. Otimizador lida com os infinitos.
        opt.step()
        opt.zero_grad(set_to_none=True)
        gstep += 1

        bs = feat.size(0)
        samples += bs
        for k in totals: totals[k] += logs[k] * bs

        if step % cfg.attn_log_freq == 0 or step + 1 == len(loader):
            avg = {k: totals[k] / max(samples, 1) for k in totals}
            log(f"pre_ep={epoch:03d} step={step:04d}/{len(loader)} lr={lr:.2e} loss={avg['loss']:.4f} aux={avg['aux']:.4f} z={avg['z']:.4f} feat={avg['feat']:.4f} vic={avg['vic']:.4f}", "PRE")

    return {k: totals[k] / max(samples, 1) for k in totals} | {"time_s": round(time.time() - t0, 1)}, gstep


# ═════════════════════════ DATASET / OPTIM ═══════════════════════════════════

class QM9Dataset(Dataset):
    def __init__(self, atom_features, fmask, targets, z_indices, positions, aux_targets, indices):
        self.atom_features, self.fmask, self.targets = atom_features[indices], fmask[indices], targets[indices]
        self.z_indices, self.positions = z_indices[indices], positions[indices]
        self.aux_targets = aux_targets[indices] if aux_targets is not None else torch.empty((len(indices), 0))
    def __len__(self): return len(self.targets)
    def __getitem__(self, i): return (self.atom_features[i], self.fmask[i], self.targets[i], self.z_indices[i], self.positions[i], self.aux_targets[i])

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.parameters(), model.parameters()): s.lerp_(m.float(), 1.0 - self.decay)
        for sb, mb in zip(self.shadow.buffers(), model.buffers()): sb.copy_(mb)

# ═════════════════════════ CÁLCULO ALIENÍGENA (ÉCALLE, 1981) ══════════════════
class BorelLaplaceResurgentOptimizer:
    """
    Otimizador Resurgente Não-Perturbativo.
    Substitui a técnica "cega" do gradient clipping artificial. 
    Se o gradiente tende ao infinito, ele é projetado no Plano de Borel, onde
    o comportamento divergente se transforma num Pólo Analítico, e um Instanton (Laplace) 
    injeta um momento não-linear direto para a bacia de atração do verdadeiro mínimo.
    """
    def __init__(self, optimizer, action_threshold=1.0):
        self.optimizer = optimizer
        self.action_threshold = action_threshold
        
    @property
    def param_groups(self): return self.optimizer.param_groups
    
    @property
    def state(self): return self.optimizer.state
    
    @property
    def defaults(self): return self.optimizer.defaults

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        grads = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grads.append(p.grad)
                    
        if not grads:
            return self.optimizer.step(closure)
            
        # 1. Indicador de Divergência Global (Norma no espaço euclidiano atual)
        global_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2).to(torch.float32) for g in grads]), 2)
        
        if global_norm > 1e-8:
            eta = self.optimizer.param_groups[0].get("lr", 1e-4) + 1e-12
            
            # 2. Transformada de Borel (Mapeamento Analítico do Infinito)
            # Ao invés do erro numérico do float32 (NaN), a explosão fatorial é enclausurada:
            # lim_{global_norm -> inf} borel_pole -> 0 (Impede overflow do hardware físico)
            borel_pole = 1.0 / (1.0 + global_norm * eta)
            
            # 3. Trans-série / Instanton (Integração Direcional de Laplace)
            # O salto quântico no ponto de singularidade. Um túnel não-perturbativo criado
            # pelo extremo erro da derivada original, cruzando o espaço de fase.
            trans_series_instanton = torch.exp(-self.action_threshold / (global_norm * eta + 1e-12))
            
            # A escala final é o "caminho resurgente" - A Matemática no seu estado puro
            resurgent_scale = borel_pole + trans_series_instanton * (1.0 / (global_norm + 1e-12))
            
            for g in grads:
                g.mul_(resurgent_scale.to(g.dtype))
        
        # O AdamW processa o tensor matematicamente curado pelas derivadas alienígenas
        return self.optimizer.step(closure)
# ══════════════════════════════════════════════════════════════════════════════

class Lookahead:
    def __init__(self, optimizer, k=6, alpha=0.5):
        self.optimizer, self.k, self.alpha, self._steps, self._slow = optimizer, k, alpha, 0, {}
    @property
    def state(self): return self.optimizer.state
    @property
    def param_groups(self): return self.optimizer.param_groups
    @property
    def defaults(self): return self.optimizer.defaults
    def zero_grad(self, set_to_none=True): self.optimizer.zero_grad(set_to_none=set_to_none)
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._steps += 1
        if not self._slow: self._slow = {id(p): p.data.clone().detach() for group in self.optimizer.param_groups for p in group["params"]}
        if self._steps % self.k == 0:
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    s = self._slow[id(p)]
                    s.add_(self.alpha * (p.data - s))
                    p.data.copy_(s)
        return loss

class AWP:
    def __init__(self, model, eps=0.003, lr=0.005):
        self.model, self.eps, self.lr, self.backup, self.on = model, eps, lr, {}, False
    def perturb(self):
        if self.on: return
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None and torch.isfinite(gn := p.grad.float().norm()) and gn > 0:
                self.backup[n] = p.data.clone()
                p.data.add_((self.lr * p.grad.float() / (gn + 1e-8)).clamp(-self.eps, self.eps).to(p.dtype))
        self.on = True
    def restore(self):
        for n, p in self.model.named_parameters():
            if n in self.backup: p.data.copy_(self.backup[n])
        self.backup.clear()
        self.on = False

def register_gc(model):
    return [p.register_hook(lambda g: g - g.mean(tuple(range(1, g.dim())), keepdim=True) if g.dim() > 1 else g) for n, p in model.named_parameters() if p.requires_grad and p.dim() > 1 and "input_proj" not in n and "z_embed" not in n]

class WarmupCosineLR:
    def __init__(self, total, wf, mf): self.T, self.W, self.mf = max(int(total), 1), max(int(wf * total), 1), mf
    def factor(self, step):
        if step < self.W: return max(step, 1) / self.W
        return self.mf + (1.0 - self.mf) * 0.5 * (1.0 + math.cos(math.pi * min(max((step - self.W) / max(self.T - self.W, 1), 0.0), 1.0)))


# ═════════════════════════ TRAIN / EVAL ══════════════════════════════════════

def to_device(batch, cfg):
    return tuple(t.to(cfg.device, non_blocking=True) for t in batch)

def compute_regression_metrics(pred, target, y_mean, y_std):
    pred_o, tgt_o = pred * y_std + y_mean, target * y_std + y_mean
    diff, absd = pred_o - tgt_o, (pred_o - tgt_o).abs()
    ss_res, ss_tot = diff.pow(2).sum().item(), (tgt_o - tgt_o.mean()).pow(2).sum().item()
    pm, tm = pred_o - pred_o.mean(), tgt_o - tgt_o.mean()
    nz = tgt_o.abs() > 0.01
    return {
        "mae": absd.mean().item(), "rmse": diff.pow(2).mean().sqrt().item(),
        "r2": 1.0 - ss_res / max(ss_tot, 1e-8), "median_ae": absd.median().item(),
        "max_ae": absd.max().item(), "mape": (absd[nz] / tgt_o[nz].abs()).mean().item() * 100.0 if nz.any() else 0.0,
        "pearson_r": ((pm * tm).sum() / (pm.norm() * tm.norm()).clamp(min=1e-8)).item(),
        "pred_mean": pred_o.mean().item(), "pred_std": pred_o.std().item(), "target_mean": tgt_o.mean().item(), "target_std": tgt_o.std().item()
    }


def train_epoch(model, ema, optimizer, base_opt, loader, awp, cfg, epoch, gstep, sched):
    model.train()
    t0, total_loss, total_mae_ev, samples, grad_sum, grad_count = time.time(), 0.0, 0.0, 0, 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        feat, fm, target, z, pos, aux = to_device(batch, cfg)
        lr = cfg.base_lr_max * sched.factor(gstep + 1)
        
        # O optimizer agora é o BorelLaplaceResurgentOptimizer (wrapper do AdamW)
        for pg in optimizer.optimizer.param_groups: pg["lr"] = lr

        with autocast(device_type=cfg.device.type, dtype=cfg.amp_dtype, enabled=cfg.amp_enabled):
            pred = model(feat, fm, z, pos)
            loss = F.smooth_l1_loss(pred, target, beta=cfg.huber_delta)

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        (loss / cfg.grad_accum).backward()

        if (step + 1) % cfg.grad_accum == 0:
            grad_sum += min(sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5, 1000.0)
            grad_count += 1
            if epoch >= cfg.awp_start_ep:
                awp.perturb()
                with autocast(device_type=cfg.device.type, dtype=cfg.amp_dtype, enabled=cfg.amp_enabled):
                    (F.smooth_l1_loss(model(feat, fm, z, pos), target, beta=cfg.huber_delta) / cfg.grad_accum).backward()
                awp.restore()
            
            # CÁLCULO ALIENÍGENA: sem clip, o ResurgentOptimizer trata dos gradientes
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            gstep += 1

        bs = target.size(0)
        samples += bs
        total_loss += loss.item() * bs
        total_mae_ev += ((pred.detach() - target).abs() * cfg._y_std).sum().item()

        if step % cfg.attn_log_freq == 0 or step + 1 == len(loader):
            log(f"ep={epoch:03d} step={step:04d}/{len(loader)} lr={lr:.2e} loss={loss.item():.5f} mae_eV={total_mae_ev/max(samples,1):.4f} gnorm_raw={grad_sum/max(grad_count,1):.3f} {time.time()-t0:.1f}s", "TRAIN")

    return {"loss": total_loss / max(samples, 1), "mae_ev": total_mae_ev / max(samples, 1), "grad_norm": grad_sum / max(grad_count, 1), "lr": lr, "time_s": round(time.time() - t0, 1)}, gstep

@torch.no_grad()
def evaluate(model, loader, cfg, y_mean, y_std):
    model.eval()
    t0, total_loss, samples, preds, tgts = time.time(), 0.0, 0, [], []
    for batch in loader:
        feat, fm, target, z, pos, aux = to_device(batch, cfg)
        with autocast(device_type=cfg.device.type, dtype=cfg.amp_dtype, enabled=cfg.amp_enabled):
            pred = model(feat, fm, z, pos)
            total_loss += F.smooth_l1_loss(pred, target, beta=1.0).item() * target.size(0)
        samples += target.size(0)
        preds.append(pred.float().cpu())
        tgts.append(target.float().cpu())

    m = compute_regression_metrics(torch.cat(preds), torch.cat(tgts), y_mean, y_std)
    m["loss"] = total_loss / max(samples, 1)
    m["time_s"] = round(time.time() - t0, 1)
    return m


# ═════════════════════════ MAIN ══════════════════════════════════════════════

def main():
    cfg = CFG()
    ensure_paths(cfg)
    set_seed(cfg.seed)

    log_separator(f"{cfg.VERSION} · ALIEN CALCULUS RESURGENCE · Run {RUN_ID}")
    
    raw = precompute_qm9_data(cfg)
    z_all, pos_all, fm_all, y_raw, targets_all = raw["z_indices"], raw["positions"], raw["fmask"], raw["targets_raw"].float(), raw["targets_all"].float()
    N = len(y_raw)
    train_idx, val_idx, test_idx = make_splits(N, cfg)
    
    y_mean, y_std = float(y_raw[train_idx].mean()), float(y_raw[train_idx].std().clamp_min(1e-8))
    y_all = (y_raw - y_mean) / y_std if cfg.normalize_y else y_raw.clone()
    if not cfg.normalize_y: y_mean, y_std = 0.0, 1.0
    cfg._y_std = y_std

    aux_norm, aux_cols, aux_mean, aux_std = build_aux_targets_train_only(targets_all, train_idx, cfg.gap_idx)
    atom_features, feat_mean, feat_std = build_atom_features_train_only(raw, train_idx, cfg)

    train_ds = QM9Dataset(atom_features, fm_all, y_all, z_all, pos_all, aux_norm, train_idx)
    val_ds = QM9Dataset(atom_features, fm_all, y_all, z_all, pos_all, aux_norm, val_idx)
    test_ds = QM9Dataset(atom_features, fm_all, y_all, z_all, pos_all, aux_norm, test_idx)

    loader_kw = dict(num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, persistent_workers=cfg.num_workers > 0)
    tr_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **loader_kw)
    pre_ld = DataLoader(train_ds, batch_size=cfg.pretrain_batch_size, shuffle=True, drop_last=True, **loader_kw)
    
    prepre_ld = DataLoader(train_ds, batch_size=cfg.prepretrain_batch_size, shuffle=True, drop_last=True, **loader_kw)
    
    va_ld = DataLoader(val_ds, batch_size=256, shuffle=False, drop_last=False, **loader_kw)
    te_ld = DataLoader(test_ds, batch_size=256, shuffle=False, drop_last=False, **loader_kw)

    log_separator("Building model")
    model = GrafoPropagationGeoQM9(cfg).to(cfg.device)
    params = sum(p.numel() for p in model.parameters())
    log(f"Parameters: {params:,} ({params/1e6:.3f}M)", "INFO")

    # ── PRE-PRETRAIN (BYOL) ──
    if cfg.use_prepretrain and cfg.prepretrain_epochs > 0:
        log_separator("PRE-PRETRAINING · DEEPMIND BYOL ASYMMETRIC")
        prepre_wrapper = DeepMindBYOLWrapper(model, cfg).to(cfg.device)
        base_opt_prepre = torch.optim.AdamW(prepre_wrapper.parameters(), lr=cfg.prepretrain_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=cfg.prepretrain_wd)
        opt_prepre = BorelLaplaceResurgentOptimizer(base_opt_prepre)
        
        sched_prepre = WarmupCosineLR(cfg.prepretrain_epochs * len(prepre_ld), cfg.prepretrain_warmup_frac, cfg.prepretrain_min_lr_frac)
        gprepre, prepre_hist = 0, []
        for ep in range(1, cfg.prepretrain_epochs + 1):
            ps, gprepre = pre_pretrain_epoch(prepre_wrapper, opt_prepre, prepre_ld, cfg, ep, sched_prepre, gprepre)
            prepre_hist.append({"epoch": ep, **ps})
            log(f"PREPRE EPOCH {ep:03d} │ loss={ps['loss']:.5f} rot={ps['rot_strength']:.2f}π time={ps['time_s']}s", "METRIC")

        model = prepre_wrapper.student_backbone
        torch.save({"model": model.state_dict()}, os.path.join(cfg.checkpoint_dir, "prepretrained_backbone.pt"))
        del prepre_wrapper

    # ── PRETRAIN (Masked / No VICReg) ──
    if cfg.pretrain_epochs > 0:
        log_separator("TRAIN-ONLY PRETRAINING")
        wrapper = PretrainWrapper(model, cfg, n_aux=aux_norm.shape[1]).to(cfg.device)
        base_opt_pre = torch.optim.AdamW(wrapper.parameters(), lr=cfg.pretrain_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=cfg.pretrain_wd)
        opt_pre = BorelLaplaceResurgentOptimizer(base_opt_pre)
        
        sched_pre = WarmupCosineLR(cfg.pretrain_epochs * len(pre_ld), cfg.warmup_frac, cfg.min_lr_frac)
        gpre, pre_hist = 0, []
        for ep in range(1, cfg.pretrain_epochs + 1):
            ps, gpre = pretrain_epoch(wrapper, opt_pre, pre_ld, cfg, ep, sched_pre, gpre)
            pre_hist.append({"epoch": ep, **ps})
            log(f"PRE EPOCH {ep:03d} │ loss={ps['loss']:.5f} aux={ps['aux']:.5f} z={ps['z']:.5f} feat={ps['feat']:.5f}", "METRIC")

        model = wrapper.backbone
        torch.save({"model": model.state_dict()}, os.path.join(cfg.checkpoint_dir, "pretrained_backbone.pt"))
        del wrapper

    # ── FINETUNE ──
    log_separator("FINETUNING (ALIEN CALCULUS · Lookahead REMOVIDO · LR 1e-3)")
    ema = EMA(model, cfg.ema_decay)
    
    # Optimizador Base com LR maior
    base_opt = torch.optim.AdamW(model.parameters(), lr=0.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=cfg.wd)
    
    # Apenas o BorelLaplaceResurgentOptimizer, sem Lookahead
    optimizer = BorelLaplaceResurgentOptimizer(base_opt)
    
    awp, gc_hooks = AWP(model, cfg.awp_eps, cfg.awp_lr), register_gc(model)
    sched = WarmupCosineLR(cfg.epochs * len(tr_ld), cfg.warmup_frac, cfg.min_lr_frac)
    best_mae, best_epoch, best_r2, gstep, history = float("inf"), 0, -float("inf"), 0, []

    for epoch in range(1, cfg.epochs + 1):
        tr_s, gstep = train_epoch(model, ema, optimizer, base_opt, tr_ld, awp, cfg, epoch, gstep, sched)
        va_s = evaluate(ema.shadow, va_ld, cfg, y_mean, y_std)

        if va_s["mae"] < best_mae:
            best_mae, best_epoch, best_r2 = va_s["mae"], epoch, va_s["r2"]
            torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict(), "y_mean": y_mean, "y_std": y_std}, os.path.join(cfg.checkpoint_dir, "best_model.pt"))
            log(f"★ New best: MAE={best_mae:.6f} eV R2={best_r2:.6f} epoch={epoch}", "METRIC")

        log(f"EPOCH {epoch:03d} │ tr_loss={tr_s['loss']:.6f} tr_mae={tr_s['mae_ev']:.4f}eV │ val_mae={va_s['mae']:.4f}eV val_rmse={va_s['rmse']:.4f}eV R2={va_s['r2']:.5f} │ best={best_mae:.4f}eV", "METRIC")
        history.append({"epoch": epoch, "val_mae": va_s["mae"], "best_mae": best_mae})
        if epoch % cfg.checkpoint_every == 0: torch.save({"epoch": epoch, "model": model.state_dict(), "ema": ema.shadow.state_dict()}, os.path.join(cfg.checkpoint_dir, f"ep{epoch:03d}.pt"))

    log_separator("FINAL TEST")
    ckpt = torch.load(os.path.join(cfg.checkpoint_dir, "best_model.pt"), map_location=cfg.device, weights_only=False)
    ema.shadow.load_state_dict(ckpt["ema"])
    test_s = evaluate(ema.shadow, te_ld, cfg, ckpt["y_mean"], ckpt["y_std"])
    
    if HAS_RICH:
        t = Table(title="FINAL TEST · QM9 HOMO-LUMO Gap", box=box.DOUBLE_EDGE)
        t.add_column("Metric", style="bold")
        t.add_column("Value", justify="right", style="bold green")
        t.add_row("Test MAE", f"{test_s['mae']:.6f} eV")
        t.add_row("Test RMSE", f"{test_s['rmse']:.6f} eV")
        t.add_row("Test R²", f"{test_s['r2']:.8f}")
        t.add_row("Test Pearson", f"{test_s['pearson_r']:.8f}")
        t.add_row("Test MedianAE", f"{test_s['median_ae']:.6f} eV")
        t.add_row("Test MaxAE", f"{test_s['max_ae']:.6f} eV")
        t.add_row("Best Val MAE", f"{best_mae:.6f} eV")
        t.add_row("Best Epoch", str(best_epoch))
        t.add_row("Parameters", f"{params:,} ({params/1e6:.3f}M)")
        t.add_row("Run ID", RUN_ID)
        console.print(t)
    else:
        log(f"FINAL TEST MAE={test_s['mae']:.6f} RMSE={test_s['rmse']:.6f} R2={test_s['r2']:.8f}", "METRIC")

    for h in gc_hooks: h.remove()
    log(f"DONE · {cfg.VERSION} · Run {RUN_ID}", "METRIC")

if __name__ == "__main__":
    main()
