"""
GrafoPropagation v26-APEX — Training Pipeline
==============================================

Complete two-phase training:
  Phase 1: WordNet dictionary pre-training (multi-label BCE)
  Phase 2: AG News fine-tuning with full regularisation stack

(c) 2025-2026 Claudio Fernandes. All rights reserved.
"""

import os
import math
import time
import random
import json
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from .config import CFG, set_seed
from .model import GrafoPropagation
from .tokenizer_utils import build_or_load_tokenizer
from .datasets import (
    TextDataset, DictionaryDataset, collate_dict, build_wordnet_multilabel,
)
from .losses import focal_ce, token_dropout
from .optimizer import EMA, Lookahead, AWP, register_gc, WarmupCosineLR
from .quantum import quantum_lr_modulation
from .logging_utils import log, console, RUN_ID

try:
    from rich.console import Console
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1 — Dictionary Pre-Training
# ═══════════════════════════════════════════════════════════════════════

def pretrain_dictionary(model, mapping, cfg, scaler, base_opt, optimizer,
                        pad_id, cls_id):
    """
    Pre-train the model with weighted multi-label BCE loss over
    ``cfg.dict_epochs`` epochs so that it learns the semantic content
    of WordNet definitions.
    """
    log(f"Starting dictionary pre-training ({cfg.dict_epochs} epochs) …", "SYS2")
    ds = DictionaryDataset(mapping)
    dl = DataLoader(
        ds, batch_size=cfg.dict_batch_size, shuffle=True,
        collate_fn=lambda b: collate_dict(b, pad_id, cls_id),
        num_workers=0, pin_memory=True,
    )
    model.train()

    # Positive-class reweighting to counter extreme label imbalance
    pos_weight = torch.tensor([cfg.dict_pos_weight], device=cfg.device)

    for epoch in range(1, cfg.dict_epochs + 1):
        # Pure cosine annealing for the dictionary phase
        dict_lr = cfg.base_lr_max * (
            0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * epoch / cfg.dict_epochs))
        )
        base_opt.param_groups[0]["lr"] = dict_lr

        t0 = time.time()
        total_loss = 0.0

        for step, (ids, fm, labels) in enumerate(dl):
            ids = ids.to(cfg.device, non_blocking=True)
            fm = fm.to(cfg.device, non_blocking=True)
            labels = labels.to(cfg.device, non_blocking=True)

            with autocast(device_type=cfg.device.type, dtype=cfg.amp_dtype):
                _, _, dict_logits = model(ids, fm, return_dict_logits=True)
                dict_loss = F.binary_cross_entropy_with_logits(
                    dict_logits, labels, pos_weight=pos_weight,
                )
                loss = cfg.dict_lambda * dict_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()

        # Log every 50 epochs to avoid spam
        if epoch % 50 == 0 or epoch == cfg.dict_epochs:
            ela = time.time() - t0
            log(f"  Dict Epoch {epoch:04d}/{cfg.dict_epochs} | "
                f"lr={dict_lr:.6f} | avg_loss={total_loss / len(dl):.5f} | "
                f"{ela:.1f}s", "SYS2")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2 — Fine-Tuning on AG News
# ═══════════════════════════════════════════════════════════════════════

def _reg_params(cfg, epoch):
    """Anneal regularisation as training progresses."""
    p = min(1.0, epoch / cfg.epochs)
    mixp = cfg.mixup_prob_base - (cfg.mixup_prob_base - cfg.mixup_prob_min) * p
    ls = cfg.label_smooth_base - (cfg.label_smooth_base - cfg.label_smooth_min) * p
    return mixp, ls


def train_epoch(model, ema, optimizer, scaler, loader, awp, cfg, epoch,
                gstep, lr_sched, base_opt, qlr_mod, unk_id):
    model.train()
    mixp, ls = _reg_params(cfg, epoch)
    n = len(loader)
    t0 = time.time()
    st = {"loss": 0.0, "ce": 0.0, "cor": 0, "tot": 0}
    optimizer.zero_grad(set_to_none=True)

    for step, (ids, fm, lbl) in enumerate(loader):
        ids = ids.to(cfg.device, non_blocking=True)
        fm = fm.to(cfg.device, non_blocking=True)
        lbl = lbl.to(cfg.device, non_blocking=True)
        ids = token_dropout(ids, fm, unk_id, cfg.token_dropout_prob, cfg.token_dropout_apply)
        use_mx = random.random() < mixp

        with autocast(device_type=cfg.device.type, dtype=cfg.amp_dtype):
            if use_mx:
                emb = model.embed_drop(model.embed(ids) * model.embed_scale)
                lam = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                idx2 = torch.randperm(emb.size(0), device=cfg.device)
                pooled, ent_loss, _ = model.encode(
                    lam * emb + (1.0 - lam) * emb[idx2], fm,
                )
                logits = model.head(pooled)
                lp = F.log_softmax(logits, -1)
                C = cfg.n_classes
                t1 = torch.full_like(lp, ls / (C - 1))
                t1.scatter_(-1, lbl.unsqueeze(-1), 1.0 - ls)
                t2 = torch.full_like(lp, ls / (C - 1))
                t2.scatter_(-1, lbl[idx2].unsqueeze(-1), 1.0 - ls)
                ce = (lam * (-(t1 * lp).sum(-1).mean())
                      + (1.0 - lam) * (-(t2 * lp).sum(-1).mean()))
            else:
                logits, ent_loss, _ = model(ids, fm)
                ce = focal_ce(logits, lbl, cfg.focal_gamma, ls)

            div = model.system2.diversity_loss() * cfg.div_weight
            loss = ce + div + ent_loss

        scaler.scale(loss / cfg.grad_accum).backward()

        if (step + 1) % cfg.grad_accum == 0:
            scaler.unscale_(optimizer)
            if epoch >= cfg.awp_start_ep:
                awp.perturb()
                with autocast(device_type=cfg.device.type, dtype=cfg.amp_dtype):
                    logits_awp, _, _ = model(ids, fm)
                    scaler.scale(
                        focal_ce(logits_awp, lbl, cfg.focal_gamma, ls) / cfg.grad_accum
                    ).backward()
                awp.restore()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            gstep += 1
            base_opt.param_groups[0]["lr"] = (
                cfg.base_lr_max * lr_sched.factor(gstep) * qlr_mod
            )

        st["loss"] += float(loss.item())
        st["ce"] += float(ce.item())
        if not use_mx:
            st["cor"] += (logits.argmax(-1) == lbl).sum().item()
            st["tot"] += lbl.size(0)

        if step % cfg.attn_log_freq == 0 or step == n - 1:
            ela = time.time() - t0
            eta = ela / (step + 1) * (n - step - 1)
            log(f"ep={epoch:2d} step={step:04d}/{n} "
                f"lr={base_opt.param_groups[0]['lr']:.6f} "
                f"loss={loss.item():.5f} ce={ce.item():.5f} "
                f"ent={float(ent_loss.item()):.5f} "
                f"tr_acc={st['cor'] / max(st['tot'], 1) * 100:.2f}% "
                f"{ela:.1f}s ETA={eta:.1f}s", "ATTN")

    return {
        "loss": st["loss"] / n,
        "ce": st["ce"] / n,
        "acc": st["cor"] / max(st["tot"], 1),
        "time_s": round(time.time() - t0, 1),
    }, gstep


@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval()
    cor = tot = 0
    vl = 0.0
    for ids, fm, lbl in loader:
        ids = ids.to(cfg.device, non_blocking=True)
        fm = fm.to(cfg.device, non_blocking=True)
        lbl = lbl.to(cfg.device, non_blocking=True)
        with autocast(device_type=cfg.device.type, dtype=cfg.amp_dtype):
            logits, _, _ = model(ids, fm)
        vl += F.cross_entropy(logits, lbl).item() * lbl.size(0)
        cor += (logits.argmax(-1) == lbl).sum().item()
        tot += lbl.size(0)
    return {"acc": cor / max(tot, 1), "loss": vl / max(tot, 1)}


# ═══════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def run_training(cfg: CFG = None, **overrides):
    """
    Execute the full training pipeline.

    Parameters
    ----------
    cfg : CFG instance (default: ``CFG()``)
    **overrides : any CFG field to override, e.g. ``d_model=128``

    Returns
    -------
    dict with best_val_acc, history, config
    """
    if cfg is None:
        cfg = CFG(**overrides)
    elif overrides:
        cfg = cfg.update(**overrides)

    set_seed(cfg.seed)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    if HAS_RICH:
        console.rule(
            f"[bold green]GrafoPropagation {cfg.VERSION} · "
            f"WordNet Pretrain · AG News · Run {RUN_ID}[/bold green]"
        )
    log(f"device={cfg.device}  amp={cfg.amp_dtype}")

    # ── Tokenizer ─────────────────────────────────────────────────────
    log("Loading AG News (for tokenizer) …", "INFO")
    from datasets import load_dataset
    raw = load_dataset(cfg.dataset_name)
    tr_raw = raw["train"].shuffle(cfg.seed).select(
        range(min(5000, len(raw["train"])))
    )
    tok = build_or_load_tokenizer([str(r["text"]) for r in tr_raw], cfg)
    unk_id = tok.token_to_id(cfg.unk_token)
    pad_id = tok.token_to_id(cfg.pad_token)
    cls_id = tok.token_to_id(cfg.cls_token)

    # ── WordNet multi-label mapping ───────────────────────────────────
    mapping = build_wordnet_multilabel(tok, cfg)

    # ── Model ─────────────────────────────────────────────────────────
    model = GrafoPropagation(cfg, tok).to(cfg.device)
    ema = EMA(model, cfg.ema_decay)
    total = sum(p.numel() for p in model.parameters())
    log(f"Total parameters: {total:,}  ({total / 1e6:.3f} M)", "SYS2")

    # ── Optimiser ─────────────────────────────────────────────────────
    base_opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.base_lr_max,
        betas=(0.9, 0.999), eps=1e-8, weight_decay=cfg.wd,
    )
    optimizer = Lookahead(base_opt, cfg.la_k, cfg.la_alpha)
    scaler = GradScaler("cuda", enabled=(cfg.amp_dtype == torch.float16))
    awp = AWP(model, scaler, cfg.awp_eps, cfg.awp_lr)
    gc_h = register_gc(model)
    log(f"GradCentralization hooks: {len(gc_h)}")

    # ── Phase 1: Dictionary Pre-Training ──────────────────────────────
    pretrain_dictionary(model, mapping, cfg, scaler, base_opt, optimizer,
                        pad_id, cls_id)

    # ── Phase 2: Fine-Tuning ──────────────────────────────────────────
    log("Starting Fine-Tuning AG News …", "INFO")
    tr_raw_full = raw["train"].shuffle(cfg.seed).select(
        range(min(cfg.max_train, len(raw["train"])))
    )
    va_raw = raw["test"].shuffle(cfg.seed).select(
        range(min(cfg.max_val, len(raw["test"])))
    )
    tr_ds = TextDataset(tr_raw_full, tok, cfg)
    va_ds = TextDataset(va_raw, tok, cfg)
    tr_ld = DataLoader(
        tr_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
        persistent_workers=True,
    )
    va_ld = DataLoader(
        va_ds, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    log(f"Train batches={len(tr_ld)}  Val batches={len(va_ld)}  "
        f"eff_bs={cfg.batch_size * cfg.grad_accum}")

    total_steps = cfg.epochs * (len(tr_ld) // cfg.grad_accum)
    lr_sched = WarmupCosineLR(total_steps, cfg.warmup_frac, cfg.min_lr_frac)
    best_acc = 0.0
    gstep = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        qlr = quantum_lr_modulation(epoch) if cfg.use_quantum_lr else 1.0
        log(f"EPOCH {epoch}/{cfg.epochs}  qlr_mod={qlr:.4f}")
        tr_s, gstep = train_epoch(
            model, ema, optimizer, scaler, tr_ld, awp, cfg,
            epoch, gstep, lr_sched, base_opt, qlr, unk_id,
        )
        va_s = evaluate(ema.shadow, va_ld, cfg)

        if HAS_RICH:
            t = Table(
                title=f"Epoch {epoch} · {cfg.VERSION} · AG News",
                show_lines=True,
            )
            t.add_column("Metric", style="bold", width=28)
            t.add_column("Value", style="green", width=36)
            t.add_row("Train Loss", f'{tr_s["loss"]:.6f}')
            t.add_row("Train CE", f'{tr_s["ce"]:.6f}')
            t.add_row("Train Acc", f'{tr_s["acc"] * 100:.3f}%')
            t.add_row("Val Acc (EMA)", f'[bold]{va_s["acc"] * 100:.3f}%[/bold]')
            t.add_row("Val Loss", f'{va_s["loss"]:.6f}')
            t.add_row("Best so far", f"[bold green]{best_acc * 100:.3f}%[/bold green]")
            t.add_row("Time epoch", f'{tr_s["time_s"]:.1f}s')
            console.print(t)

        log(f'EPOCH_END {epoch}  val_acc={va_s["acc"] * 100:.4f}%  '
            f'tr_acc={tr_s["acc"] * 100:.4f}%  t={tr_s["time_s"]:.1f}s', "METRIC")

        history.append({"epoch": epoch, "train": tr_s, "val": va_s})

        if va_s["acc"] > best_acc:
            best_acc = va_s["acc"]
            torch.save({
                "model": model.state_dict(),
                "ema": ema.shadow.state_dict(),
                "epoch": epoch,
                "acc": best_acc,
                "run_id": RUN_ID,
                "history": history,
                "config": cfg.to_dict(),
            }, os.path.join(cfg.checkpoint_dir, "best_model.pt"))
            log(f"New best: {best_acc * 100:.4f}%", "METRIC")

        if epoch % cfg.checkpoint_every == 0:
            p = os.path.join(cfg.checkpoint_dir, f"ep{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema.shadow.state_dict(),
                "best_acc": best_acc,
                "run_id": RUN_ID,
                "history": history,
                "config": cfg.to_dict(),
            }, p)
            log(f"Periodic checkpoint: {p}", "METRIC")

    for h in gc_h:
        h.remove()
    log(f"DONE  best_val_acc={best_acc * 100:.4f}%", "METRIC")

    hist_p = os.path.join(cfg.checkpoint_dir, f"history_{RUN_ID}.json")
    with open(hist_p, "w") as f:
        json.dump(history, f, indent=2, default=str)
    log(f"History → {hist_p}")

    return {"best_val_acc": best_acc, "history": history, "config": cfg.to_dict()}
