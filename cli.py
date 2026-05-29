"""
GrafoPropagation v26-APEX — CLI Entry Point
============================================

Usage examples
--------------
    # Default training (~990k params)
    grafoprop-train

    # Scale up architecture
    grafoprop-train --d_model 128 --n_layers 4 --n_heads 8 --head_dim 32

    # Custom training schedule
    grafoprop-train --epochs 50 --batch_size 128 --base_lr_max 0.002

    # Disable quantum LR
    grafoprop-train --use_quantum_lr false

    # Save config to JSON and load it later
    grafoprop-train --export_config my_config.json
    grafoprop-train --config my_config.json

(c) 2025-2026 Claudio Fernandes. All rights reserved.
"""

import argparse
import json
import sys

from .config import CFG
from .train import run_training


def _parse_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def _build_parser():
    p = argparse.ArgumentParser(
        description="GrafoPropagation v26-APEX — Train from the CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    p.add_argument("--config", type=str, default=None,
                   help="Path to JSON config file (overrides defaults)")
    p.add_argument("--export_config", type=str, default=None,
                   help="Export effective config to this JSON file and exit")

    # Architecture
    arch = p.add_argument_group("Architecture")
    arch.add_argument("--d_model", type=int, default=None)
    arch.add_argument("--n_layers", type=int, default=None)
    arch.add_argument("--n_heads", type=int, default=None)
    arch.add_argument("--head_dim", type=int, default=None)
    arch.add_argument("--d_ff", type=int, default=None)
    arch.add_argument("--dropout", type=float, default=None)
    arch.add_argument("--stoch_depth", type=float, default=None)
    arch.add_argument("--conv_kernel", type=int, default=None)
    arch.add_argument("--char_dim", type=int, default=None)

    # System-2
    sys2 = p.add_argument_group("System-2")
    sys2.add_argument("--K_think", type=int, default=None)
    sys2.add_argument("--memory_slots", type=int, default=None)
    sys2.add_argument("--d_cot_ff", type=int, default=None)
    sys2.add_argument("--search_branches", type=int, default=None)
    sys2.add_argument("--max_refinements", type=int, default=None)
    sys2.add_argument("--latent_actions", type=int, default=None)
    sys2.add_argument("--mcts_simulations", type=int, default=None)
    sys2.add_argument("--mcts_rollout_depth", type=int, default=None)

    # vMF
    vmf = p.add_argument_group("vMF Attention")
    vmf.add_argument("--vmf_kappa_init", type=float, default=None)
    vmf.add_argument("--vmf_kappa_max", type=float, default=None)
    vmf.add_argument("--vmf_dual_scale", type=_parse_bool, default=None)
    vmf.add_argument("--vmf_asymmetric_qk", type=_parse_bool, default=None)
    vmf.add_argument("--vmf_entropy_reg", type=float, default=None)

    # Temporal & RoPE
    temp = p.add_argument_group("Temporal & RoPE")
    temp.add_argument("--use_temporal_transition", type=_parse_bool, default=None)
    temp.add_argument("--rope_base", type=float, default=None)
    temp.add_argument("--rope_max_seq", type=int, default=None)

    # Dictionary pre-training
    dpre = p.add_argument_group("Dictionary Pre-Training")
    dpre.add_argument("--dict_epochs", type=int, default=None)
    dpre.add_argument("--dict_lambda", type=float, default=None)
    dpre.add_argument("--dict_max_defs", type=int, default=None)
    dpre.add_argument("--dict_batch_size", type=int, default=None)
    dpre.add_argument("--dict_pos_weight", type=float, default=None)

    # Fine-tuning
    ft = p.add_argument_group("Fine-Tuning")
    ft.add_argument("--epochs", type=int, default=None)
    ft.add_argument("--batch_size", type=int, default=None)
    ft.add_argument("--grad_accum", type=int, default=None)
    ft.add_argument("--base_lr_max", type=float, default=None)
    ft.add_argument("--wd", type=float, default=None)
    ft.add_argument("--warmup_frac", type=float, default=None)
    ft.add_argument("--min_lr_frac", type=float, default=None)
    ft.add_argument("--ema_decay", type=float, default=None)
    ft.add_argument("--label_smooth_base", type=float, default=None)
    ft.add_argument("--mixup_alpha", type=float, default=None)
    ft.add_argument("--focal_gamma", type=float, default=None)

    # Regularisation
    reg = p.add_argument_group("Regularisation")
    reg.add_argument("--token_dropout_prob", type=float, default=None)
    reg.add_argument("--token_dropout_apply", type=float, default=None)
    reg.add_argument("--awp_eps", type=float, default=None)
    reg.add_argument("--awp_start_ep", type=int, default=None)

    # Misc
    misc = p.add_argument_group("Misc")
    misc.add_argument("--seed", type=int, default=None)
    misc.add_argument("--use_quantum_lr", type=_parse_bool, default=None)
    misc.add_argument("--checkpoint_dir", type=str, default=None)
    misc.add_argument("--tokenizer_vocab_size", type=int, default=None)
    misc.add_argument("--max_train", type=int, default=None)
    misc.add_argument("--max_val", type=int, default=None)
    misc.add_argument("--device_str", type=str, default=None)

    return p


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # Start from defaults
    overrides = {}

    # Load JSON config if provided
    if args.config:
        with open(args.config) as f:
            file_cfg = json.load(f)
        overrides.update(file_cfg)

    # CLI overrides (skip None values)
    for k, v in vars(args).items():
        if v is not None and k not in ("config", "export_config"):
            overrides[k] = v

    cfg = CFG.from_dict(overrides)

    # Export config if requested
    if args.export_config:
        with open(args.export_config, "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)
        print(f"Config exported to {args.export_config}")
        sys.exit(0)

    # Print effective config summary
    print(f"\n{'='*70}")
    print(f"  GrafoPropagation {cfg.VERSION}")
    print(f"  Parameters (est.): {cfg.count_parameters():,}")
    print(f"  Device: {cfg.device}  |  AMP: {cfg.amp_dtype}")
    print(f"  Architecture: d_model={cfg.d_model}  n_layers={cfg.n_layers}  "
          f"n_heads={cfg.n_heads}  head_dim={cfg.head_dim}")
    print(f"  Dict pre-train: {cfg.dict_epochs} epochs")
    print(f"  Fine-tune: {cfg.epochs} epochs × batch {cfg.batch_size}")
    print(f"{'='*70}\n")

    result = run_training(cfg)
    print(f"\nTraining complete. Best val acc: {result['best_val_acc']*100:.4f}%")


if __name__ == "__main__":
    main()
