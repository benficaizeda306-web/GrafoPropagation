# GrafoPropagation v26-APEX

**Status:** Proprietary / Confidential  
**Author:** Claudio Fernandes  
**License:** Proprietary — see [LICENSE](LICENSE)  
**📖 Paper on Zenodo:** https://zenodo.org/records/20446506

## Overview

GrafoPropagation is a compact (~990k parameter) text classification architecture built on **geometric von Mises-Fisher (vMF) attention** with **WordNet dictionary pre-training** and **AG News fine-tuning**.

For detailed methodology, cite: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20446506.svg)](https://doi.org/10.5281/zenodo.20446506)

### Key Innovations

| Component | Description |
|---|---|
| **vMF Dual-Scale Attention** | Queries/keys on the unit hypersphere with learnable concentration κ (local × global) |
| **Asymmetric Q/K** | Separate projections for queries and keys for improved expressivity |
| **Riemannian Temporal Embedding** | Log-Map + Parallel Transport on the sphere for position-aware dynamics |
| **RoPE** | Rotary Position Embedding on normalised direction vectors |
| **Dynamic GrafoConnect** | Learned cross-layer skip-connection graph modulated by curvature |
| **Global Workspace Memory** | Learnable broadcast slots (Global Workspace Theory) |
| **System-2 Latent Search** | Iterative branch-evaluate-merge with GumbelMCTS |
| **Dictionary Pre-Training** | Multi-label BCE over WordNet definitions |
| **Quantum LR Modulation** | 8-qubit PennyLane circuit for epoch-dependent LR scaling |

## Installation

```bash
pip install .
```

This automatically installs all dependencies: PyTorch, PennyLane, HuggingFace datasets/tokenizers, NLTK, Rich.

For GPU-accelerated quantum simulation:
```bash
pip install ".[gpu]"
```

## Quick Start

### Python API

```python
from grafopropagation import CFG, GrafoPropagation, run_training, build_or_load_tokenizer

# Default configuration (~990k params)
cfg = CFG()
result = run_training(cfg)

# Scale up the architecture
cfg = CFG(d_model=128, n_layers=4, n_heads=8, head_dim=32, d_ff=640)
result = run_training(cfg)

# Full control via dict
cfg = CFG.from_dict({
    "d_model": 256,
    "n_layers": 6,
    "n_heads": 8,
    "head_dim": 32,
    "d_ff": 1024,
    "dict_epochs": 500,
    "epochs": 50,
})
print(f"Estimated parameters: {cfg.count_parameters():,}")
result = run_training(cfg)
```

### CLI

```bash
# Default training
grafoprop-train

# Scale up architecture
grafoprop-train --d_model 128 --n_layers 4 --n_heads 8 --head_dim 32

# Custom training schedule
grafoprop-train --epochs 50 --batch_size 128 --base_lr_max 0.002

# Export/load config
grafoprop-train --export_config my_config.json
grafoprop-train --config my_config.json

# Disable quantum LR modulation
grafoprop-train --use_quantum_lr false
```

## Configuration

All parameters are exposed in the `CFG` dataclass. Key scaling dimensions:

| Parameter | Default | Description |
|---|---|---|
| `d_model` | 64 | Model hidden dimension |
| `n_layers` | 2 | Transformer layers |
| `n_heads` | 4 | Attention heads |
| `head_dim` | 16 | Per-head dimension |
| `d_ff` | 320 | Feed-forward inner dim |
| `K_think` | 2 | System-2 thought tokens |
| `memory_slots` | 6 | Global Workspace slots |
| `latent_actions` | 6 | World model actions |
| `mcts_simulations` | 12 | MCTS simulations per step |
| `dict_epochs` | 1500 | WordNet pre-training epochs |
| `epochs` | 30 | Fine-tuning epochs |

### Scaling Recipes

| Target | Config |
|---|---|
| ~990k (default) | `CFG()` |
| ~3M | `CFG(d_model=96, n_layers=3, n_heads=6, head_dim=24, d_ff=512)` |
| ~7M | `CFG(d_model=128, n_layers=4, n_heads=8, head_dim=32, d_ff=640)` |
| ~15M | `CFG(d_model=192, n_layers=6, n_heads=8, head_dim=48, d_ff=1024)` |
| ~30M | `CFG(d_model=256, n_layers=8, n_heads=8, head_dim=64, d_ff=1280)` |

## Training Results (Default ~990k)

| Epoch | Train Acc | Val Acc |
|---|---|---|
| 1 | 67.2% | 25.6% |
| 3 | 90.6% | 84.4% |
| 5 | 92.6% | 91.1% |
| 10 | 94.9% | 92.7% |
| 13 | 95.9% | **93.1%** |
| 15 | 96.4% | 93.1% |

Pre-training: 1500 epochs on WordNet definitions, final dict loss: 0.04481

## Architecture

```
Input IDs
  │
  ├── Character-Seeded Token Embedding (Fibonacci sphere init)
  │
  ├── Temporal Transition Embedding (Log-Map + Parallel Transport)
  │
  ├── Local Depthwise Conv Mixer
  │
  ├── Global Workspace Memory (prepend slots)
  │
  ├── ×N TransformerBlock
  │   ├── vMF Dual-Scale Attention (RoPE + κ gating)
  │   └── SwiGLU FFN (stochastic depth)
  │
  │   └── Dynamic GrafoConnect (cross-layer skips)
  │
  ├── System-2 Latent Search
  │   ├── Branch-Evaluate-Merge iterations
  │   └── Gumbel MCTS (soft search / hard search)
  │
  ├── Pooling Fusion (gated think + seq_avg)
  │
  ├── Multi-Sample Dropout Head → class logits
  │
  └── Multi-Label Head → dict logits (pre-training)
```

## File Structure

```
grafopropagation/
├── __init__.py           # Package exports
├── config.py             # CFG dataclass (fully configurable)
├── primitives.py         # RMSNorm, trunc_normal, char embeddings
├── positional.py         # RoPE, TemporalTransitionEmbedding
├── attention.py          # VonMisesFisherAttention
├── transformer.py        # LocalConvMix, SwiGLU, TransformerBlock
├── memory.py             # GlobalWorkspaceMemory, DynamicGrafoConnect
├── system2.py            # WorldModel, PolicyValueHead, GumbelMCTS, System2LatentSearch
├── heads.py              # PoolingFusion, MultiSampleDropoutHead, MultiLabelHead
├── model.py              # GrafoPropagation (main model)
├── datasets.py           # TextDataset, DictionaryDataset, WordNet builder
├── tokenizer_utils.py    # BPE tokenizer build/load
├── losses.py             # focal_ce, token_dropout
├── optimizer.py          # EMA, Lookahead, AWP, GC, WarmupCosineLR
├── quantum.py            # Quantum LR modulation (PennyLane)
├── logging_utils.py      # Rich-based logging
├── train.py              # Full training pipeline
├── cli.py                # CLI entry point
└── py.typed              # PEP 561 marker
```

## Dependencies

- Python >= 3.9
- PyTorch >= 2.0
- PennyLane >= 0.33
- HuggingFace datasets >= 2.14
- HuggingFace tokenizers >= 0.14
- NLTK >= 3.8
- Rich >= 13.0
- NumPy >= 1.24
