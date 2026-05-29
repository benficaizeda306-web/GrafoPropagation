"""
GrafoPropagation v26-APEX — Loss Functions
FocalCrossEntropy · TokenDropout

(c) 2025-2026 Claudio Fernandes. All rights reserved.
"""

import random
import torch
import torch.nn.functional as F


def focal_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Focal cross-entropy loss with optional label smoothing.

    Parameters
    ----------
    logits          : (B, C) raw logits
    targets         : (B,)  integer class indices
    gamma           : focusing exponent (0 = standard CE)
    label_smoothing : ε for uniform smoothing

    Returns
    -------
    scalar loss
    """
    C = logits.size(-1)
    logp = F.log_softmax(logits, -1)

    if label_smoothing > 0.0:
        s = torch.full_like(logp, label_smoothing / (C - 1))
        s.scatter_(-1, targets.unsqueeze(-1), 1.0 - label_smoothing)
        ce = -(s * logp).sum(-1)
    else:
        ce = F.nll_loss(logp, targets, reduction="none")

    if gamma == 0.0:
        return ce.mean()
    return ((1.0 - torch.exp(-ce.detach())).pow(gamma) * ce).mean()


def token_dropout(
    ids: torch.Tensor,
    fmask: torch.Tensor,
    unk_id: int,
    token_prob: float = 0.10,
    apply_prob: float = 0.20,
) -> torch.Tensor:
    """
    Stochastic token masking (applied with probability `apply_prob`).
    Each non-CLS, non-PAD token is replaced by [UNK] with probability
    `token_prob`, providing a mild data-augmentation / regularisation.

    Parameters
    ----------
    ids        : (B, T) input token ids
    fmask      : (B, T) float attention mask
    unk_id     : vocabulary index of the [UNK] token
    token_prob : per-token replacement probability when mask is applied
    apply_prob : probability of applying any masking in a given batch

    Returns
    -------
    (B, T) augmented token ids
    """
    if random.random() > apply_prob:
        return ids
    pad_mask = fmask == float("-inf")
    cls_mask = torch.zeros_like(pad_mask)
    cls_mask[:, 0] = True
    drop = (torch.rand_like(ids.float()) < token_prob) & (~pad_mask) & (~cls_mask)
    out = ids.clone()
    out[drop] = unk_id
    return out
