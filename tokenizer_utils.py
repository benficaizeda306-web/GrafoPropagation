"""
GrafoPropagation v26-APEX — Tokenizer
BPE tokenizer build / load with CLS post-processor.

(c) 2025-2026 Claudio Fernandes. All rights reserved.
"""

import os
from tokenizers import (
    Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers,
)
from tokenizers.processors import TemplateProcessing

from .logging_utils import log


def build_or_load_tokenizer(texts, cfg) -> Tokenizer:
    """
    Load a BPE tokenizer from `cfg.tokenizer_path` if it exists and has
    sufficient vocab size; otherwise train a new one from `texts`.

    Parameters
    ----------
    texts : iterable of str  (training corpus)
    cfg   : CFG instance

    Returns
    -------
    Tokenizer
    """
    if os.path.exists(cfg.tokenizer_path):
        tok = Tokenizer.from_file(cfg.tokenizer_path)
        if tok.get_vocab_size() >= cfg.tokenizer_vocab_size - 200:
            log(f"Tokenizer loaded from {cfg.tokenizer_path} "
                f"({tok.get_vocab_size()} tokens)")
            return tok

    log("Training new BPE tokenizer …", "INFO")
    tok = Tokenizer(models.BPE(unk_token=cfg.unk_token))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.normalizer = normalizers.NFKC()

    special_tokens = [cfg.pad_token, cfg.cls_token, cfg.unk_token]
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.tokenizer_vocab_size,
        special_tokens=special_tokens,
        min_frequency=1,
    )
    tok.train_from_iterator(texts, trainer=trainer)

    cls_id = tok.token_to_id(cfg.cls_token)
    tok.post_processor = TemplateProcessing(
        single=f"{cfg.cls_token} $A",
        special_tokens=[(cfg.cls_token, cls_id)],
    )
    tok.save(cfg.tokenizer_path)
    log(f"Tokenizer saved → {cfg.tokenizer_path}")
    return tok
