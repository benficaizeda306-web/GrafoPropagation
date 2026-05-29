"""
GrafoPropagation v26-APEX — Dataset & WordNet Utilities
TextDataset · DictionaryDataset · collate_dict · build_wordnet_multilabel

(c) 2025-2026 Claudio Fernandes. All rights reserved.
"""

import torch
from torch.utils.data import Dataset

import nltk
nltk.download("wordnet", quiet=True)
from nltk.corpus import wordnet as wn

from .logging_utils import log


# ─────────────────────────────────────────────────────────────────────────────
# AG News Text Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """
    HuggingFace dataset → tokenised, padded AG News samples.
    All samples are pre-tokenised at construction time for fast DataLoader
    iteration (no per-worker tokenisation overhead).
    """

    def __init__(self, hf_ds, tokenizer, cfg):
        pad_id = tokenizer.token_to_id(cfg.pad_token)
        self.samples = []
        for item in hf_ds:
            ids = tokenizer.encode(str(item["text"])[:4096]).ids[:cfg.max_len]
            vl = len(ids)
            inp = ids + [pad_id] * (cfg.max_len - vl)
            fm = [0.0] * vl + [float("-inf")] * (cfg.max_len - vl)
            self.samples.append((
                torch.tensor(inp, dtype=torch.long),
                torch.tensor(fm, dtype=torch.float32),
                torch.tensor(int(item["label"]), dtype=torch.long),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


# ─────────────────────────────────────────────────────────────────────────────
# Dictionary Dataset (WordNet pre-training)
# ─────────────────────────────────────────────────────────────────────────────

class DictionaryDataset(Dataset):
    """
    (token_id, multi_label_vector) pairs built from WordNet definitions.
    label[j] = 1.0  iff token j appears in any definition of the input word.
    """

    def __init__(self, mapping: dict):
        self.items = list(mapping.items())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        tid, label_vec = self.items[idx]
        return torch.tensor(tid, dtype=torch.long), label_vec


def collate_dict(batch, pad_id: int, cls_id: int):
    """
    Collate dictionary samples into padded mini-batch tensors.
    Sequence layout: [CLS] [WORD] [PAD] [PAD]  (seq_len=4)
    Minimum length 4 prevents Conv1D and the Riemannian Log-Map from
    collapsing on single-token inputs.
    """
    seq_len = 4
    ids_batch = torch.full((len(batch), seq_len), pad_id, dtype=torch.long)
    fm_batch = torch.full((len(batch), seq_len), float("-inf"), dtype=torch.float32)

    for i, (tid, _) in enumerate(batch):
        ids_batch[i, 0] = cls_id
        fm_batch[i, 0] = 0.0
        ids_batch[i, 1] = tid
        fm_batch[i, 1] = 0.0

    labels_batch = torch.stack([item[1] for item in batch], dim=0)
    return ids_batch, fm_batch, labels_batch


def build_wordnet_multilabel(tokenizer, cfg) -> dict:
    """
    Construct {token_id → multi-label-tensor} from WordNet definitions.

    For each vocabulary token whose surface form maps to >=1 WordNet synset,
    the label vector is 1 for every token that appears in up to
    `cfg.dict_max_defs` definitions.

    Parameters
    ----------
    tokenizer : trained HuggingFace Tokenizer
    cfg       : CFG instance

    Returns
    -------
    dict  {int: torch.FloatTensor(vocab_size)}
    """
    log("Building WordNet multi-label mapping …", "INFO")
    vocab = tokenizer.get_vocab()
    id2word = {v: k for k, v in vocab.items()}
    V = cfg.tokenizer_vocab_size
    mapping = {}
    pad_id = tokenizer.token_to_id(cfg.pad_token)
    skip_set = {"[pad]", "[cls]", "[unk]", ".", ",", "?", "!"}

    for tid in range(V):
        word = id2word.get(tid, "")
        word_clean = word.replace("Ġ", "").strip().lower()
        if not word_clean or word_clean in skip_set:
            continue

        synsets = wn.synsets(word_clean)
        if not synsets:
            continue

        def_tokens = set()
        for syn in synsets[:cfg.dict_max_defs]:
            encoded = tokenizer.encode(syn.definition())
            for tok_id in encoded.ids:
                if tok_id != pad_id:
                    def_tokens.add(tok_id)

        if def_tokens:
            label_vec = torch.zeros(V, dtype=torch.float32)
            for dt in def_tokens:
                label_vec[dt] = 1.0
            mapping[tid] = label_vec

    log(f"  WordNet: {len(mapping)} tokens with valid definitions", "INFO")
    return mapping
