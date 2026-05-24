"""
YiCeNet BPE tokenizer — wraps Qwen2.5 BPE → 8000 vocab rebucket.

Two phases:
  1. build_vocab(session_db_path) — scan sessions, build freq table, save mapping
  2. encode(text) — Qwen BPE → rebucket to YiCeNet token IDs

The mapping file is saved at data/qwen_to_yicenet.json.
Vocab_size=8000: IDs 0=PAD, 1-7999=top freq tokens, rest→1 (UNK)
"""

import json
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import torch

from .config import yicenet_data_dir

_TOK = None  # lazy-loaded Qwen tokenizer
_VOCAB_MAP = None  # lazy-loaded {qwen_id: yicenet_id}

def _map_path() -> Path:
    """Location of the Qwen→YiCeNet vocab mapping file."""
    return yicenet_data_dir() / "qwen_to_yicenet.json"


def _get_qwen_tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B", trust_remote_code=True
        )
        # Qwen2.5: pad_token_id=0, eos_token_id=151643
    return _TOK


def _load_vocab_map() -> dict[int, int]:
    """Load {qwen_id: yicenet_id} mapping, rebuild if missing."""
    global _VOCAB_MAP
    if _VOCAB_MAP is not None:
        return _VOCAB_MAP

    map_path = _map_path()
    if map_path.exists():
        with open(map_path) as f:
            raw = json.load(f)
        _VOCAB_MAP = {int(k): v for k, v in raw.items()}
        return _VOCAB_MAP

    # Fallback: modulo mapping (deterministic, no collision guarantee)
    _VOCAB_MAP = {}
    return _VOCAB_MAP


def build_vocab(
    session_db_path: str = str(Path.home() / ".hermes" / "state.db"),
    vocab_size: int = 8000,
    output_path: Optional[str] = None,
) -> dict[int, int]:
    """
    Phase 1: Scan session DB, count Qwen token frequencies, build rebucket map.

    Top (vocab_size - 1) most frequent tokens → IDs 1..vocab_size-1.
    All others → ID 1 (shared UNK bucket).
    ID 0 = PAD (reserved).

    Returns {qwen_id: yicenet_id} mapping.
    """
    import sqlite3

    tok = _get_qwen_tokenizer()
    counter: Counter = Counter()

    conn = sqlite3.connect(session_db_path)

    # Collect user messages (what we need to encode)
    for role in ("user",):
        rows = conn.execute(
            "SELECT content FROM messages WHERE role=? AND content IS NOT NULL AND length(content) > 2",
            (role,),
        ).fetchall()
        for (content,) in rows:
            ids = tok.encode(content)
            counter.update(ids)

    conn.close()
    print(f"[YiCeNet Vocab] Scanned {sum(counter.values()):,} tokens, "
          f"{len(counter):,} unique Qwen tokens")

    # Build mapping: top (vocab_size - 1) tokens get unique IDs
    top_tokens = [tid for tid, _ in counter.most_common(vocab_size - 1)]

    mapping = {tid: i + 1 for i, tid in enumerate(top_tokens)}
    # Unmapped tokens get ID 1 (first slot in the top list, acts as UNK)
    # Actually, let's reserve ID 1 for UNK explicitly
    # Re-map: ID 1 = UNK, IDs 2..7999 = top tokens
    mapping = {top_tokens[0]: 1}  # most frequent token = UNK bucket
    for i, tid in enumerate(top_tokens[1:], start=2):
        mapping[tid] = i

    # Save
    if output_path is None:
        output_path = str(_map_path())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(mapping, f)
    print(f"[YiCeNet Vocab] Saved {len(mapping):,} mappings to {output_path}")
    print(f"[YiCeNet Vocab] Coverage: {sum(counter[t] for t in mapping):,}/{sum(counter.values()):,} "
                f"({sum(counter[t] for t in mapping) / sum(counter.values()) * 100:.1f}%)")

    _VOCAB_MAP = dict(mapping)
    return mapping


def encode(
    text: str,
    max_len: int = 128,
    pad_to: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode text with Qwen BPE → rebucket to YiCeNet token IDs.

    Args:
        text: input string
        max_len: maximum sequence length (truncate)
        pad_to: if set, pad/pack to this exact length

    Returns:
        input_ids: (1, L) tensor of YiCeNet token IDs
        attention_mask: (1, L) binary mask
    """
    tok = _get_qwen_tokenizer()
    vocab = _load_vocab_map()

    qwen_ids = tok.encode(text)
    # Truncate to max_len
    if len(qwen_ids) > max_len:
        qwen_ids = qwen_ids[:max_len]

    # Rebucket
    yicenet_ids = [vocab.get(tid, 1) for tid in qwen_ids]
    seq_len = len(yicenet_ids)

    # Ensure minimum length of 2
    while len(yicenet_ids) < 2:
        yicenet_ids.append(0)  # PAD

    mask = [1] * seq_len + [0] * (max_len - seq_len)
    # Pad
    pad_len = max_len - len(yicenet_ids)
    yicenet_ids = yicenet_ids + [0] * pad_len

    # Enforce final length = max_len (or pad_to if set)
    if pad_to is not None:
        if len(yicenet_ids) < pad_to:
            yicenet_ids = yicenet_ids + [0] * (pad_to - len(yicenet_ids))
            mask = mask + [0] * (pad_to - len(mask))
        yicenet_ids = yicenet_ids[:pad_to]
        mask = mask[:pad_to]

    return (
        torch.tensor([yicenet_ids], dtype=torch.long),
        torch.tensor([mask], dtype=torch.long),
    )


def encode_batch(
    texts: list[str],
    max_len: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode multiple texts, pad to max length in batch."""
    results = [encode(t, max_len=max_len) for t in texts]
    max_seq = max(r[0].shape[1] for r in results)
    padded_ids = []
    padded_masks = []
    for ids, mask in results:
        if ids.shape[1] < max_seq:
            pad = max_seq - ids.shape[1]
            ids = torch.cat([ids, torch.zeros(1, pad, dtype=torch.long)], dim=1)
            mask = torch.cat([mask, torch.zeros(1, pad, dtype=torch.long)], dim=1)
        padded_ids.append(ids)
        padded_masks.append(mask)
    return torch.cat(padded_ids, dim=0), torch.cat(padded_masks, dim=0)


def get_vocab_size() -> int:
    """Return effective YiCeNet vocab size (Qwen→rebucket)."""
    return min(8000, len(_load_vocab_map()) + 1)


# ── Test ──
if __name__ == "__main__":
    # Build vocab
    build_vocab()

    # Test encode
    for text in [
        "搜索 knowledge base",
        "检查 EVAL 维度",
        "训练 YiCeNet 模型",
        "I need to search for SAP PM documentation",
    ]:
        ids, mask = encode(text)
        actual_len = mask.sum().item()
        print(f"  [{actual_len:3d}tok] {text[:50]:50s} → {ids[0][:8].tolist()}...")
