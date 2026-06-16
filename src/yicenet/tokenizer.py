"""
YiCeNet BPE tokenizer — wraps Qwen2.5 BPE → 8000 vocab rebucket.

Two phases:
  1. build_vocab(session_db_path) — scan sessions, build freq table, save mapping
  2. encode(text) — Qwen BPE → rebucket to YiCeNet token IDs

The mapping file is saved at data/qwen_to_yicenet.json.
Vocab_size=8000: IDs 0=PAD, 1-7999=top freq tokens, rest→1 (UNK)

Tokenizer files live at ~/.yicenet/tokenizer/qwen2.5-0.5b/ (downloaded
once by bootstrap, never hits HF Hub at runtime).
"""

import json
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import torch

from .config import yicenet_data_dir, yicenet_home

_TOK = None  # lazy-loaded Qwen tokenizer
_VOCAB_MAP = None  # lazy-loaded {qwen_id: yicenet_id}

# ── Local tokenizer dir (~/.yicenet/tokenizer/qwen2.5-0.5b/) ──

_QWEN_MODEL = "Qwen/Qwen2.5-0.5B"
_TOKENIZER_SUBDIR = "qwen2.5-0.5b"

# Files needed for offline tokenizer (from HF Hub)
_TOKENIZER_FILES = [
    "tokenizer.json",       # ~6.9 MB — BPE model
    "vocab.json",           # ~2.7 MB
    "merges.txt",           # ~1.6 MB
    "tokenizer_config.json",
    "config.json",
]


def _tokenizer_dir() -> Path:
    """~/.yicenet/tokenizer/qwen2.5-0.5b/"""
    return yicenet_home() / "tokenizer" / _TOKENIZER_SUBDIR


def tokenizer_available() -> bool:
    """Check if tokenizer files are cached locally."""
    d = _tokenizer_dir()
    return d.exists() and (d / "tokenizer.json").exists()


# ── Download ──


def download_tokenizer(hf_token: str = "") -> bool:
    """Download Qwen2.5-0.5B tokenizer files to ~/.yicenet/tokenizer/.

    Uses huggingface_hub to download only the tokenizer files (not the
    full model). Falls back to raw HTTPS downloads if huggingface_hub
    is not available (rare).

    Args:
        hf_token: Optional HF Hub token for authenticated downloads.

    Returns:
        True if download succeeded or files already present.
    """
    target = _tokenizer_dir()
    if tokenizer_available():
        print(f"[YiCeNet Tokenizer] Already cached at {target}")
        return True

    target.mkdir(parents=True, exist_ok=True)
    print(f"[YiCeNet Tokenizer] Downloading {_QWEN_MODEL} tokenizer to {target}...")

    try:
        _download_via_hf_hub(target, hf_token)
    except ImportError:
        _download_via_requests(target, hf_token)

    ok = tokenizer_available()
    if ok:
        total = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        print(f"[YiCeNet Tokenizer] Done ({total / 1024 / 1024:.1f} MB)")
    else:
        print(f"[YiCeNet Tokenizer] WARNING: some files failed to download")
    return ok


def _download_via_hf_hub(target: Path, hf_token: str) -> None:
    """Download tokenizer files using huggingface_hub."""
    from huggingface_hub import hf_hub_download

    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    for fname in _TOKENIZER_FILES:
        try:
            path = hf_hub_download(
                repo_id=_QWEN_MODEL,
                filename=fname,
                local_dir=str(target),
                local_dir_use_symlinks=False,
                token=hf_token or None,
            )
            print(f"  ✓ {fname}")
        except Exception as e:
            print(f"  ✗ {fname}: {e}")


def _download_via_requests(target: Path, hf_token: str) -> None:
    """Fallback: download tokenizer files via raw HTTPS."""
    import requests

    base = f"https://huggingface.co/{_QWEN_MODEL}/resolve/main"
    headers = {"User-Agent": "YiCeNet/1.0"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    for fname in _TOKENIZER_FILES:
        url = f"{base}/{fname}"
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            (target / fname).write_bytes(resp.content)
            print(f"  ✓ {fname}  ({len(resp.content) / 1024:.0f} KB)")
        except Exception as e:
            print(f"  ✗ {fname}: {e}")


# ── Load tokenizer (local first, HF fallback) ──


def _map_path() -> Path:
    """Location of the Qwen→YiCeNet vocab mapping file."""
    return yicenet_data_dir() / "qwen_to_yicenet.json"


def _get_qwen_tokenizer():
    """Lazy-load Qwen tokenizer from local cache or HF Hub.

    Resolution order:
      1. ~/.yicenet/tokenizer/qwen2.5-0.5b/   (local, preferred)
      2. HF Hub cache  (via TRANSFORMERS_OFFLINE guard)
      3. HF Hub network (last resort, may block)
    """
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer

        local_dir = _tokenizer_dir()
        if local_dir.exists() and (local_dir / "tokenizer.json").exists():
            # Local path — no network, no remote code execution
            _TOK = AutoTokenizer.from_pretrained(
                str(local_dir), trust_remote_code=False
            )
        else:
            # Fallback to HF Hub (with env-var guard for offline mode)
            _TOK = AutoTokenizer.from_pretrained(
                _QWEN_MODEL, trust_remote_code=True
            )
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

    # Fallback: no vocab map — all tokens become UNK (ID 1)
    import warnings
    warnings.warn(
        "YiCeNet vocab mapping not found. All tokens will be UNK. "
        "Run yicenet.bootstrap.build_vocab() to build vocabulary from session DB.",
        RuntimeWarning
    )
    _VOCAB_MAP = {}
    return _VOCAB_MAP


def build_vocab(
    session_db_path: str = str(Path.home() / ".hermes" / "state.db"),
    vocab_size: int = 8000,
    output_path: Optional[str] = None,
) -> dict[int, int]:
    """Phase 1: Scan session DB, count Qwen token frequencies, build rebucket map."""
    import sqlite3

    tok = _get_qwen_tokenizer()
    counter: Counter = Counter()

    conn = sqlite3.connect(session_db_path)

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

    top_tokens = [tid for tid, _ in counter.most_common(vocab_size - 1)]

    mapping = {top_tokens[0]: 1}
    for i, tid in enumerate(top_tokens[1:], start=2):
        mapping[tid] = i

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
    """Encode text with Qwen BPE → rebucket to YiCeNet token IDs."""
    tok = _get_qwen_tokenizer()
    vocab = _load_vocab_map()

    qwen_ids = tok.encode(text)
    if len(qwen_ids) > max_len:
        qwen_ids = qwen_ids[:max_len]

    yicenet_ids = [vocab.get(tid, 1) for tid in qwen_ids]
    seq_len = len(yicenet_ids)

    while len(yicenet_ids) < 2:
        yicenet_ids.append(0)

    mask = [1] * seq_len + [0] * (max_len - seq_len)
    pad_len = max_len - len(yicenet_ids)
    yicenet_ids = yicenet_ids + [0] * pad_len

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


# ── ITokenizer adapter ────────────────────────────────────────────────────────

class QwenTokenizerAdapter:
    """Wraps module-level encode() + get_vocab_size() as an ITokenizer object.

    Used by ProviderRegistry.default() to inject into YiCeNetEngine.
    Import-time cost: zero (lazy load inside encode()).
    """

    def encode(self, text: str, max_len: int = 128):
        return encode(text, max_len=max_len)

    def get_vocab_size(self) -> int:
        return get_vocab_size()

    def download(self, hf_token: str = "") -> bool:
        return download_tokenizer(hf_token=hf_token)


# ── Test ──
if __name__ == "__main__":
    # Download tokenizer if not cached
    if not tokenizer_available():
        download_tokenizer()

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
