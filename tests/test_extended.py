"""
Extended tests for YiCeNet — tokenizer, world model, config, engine.
"""
import sys
import os
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np
from pathlib import Path


# ═══════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════

def test_yicenet_home_auto_detect():
    """yicenet_home() should auto-detect from source tree."""
    from yicenet.config import yicenet_home
    home = yicenet_home()
    assert home.exists(), f"yicenet_home does not exist: {home}"
    assert (home / "pyproject.toml").exists(), f"pyproject.toml not found at {home}"


def test_yicenet_config_defaults():
    """YiCeNetConfig should have sensible defaults."""
    from yicenet.config import YiCeNetConfig
    cfg = YiCeNetConfig()
    assert cfg.vocab_size == 8000
    assert cfg.hidden_dim == 256
    assert cfg.num_hexagrams == 64
    assert cfg.num_actions == 50
    assert len(cfg.hexagram_patterns) == 64


# ═══════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════

def test_precomputed_tables():
    """PrecomputedHexagramTables should compute all derived tensors correctly."""
    from yicenet.constants import PrecomputedHexagramTables
    tables = PrecomputedHexagramTables.get_instance()

    # hex_bits: (64, 6)
    assert tables.hex_bits.shape == (64, 6)
    assert tables.hex_bits.dtype == torch.int64

    # All hexagrams should have valid bits (0 or 1)
    assert ((tables.hex_bits == 0) | (tables.hex_bits == 1)).all()

    # upper_trigrams / lower_trigrams: (64,) in range [0, 7]
    assert tables.upper_trigrams.shape == (64,)
    assert tables.lower_trigrams.shape == (64,)
    assert (tables.upper_trigrams >= 0).all() and (tables.upper_trigrams < 8).all()
    assert (tables.lower_trigrams >= 0).all() and (tables.lower_trigrams < 8).all()

    # hamming_matrix: (64, 64), diagonal = 0
    assert tables.hamming_matrix.shape == (64, 64)
    assert tables.hamming_matrix.diag().sum() == 0.0

    # opposite_upper: (64,)
    assert tables.opposite_upper.shape == (64,)


# ═══════════════════════════════════════════════════════════
# Tokenizer
# ═══════════════════════════════════════════════════════════

def test_tokenizer_vocab_map_warning():
    """Tokenizer should warn when vocab map is missing."""
    from yicenet.tokenizer import _load_vocab_map, _VOCAB_MAP

    # Temporarily clear cache to force reload
    import yicenet.tokenizer as tok
    saved = tok._VOCAB_MAP
    tok._VOCAB_MAP = None

    # Check if vocab file exists
    from yicenet.config import yicenet_data_dir
    map_path = yicenet_data_dir() / "qwen_to_yicenet.json"

    if not map_path.exists():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _load_vocab_map()
            assert len(w) >= 1, "Should warn when vocab map missing"
            assert "UNK" in str(w[0].message)
    else:
        # Vocab exists — should load without warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mapping = _load_vocab_map()
            assert len(mapping) > 0, f"Vocab map should be non-empty (got {len(mapping)})"

    tok._VOCAB_MAP = saved


def test_tokenizer_encode_shape():
    """encode should return correctly shaped tensors."""
    from yicenet.tokenizer import encode
    ids, mask = encode("test message", max_len=32)

    assert ids.shape == (1, 32), f"Expected (1, 32), got {ids.shape}"
    assert mask.shape == (1, 32)
    assert ids.dtype == torch.long
    assert mask.dtype == torch.long

    # Mask should have 1s for tokens, 0s for padding
    n_tokens = mask.sum().item()
    assert n_tokens >= 2, f"Expected at least 2 tokens, got {n_tokens}"
    assert mask[0, :n_tokens].sum() == n_tokens
    assert mask[0, n_tokens:].sum() == 0


def test_tokenizer_pad_to():
    """encode with pad_to should produce exact length."""
    from yicenet.tokenizer import encode
    ids, mask = encode("hello", max_len=128, pad_to=64)
    assert ids.shape == (1, 64)


# ═══════════════════════════════════════════════════════════
# World Model
# ═══════════════════════════════════════════════════════════

def test_world_model_forward():
    """WorldModelV2 forward pass should return correct shapes."""
    from yicenet.world_model import WorldModelV2

    wm = WorldModelV2()
    B = 4
    probes = torch.randn(B, 9)
    hex_id = torch.randint(0, 64, (B,))

    hex_dist, ext_vec = wm(probes, hex_id)

    assert hex_dist.shape == (B, 64)
    assert ext_vec.shape == (B, 3)

    # hex_dist should be a valid probability distribution
    assert (hex_dist >= 0).all() and (hex_dist <= 1).all()
    for b in range(B):
        assert abs(hex_dist[b].sum().item() - 1.0) < 0.01

    # ext_vec should be in [0, 1] (sigmoid output)
    assert (ext_vec >= 0).all() and (ext_vec <= 1).all()


def test_power_law_weight():
    """power_law_weight should decay correctly."""
    from yicenet.world_model import power_law_weight

    now = 1_000_000_000.0

    # Recent: weight ~1
    w = power_law_weight(now - 1, now, tau_days=30, alpha=1.5)
    assert 0.99 < w <= 1.0, f"Recent sample weight {w} should be ~1"

    # Old (30 days): weight should be ~ (1+1)^(-1.5) = 2^(-1.5) ≈ 0.354
    w = power_law_weight(now - 30 * 86400, now, tau_days=30, alpha=1.5)
    assert 0.3 < w < 0.4, f"30-day weight {w} should be ~0.354"

    # Very old (300 days): weight should approach 0 but never reach it
    w = power_law_weight(now - 300 * 86400, now, tau_days=30, alpha=1.5)
    assert 0 < w < 0.05, f"300-day weight {w} should be small but > 0"


def test_world_model_endogenous_weight():
    """compute_endogenous_weight should return valid weights."""
    from yicenet.world_model import WorldModelV2

    wm = WorldModelV2()

    # Predictable case: use WM's own prediction as target → should be high weight
    probes = torch.randn(4, 9)
    hex_id = torch.randint(0, 64, (4,))
    with torch.no_grad():
        pred_dist, _ = wm(probes, hex_id)

    weights = wm.compute_endogenous_weight(probes, hex_id, pred_dist)
    assert weights.shape == (4,)
    assert (weights >= 0).all() and (weights <= 1).all()

    # Self-prediction should have high weight (low KL → high weight)
    # Average should be > 0.5
    assert weights.mean().item() > 0.5, f"Self-prediction weight {weights.mean().item():.3f} should be high"


def test_world_model_save_load():
    """WorldModelV2 save/load round-trip."""
    from yicenet.world_model import WorldModelV2
    import tempfile

    wm = WorldModelV2()
    probes = torch.randn(2, 9)
    hex_id = torch.tensor([10, 42])

    with torch.no_grad():
        orig_dist, orig_ext = wm(probes, hex_id)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name
    try:
        wm.save(tmp)
        wm2 = WorldModelV2.load(tmp)

        with torch.no_grad():
            loaded_dist, loaded_ext = wm2(probes, hex_id)

        assert torch.allclose(orig_dist, loaded_dist), "Hex distribution mismatch after load"
        assert torch.allclose(orig_ext, loaded_ext), "External vector mismatch after load"
    finally:
        os.unlink(tmp)


# ═══════════════════════════════════════════════════════════
# Probes
# ═══════════════════════════════════════════════════════════

def test_probe_extraction():
    """extract_probes_tensor should return ℝ⁹ tensor."""
    from yicenet.config import YiCeNetConfig
    from yicenet.model import YiCeNet
    from yicenet.probes import extract_probes_tensor

    config = YiCeNetConfig()
    model = YiCeNet(config)

    B = 2
    input_ids = torch.randint(1, config.vocab_size, (B, 8))
    mask = torch.ones(B, 8)

    with torch.no_grad():
        out = model(input_ids, mask, tau=0.5, hard=True)
        probes = extract_probes_tensor(
            h=out["h"],
            router_logits=out["router_logits"],
            router_probs=out["hexagram_probs"],
            candidate_values=out["candidate_values"],
            hexagram_idx=out["hexagram_idx"],
            prev_hexagram_idx=None,
            action_logits=out["action_logits"],
        )

    assert probes.shape == (9,), f"Expected (9,), got {probes.shape}"
    assert probes.dtype == torch.float32


# ═══════════════════════════════════════════════════════════
# External Metrics
# ═══════════════════════════════════════════════════════════

def test_compute_satisfaction():
    """Satisfaction scores should match expected patterns."""
    from yicenet.external_metrics import compute_satisfaction

    # Praise → 1.0
    assert compute_satisfaction("thanks, great work!", "task") == 1.0
    assert compute_satisfaction("很好，不錯！", "task") == 1.0

    # Correction → -1.0
    assert compute_satisfaction("no, that's wrong", "task") == -1.0
    assert compute_satisfaction("不對，你錯了", "task") == -1.0

    # Completion → 0.5
    assert compute_satisfaction("ok, done", "task") == 0.5

    # Normal continuation → 0.3
    assert compute_satisfaction("what about this?", "task") == 0.3

    # Abandoned (no follow-up, current not completion) → -0.5
    assert compute_satisfaction(None, "task") == -0.5

    # Completion with no follow-up → 0.5
    assert compute_satisfaction(None, "ok") == 0.5
    assert compute_satisfaction(None, "done") == 0.5


def test_estimate_token_cost():
    """Token cost should be normalized 0-1."""
    from yicenet.external_metrics import estimate_token_cost

    assert estimate_token_cost("") > 0
    assert estimate_token_cost("hello") < estimate_token_cost("x" * 2000)
    assert 0 <= estimate_token_cost("test") <= 1.0


# ═══════════════════════════════════════════════════════════
# Value Network
# ═══════════════════════════════════════════════════════════

def test_value_network_shapes():
    """ValueNetwork should accept (B, D) and (B, K, D) inputs."""
    from yicenet.value_net import ValueNetwork

    vn = ValueNetwork(256, 128)

    # (B, D) → (B, 1)
    x = torch.randn(4, 256)
    out = vn(x)
    assert out.shape == (4, 1)

    # (B, K, D) → (B, K, 1)
    x = torch.randn(4, 8, 256)
    out = vn(x)
    assert out.shape == (4, 8, 1)


# ═══════════════════════════════════════════════════════════
# Engine (needs checkpoint — smoke test only)
# ═══════════════════════════════════════════════════════════

def test_engine_singleton():
    """get_engine() should return singleton."""
    from yicenet.yicenet_engine import get_engine

    e1 = get_engine()
    e2 = get_engine()
    assert e1 is e2, "get_engine() should return the same instance"


def test_engine_device_resolution():
    """Device resolution should return a valid string."""
    from yicenet.yicenet_engine import YiCeNetEngine
    engine = YiCeNetEngine()
    dev = engine._resolve_device()
    assert dev in ("cuda", "cpu"), f"Unexpected device: {dev}"


# ═══════════════════════════════════════════════════════════
# RL Train utilities
# ═══════════════════════════════════════════════════════════

import numpy as np


def test_project_to_hexagram_space():
    """project_to_hexagram_space should return valid distribution."""
    from yicenet.rl_train import project_to_hexagram_space

    dist = project_to_hexagram_space({"continued": True})
    assert dist.shape == (64,)
    assert abs(dist.sum().item() - 1.0) < 0.01, f"Sum {dist.sum().item()} should be ~1"
    assert dist.max() > dist.min(), "Distribution should not be uniform"

    # Empty signals → uniform
    dist = project_to_hexagram_space({})
    assert torch.allclose(dist, torch.ones(64) / 64, atol=0.01)


def test_compute_hexagram_reward():
    """compute_hexagram_reward should be in [0, 1]."""
    from yicenet.rl_train import compute_hexagram_reward

    # Identical distributions → reward ~1
    dist = torch.ones(1, 64) / 64
    r = compute_hexagram_reward(dist, dist)
    assert 0.99 <= r.item() <= 1.01

    # Opposite one-hot distributions — cosine=0 → reward=0.5
    a = torch.zeros(1, 64)
    a[0, 0] = 1.0
    b = torch.zeros(1, 64)
    b[0, 63] = 1.0
    r = compute_hexagram_reward(a, b)
    assert abs(r.item() - 0.5) < 0.01, f"Opposite one-hot: cosine=0 → reward=0.5 (got {r.item()})"

    # Slightly different distributions → reward between 0.5 and 1
    c = torch.ones(1, 64) * 0.5 / 32
    c[0, :32] = 1.5 / 32
    d = torch.ones(1, 64) * 0.5 / 32
    d[0, 32:] = 1.5 / 32
    r2 = compute_hexagram_reward(c, d)
    assert 0.4 < r2.item() < 0.99, f"Different distributions reward should be <1 (got {r2.item()})"


# ═══════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════

def run_all():
    tests = [
        test_yicenet_home_auto_detect,
        test_yicenet_config_defaults,
        test_precomputed_tables,
        test_tokenizer_vocab_map_warning,
        test_tokenizer_encode_shape,
        test_tokenizer_pad_to,
        test_world_model_forward,
        test_power_law_weight,
        test_world_model_endogenous_weight,
        test_world_model_save_load,
        test_probe_extraction,
        test_compute_satisfaction,
        test_estimate_token_cost,
        test_value_network_shapes,
        test_engine_singleton,
        test_engine_device_resolution,
        test_project_to_hexagram_space,
        test_compute_hexagram_reward,
    ]

    print("=" * 60)
    print("YiCeNet Extended Test Suite")
    print("=" * 60)

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    print(f"{'=' * 60}")

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
