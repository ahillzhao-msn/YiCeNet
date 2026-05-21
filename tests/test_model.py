"""
Tests for YiCeNet components.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.config import YiCeNetConfig
from src.model import YiCeNet
from src.encoder import TinyEncoder
from src.hexagram import (
    hexagram_to_lines, lines_to_hexagram,
    cuo_hexagram, zong_hexagram, hu_hexagram,
    zhi_hexagram, generate_candidates,
)


def test_hexagram_conversion():
    """Test hexagram ↔ lines conversion."""
    config = YiCeNetConfig()
    
    for idx in range(64):
        idx_t = torch.tensor([idx])
        lines = hexagram_to_lines(idx_t)
        back = lines_to_hexagram(lines)
        assert back[0].item() == idx, f"Round-trip failed for hexagram {idx}"
    
    print("  ✓ hexagram_to_lines ↔ lines_to_hexagram round-trip")


def test_cuo_hexagram():
    """Test opposite hexagram (错卦)."""
    # 乾 (0b111111 = 63) ↔ 坤 (0b000000 = 0)
    qian = torch.tensor([63])
    kun = cuo_hexagram(qian)
    assert kun[0].item() == 0, f"错卦: 乾→坤 failed: {kun[0].item()}"

    # Double flip returns to original
    back = cuo_hexagram(kun)
    assert back[0].item() == 63, f"错卦 double flip failed"

    # Random test: cuo(cuo(x)) == x for all x
    for idx in range(64):
        x = torch.tensor([idx])
        assert cuo_hexagram(cuo_hexagram(x))[0].item() == idx
    
    print("  ✓ cuo_hexagram (错卦): opposite + idempotent double-flip")


def test_zong_hexagram():
    """Test upside-down hexagram (综卦)."""
    # Lines [top, ..., bottom] reversed
    # For a symmetric hexagram, zong(zong(x)) == x
    for idx in range(64):
        x = torch.tensor([idx])
        lines = hexagram_to_lines(x)
        # Apply zong twice → restore
        lines1 = hexagram_to_lines(torch.tensor([lines_to_hexagram(lines.flip(-1)).item()]))
        zong_twice = lines1.flip(-1)
        assert torch.all(lines == zong_twice), f"综卦 double flip failed for {idx}"
    
    print("  ✓ zong_hexagram (综卦): double-flip idempotent")


def test_hu_hexagram():
    """Test inner hexagram (互卦)."""
    config = YiCeNetConfig()
    
    # Hu should always produce a valid hexagram (0-63)
    for idx in range(64):
        x = torch.tensor([idx])
        lines = hexagram_to_lines(x)
        hu = hu_hexagram(lines)
        assert 0 <= hu[0].item() < 64, f"互卦 invalid for {idx}"
    
    print("  ✓ hu_hexagram (互卦): always produces valid hexagram")


def test_zhi_hexagram():
    """Test changing hexagram (之卦)."""
    # Flipping one line should change the hexagram
    x = torch.tensor([63])  # 乾 (all yang)
    lines = hexagram_to_lines(x)
    
    for pos in range(6):
        pos_t = torch.tensor([pos])
        zhi = zhi_hexagram(lines, pos_t)
        # Should be different from original
        assert zhi[0].item() != 63, f"之卦: flipping line {pos} should change hexagram"
    
    print("  ✓ zhi_hexagram (之卦): each line flip changes hexagram")


def test_generate_candidates():
    """Test candidate generation produces exactly 8 candidates."""
    x = torch.tensor([42])  # arbitrary starting hexagram
    candidates = generate_candidates(x)
    assert candidates.shape == (1, 8), f"Expected (1, 8), got {candidates.shape}"
    # All candidates should be valid hexagrams
    for i in range(8):
        assert 0 <= candidates[0, i].item() < 64, f"Candidate {i} invalid"
    
    # With attention weights
    attn = torch.rand(1, 6)
    candidates2 = generate_candidates(x, attn)
    assert candidates2.shape == (1, 8)
    
    print("  ✓ generate_candidates: 8 valid candidates")


def test_encoder_forward():
    """Test TinyEncoder forward pass."""
    config = YiCeNetConfig()
    encoder = TinyEncoder(config)
    
    B, T = 4, 16
    input_ids = torch.randint(1, config.vocab_size, (B, T))
    mask = torch.ones(B, T)
    
    h = encoder(input_ids, mask)
    assert h.shape == (B, config.hidden_dim), f"Expected ({B}, {config.hidden_dim}), got {h.shape}"
    
    counts = encoder.get_param_count()
    assert counts["total"] > 4_000_000, f"Encoder too small: {counts['total']:,}"
    assert counts["total"] < 7_000_000, f"Encoder too large: {counts['total']:,}"
    
    print(f"  ✓ TinyEncoder forward: {h.shape}, params: {counts['total']:,}")


def test_full_model():
    """Test full YiCeNet model forward pass."""
    config = YiCeNetConfig()
    model = YiCeNet(config)
    
    B, T = 4, 16
    input_ids = torch.randint(1, config.vocab_size, (B, T))
    mask = torch.ones(B, T)
    
    output = model(input_ids, mask, tau=1.0, hard=False)
    
    expected_keys = [
        "h", "hexagram_idx", "hexagram_probs",
        "best_candidate_idx", "candidate_idxs",
        "candidate_values", "action_ids", "action_logits"
    ]
    for key in expected_keys:
        assert key in output, f"Missing output key: {key}"
    
    assert output["h"].shape == (B, config.hidden_dim)
    assert output["hexagram_idx"].shape == (B,)
    assert output["candidate_idxs"].shape == (B, 8)
    assert output["candidate_values"].shape == (B, 8, 1)
    assert output["action_ids"].shape == (B,)
    
    # Verify param count
    counts = model.get_param_count()
    total = counts["total"]
    print(f"  ✓ YiCeNet forward: all {len(expected_keys)} outputs valid")
    print(f"  ✓ Total params: {total:,} (~{total/1e6:.1f}M)")
    
    return total


def test_memory_footprint():
    """Verify memory footprint < 100 MB FP32."""
    model = YiCeNet(YiCeNetConfig())
    mem = model.estimate_memory(quantized=False)
    int8_mem = model.estimate_memory(quantized=True)
    
    assert mem < 100, f"FP32 memory too high: {mem:.1f} MB"
    assert int8_mem < 20, f"INT8 memory too high: {int8_mem:.1f} MB"
    
    print(f"  ✓ Memory: {mem:.1f} MB FP32 / {int8_mem:.1f} MB INT8")


def run_all():
    print("=" * 60)
    print("YiCeNet Test Suite")
    print("=" * 60)
    
    tests = [
        test_hexagram_conversion,
        test_cuo_hexagram,
        test_zong_hexagram,
        test_hu_hexagram,
        test_zhi_hexagram,
        test_generate_candidates,
        test_encoder_forward,
        test_full_model,
        test_memory_footprint,
    ]
    
    passed = 0
    failed = 0
    
    for t in tests:
        try:
            t()
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
