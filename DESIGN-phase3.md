# Phase 3 — WorldModelV3: 27-dim Context Vector Upgrade

> Goal: Replace WorldModelV2's 9-dim `probes` input with the new 27-dim
> `context_vector`, retrain from 4561 flywheel buffer entries, verify quality,
> push v17.0.0.

---

## 1. Motivation

The ContextCollector (Phase 1+2) produces a 27-dim `context_vector` per turn,
stored in MemoryBank metadata. The WorldModel currently consumes only 9-dim
probe embeddings from the encoder — it cannot see the rich environment signals
available in the context vector.

## 2. WorldModelV3 Design

```python
class WorldModelV3(nn.Module):
    """
    Input:  context_vector (B, 27) + hexagram_id (B,) → one-hot (B, 64)
            Total: ℝ⁹¹  (was ℝ⁷³)
    HeadA:  hexagram distribution (B, 64)  — KL divergence (same as V2)
    HeadB:  predicted metrics (B, 3)       — MSE (same as V2)
    Params: Shared(91→128) + HeadA(128→64) + HeadB(128→3) ≈ 21K
    """
    def __init__(self,
        context_dim: int = 27,          # ← NEW: was probe_dim=9
        num_hexagrams: int = 64,
        shared_dim: int = 128,
        num_external_metrics: int = 3,  # satisfaction, tool_success, hex_conf
        slow_tau_days: float = 30.0,
        fast_tau_days: float = 3.0,
        alpha: float = 1.5,
        beta: float = 0.3,
    ):
```

Changes from V2:
| Aspect | V2 | V3 |
|--------|-----|-----|
| Input | `probes(B,9)` from encoder | `probes(B,9) + context_vector(B,27)` |
| | | **Total: ℝ³⁶** (察言+观色) |
| Hexagram | `hexagram_id(B,)` one-hot | Same, unchanged |
| Total input dim | 9+64=73 | 9+27+64=**100** |
| HeadB targets | 3 from flywheel | Same 3 (satisfaction, success, conf) |
| forward() signature | `(probes, hexagram_id)` | `(probes, context_vector, hexagram_id)` |
| save/load config | `probe_dim` key | `probe_dim=9, context_dim=27` |

## 3. Training Data Pipeline

### Source: Flywheel Buffer (4561 entries, June 8–22)

Each flywheel entry has:
```json
{
  "timestamp": 17809...,        "satisfaction": 0.60,
  "completed": true/false,      "corrected": true/false,
  "praised": true/false,        "abandoned": true/false,
  "continued": true/false,      "token_cost": 37699,
  "hexagram_evolution": [...],  "user_text": "...",
  "conversation_id": "..."
}
```

### Map to 27-dim context_vector (training feature)

```
tok_user_input_len     → min(len(user_text)/512, 1.0)
tok_is_first_turn      → 0 (no per-turn data in flywheel)
tok_prompt_tokens      → token_cost / 4096
tok_completion_tokens  → token_cost * 0.3 / 4096  (estimate 30% completion)
tok_api_duration       → 0.5 (default)
tok_tool_count         → 0.0 (no tool data)
tok_tool_success_rate  → 1.0 (default)
tok_tool_retry_count   → 0.0
tok_tool_duration      → 0.0
tok_tool_output_size   → 0.0
tok_tool_diversity     → 0.0
tok_response_len       → token_cost * 0.01 / 4000
tok_has_code           → float("```" in user_text)
tok_code_block_count   → 0.0
tok_hex_conf           → 0.5 (default)
tok_hex_q_gap          → 0.0
tok_hex_entropy        → 0.5 (default)
tok_user_speed         → 0.3 (default)
tok_user_speed_ratio   → 1.0
tok_mood_trend         → satisfaction - 0.5
tok_drift_trend        → 0.0
tok_is_prev_correction → 0.0
tok_is_prev_praise     → 0.0
tok_user_satisfaction  → satisfaction (from flywheel)
tok_is_correction      → float(corrected)
tok_is_praise          → float(praised)
tok_is_abandon         → float(abandoned)
```

### HeadB target vector (3-dim)
```
target_ext[0] = satisfaction
target_ext[1] = 1.0 if completed else 0.0  (tool_success proxy)
target_ext[2] = 0.5 (hex_conf default)
```

### HeadA target (hexagram distribution)
Only available when `hexagram_evolution` is non-empty (~5/4561 entries).
For the rest, HeadA is masked (returns 0 loss contribution).

## 4. Implementation Steps

| Step | File | Change |
|------|------|--------|
| 1 | `world_model.py` | Add WorldModelV3 class. forward() accepts context_vector. |
| 2 | `world_model.py` | Keep backward compat: WorldModel = WorldModelV2 (old code) |
| 3 | `scripts/rl_train.py` | `supervised_wm_training()`: generate context_vector from flywheel, feed to WM |
| 4 | `scripts/rl_train.py` | `rl_fine_tune_v14()`: use V3's predict_headA/B |
| 5 | `yicenet_engine.py` | YiCeNetEngine: create WM with context_dim=27 |
| 6 | `world_model.py` | WorldModelV3.save(): use context_dim in config |
| 7 | `world_model.py` | WorldModelV3.load(): parse context_dim from config |

## 5. Training & Validation

```bash
cd ~/YiCeNet && python scripts/rl_train.py \
  --version v17 \
  --buffer ~/.yicenet/data/flywheel_buffer.jsonl \
  --epochs 100 \
  --endogenous
```

Quality metrics:
| Metric | Target | Measurement |
|--------|--------|-------------|
| HeadA val KL | < 0.1 | KL divergence on held-out hexagram entries |
| HeadB val MSE | < 0.05 | MSE on satisfaction/success/conf prediction |
| Correlation | > 0.5 | pred_satisfaction vs actual satisfaction (Pearson r) |

## 6. Release

```bash
# After training produces world_model_v17.pt
cp checkpoints/world_model_v17.pt checkpoints/world_model_best.pt
```

Version: 17.0.0
Release tag: v17.0.0
Artifacts: world_model_v17.pt (~127KB), yicenet_v17.pt (~22MB)

---

## 7. Files Changed Summary

| File | Change |
|------|--------|
| `src/yicenet/world_model.py` | +WorldModelV3 class (~150 lines) |
| `scripts/rl_train.py` | Update training loop for V3 context_vector input |
| `src/yicenet/yicenet_engine.py` | Create WM with context_dim=27 |
| `pyproject.toml` | v16.2.0 → v17.0.0 |
| `src/yicenet/__init__.py` | __version__ = "17.0.0" |
