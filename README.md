# YiCeNet (易策网络)

> I-Ching inspired tiny neural network (~5.6M params) for conversational navigation.

YiCeNet maps user text to one of 64 I-Ching hexagrams, learning to recognize conversational patterns and select appropriate orchestration actions.

## Key Features (Implemented)

| Feature | Status | Location |
|---------|--------|----------|
| TinyEncoder (4-layer Transformer) | ✅ | `src/model.py` |
| GumbelRouter (64-hexagram selection) | ✅ | `src/model.py` |
| Dual-Head World Model v2 | ✅ | `src/world_model.py` |
| Power-law forgetting curve | ✅ | `src/world_model.py` |
| DS-supervised training pipeline | ✅ | `scripts/ds_train.py` |
| Endogenous noise weighting | ✅ | `src/world_model.py` → `flywheel.py` |
| Hot-swap checkpoint registry | ✅ | `scripts/checkpoint_manager.py` |
| Autonomous 12h flywheel | ✅ | `src/flywheel.py` |
| Sampling stratification (planned) | ⏳ | Not yet implemented |

## Quick Start

```bash
# Train with DS supervision + endogenous weighting
python scripts/ds_train.py \
  --version v16 \
  --buffer data/flywheel_buffer.jsonl \
  --ds-results data/ds_eval_all.jsonl \
  --endogenous

# Evaluate new samples via DeepSeek
python scripts/ds_evaluate.py \
  --input samples.jsonl \
  --output evaluations.jsonl \
  --batch-size 20

# Manage checkpoints
python scripts/checkpoint_manager.py prune     # remove low-score checkpoints
python scripts/checkpoint_manager.py clean     # validate registry.json
python scripts/checkpoint_manager.py register v16 path/to/model.pt 0.99
```

## Performance Summary

| Version | Samples | Unique Hexagrams | Confidence | Noise Adaptation |
|---------|---------|:----------------:|:----------:|:----------------:|
| v4 | 10K synthetic | 48/64 | 0.708 | None |
| v6 | 200 real | 38/64 | 0.981 | None |
| **v15** | **997 real** | **58/64** | **0.966** | **2 layers** |

## Design Principles

**诚·直** — Documents describe only what code implements. Aspirational design is clearly marked as "planned."

**知之为知之，不知为不知** — The system's own prediction surprise (KL divergence) is the measure of its knowledge boundary. Samples outside this boundary are naturally de-weighted.

**降噪即訓練，訓練即降噪** — Denoising emerges from training dynamics, not preprocessing.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full documentation of all components.
