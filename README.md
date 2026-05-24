# YiCeNet (易策网络)

> I-Ching inspired tiny neural network (~5.6M params) for conversational navigation.

YiCeNet maps user text to one of 64 I-Ching hexagrams, learning to recognize conversational patterns and select appropriate orchestration actions.

## Key Features (Implemented)

| Feature | Status | Location |
|---------|--------|----------|
| TinyEncoder (4-layer Transformer) | ✅ | `src/yicenet/model.py` |
| GumbelRouter (64-hexagram selection) | ✅ | `src/yicenet/model.py` |
| Dual-Head World Model v2 | ✅ | `src/yicenet/world_model.py` |
| Power-law forgetting curve | ✅ | `src/yicenet/world_model.py` |
| API-supervised RL training | ✅ | `scripts/rl_train.py` |
| Endogenous noise weighting | ✅ | `src/yicenet/world_model.py` → `src/yicenet/flywheel.py` |
| Hot-swap checkpoint registry | ✅ | `scripts/checkpoint_manager.py` |
| Autonomous 12h flywheel | ✅ | `src/yicenet/flywheel.py` |
| Sampling stratification (planned) | ⏳ | Not yet implemented |

## Quick Start

```bash
# Train with API supervision (configurable endpoint + model)
python scripts/rl_train.py \
  --version v16 \
  --buffer data/flywheel_buffer.jsonl \
  --eval-results data/ds_eval_all.jsonl \
  --endogenous

# Evaluate new samples via any OpenAI-compatible API
# In env: EVAL_API_URL=... EVAL_MODEL=... EVAL_API_KEY=...
python scripts/eval_api.py \
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

See [ARCHITECTURE.md](ARCHITECTURE.md) for full documentation of all components.

## Installation

See [INSTALL.md](INSTALL.md) for detailed setup guide.

```bash
# Editable install (recommended for development)
pip install -e /path/to/YiCeNet

# Then from any Python session:
#   from yicenet.model import YiCeNet
#   from yicenet.config import YiCeNetConfig
```

## Configuration (Environment Variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `EVAL_API_URL` | `https://api.deepseek.com/v1/chat/completions` | API endpoint for evaluation |
| `EVAL_MODEL` | `deepseek-chat` | Model name for evaluation |
| `EVAL_API_KEY` | (from env or .env) | API key for evaluation |
| `DEEPSEEK_API_KEY` | (fallback) | Legacy compat: replaces EVAL_API_KEY |
| `YICENET_ROOT` | auto-detected | Override project root |

## Project Layout

```
YiCeNet/
├── src/yicenet/         # Core library (editable package)
├── scripts/             # Training/evaluation CLI scripts
├── data/                # Training data & flywheel buffer
├── checkpoints/         # Model weights & registry (gitignored)
└── pyproject.toml       # Build & dependency config
```
