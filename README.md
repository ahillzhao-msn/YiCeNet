# YiCeNet (易策网络)

> I-Ching inspired tiny neural network for conversational navigation.

YiCeNet is a ~5.6M parameter model that learns to map user conversations to I-Ching hexagrams, using the 64-hexagram system as a navigation compass for AI agent decision-making.

## Core Idea

The 64 hexagrams of the I Ching represent 64 fundamental patterns of change. YiCeNet learns to recognize which pattern best describes each conversation turn, then selects an appropriate orchestration action — predicting not just *what* to do, but *how* to be.

## Key Features

- **Self-modulating denoising** — Model learns to ignore noise by its own prediction confidence, not external classification
- **Dual-head World Model** — Short-term fluctuations (3-day half-life) feed into long-term patterns (30-day half-life)
- **Hot-swappable** — Registry-based checkpoint management, swap models without restart
- **Autonomous flywheel** — 12-hour training cycle, self-improves without human intervention

## Quick Start

```bash
# Train a new model with DS supervision
python scripts/ds_train.py \
  --version v15 \
  --buffer data/flywheel_buffer.jsonl \
  --ds-results data/ds_eval_all.jsonl \
  --endogenous

# Prune old checkpoints
python scripts/checkpoint_manager.py prune

# Check current active model
python -c "import json; print(json.load(open('checkpoints/registry.json'))['active']['version'])"
```

## Performance

| Metric | v4 | v6 | v15 |
|--------|----|----|-----|
| Unique hexagrams | 48/64 | 38/64 | **58/64** |
| Confidence | 0.708 | 0.981 | **0.966** |
| Avg RL reward | — | — | **0.982** |

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full design details.

## Design Philosophy

**诚·直** — Sincerity and directness. The system does not pretend to know what it doesn't, and its self-critique is genuine, not performative.

**知之为知之，不知为不知** — Know the exact boundary of what you know and don't know. The WM's prediction surprise is the measure of this boundary.

**降噪即訓練，訓練即降噪** — Denoising is not a separate preprocessing step. It emerges naturally from the training dynamics.
