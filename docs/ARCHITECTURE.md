# YiCeNet (易策网络) — Architecture

> 太極生兩儀，兩儀生四象，四象生八卦

## Overview

YiCeNet is a tiny neural network (~5.6M params, ~22MB) that learns I-Ching hexagram prediction as a proxy for conversational navigation. It predicts which of the 64 hexagrams best represents a user's intent, then selects an orchestration action.

```
User Input → TinyEncoder → 256-dim state → GumbelRouter → Hexagram (1/64)
                                                              ↓
                                              Value Network + Action Decoder
                                                              ↓
                                                    Navigation Decision
```

## Three-Layer Noise Design

The core innovation is a **self-modulating denoising** system — the model learns what to ignore, not by external classification, but by its own prediction confidence.

### Layer 1: DS Confidence Temperature Modulation

```
project_to_hexagram_space(reward_signals, satisfaction)
  → effective_temp = max(0.1, 1.0 - |satisfaction|)
  → target = softmax(logits / effective_temp)
```

- High |satisfaction| → low temp → sharp target → strong learning signal
- Low |satisfaction| → high temp → near-uniform → near-zero KL loss
- **No external noise classifier needed** — DS evaluator's own uncertainty is the noise detector

### Layer 2: Endogenous Prediction Surprise

```
WM.predict(probes, hex_id)   → predicted hexagram distribution
target = project(signals)     → actual outcome distribution
surprise = KL(pred || target) → how surprised is the WM?

weight = 1 - sigmoid((surprise - 0.03) × 50)
```

- Low surprise (WM predicted correctly) → high weight → strong learning
- High surprise (WM prediction missed) → low weight → noise suppression
- **Fully endogenous** — no external evaluator needed after initial training

### Layer 3: Sampling Stratification (Future)

Classify buffer samples by quality before training, stratified sampling to ensure diverse coverage.

## Architecture Components

### TinyEncoder (4-layer Transformer)
- Input: BPE-tokenized text (max 128 tokens, 8000 vocab)
- 4 transformer layers, 256-dim hidden, 4 heads
- Output: 256-dim state vector `h`

### GumbelRouter
- Projects `h` to 64-dim logits → Gumbel-Softmax → hexagram sample
- Temperature annealing: 1.0 → 0.1 over training

### Dual-Head World Model (WorldModelV2)
~20K parameters, the smallest but most important component.

```
Input: probes(ℝ⁹) + hexagram_onehot(ℝ⁶⁴) → ℝ⁷³
  → Shared(73→128) → GELU
      ├── Head A (128→64):  hexagram distribution prediction (long-term, τ=30d)
      └── Head B (128→N):   external metrics (short-term, τ=3d)
```

- **Head A**: Predicts which hexagram distribution will result from this (probe, hex) pair
- **Head B**: Predicts [token_cost, response_length, satisfaction]
- **Shared layer**: Short-term fluctuations feed back into long-term patterns
- **Power-law forgetting**: Older samples decay gracefully, never reach zero

### Endogenous Weight (Layer 2 implementation)

```python
wm.compute_endogenous_weight(probes, hex_id, target_dist) → weight [0, 1]
```

Uses KL divergence between WM's prediction and actual target as a self-confidence metric. Called during training to modulate each sample's contribution to the loss.

## Training Pipeline

### DS-Supervised Training

```
scripts/ds_train.py \
  --version v15 \
  --buffer data/flywheel_buffer.jsonl \
  --ds-results data/ds_eval_all.jsonl \
  --endogenous

Phase 1: WM training on all samples with power-law + endogenous weighting
Phase 2: RL fine-tuning (200 episodes, 64-dim projection reward)
```

### Flywheel (12h cron)

```
src/flywheel.py → _collect_new_messages → _update_world_model_v2 → _rl_fine_tune_v5
```

The flywheel runs autonomously every 12 hours. The WM gets incremental updates with endogenous noise-weighting. New checkpoints are registered in `checkpoints/registry.json` for hot-swap.

## Checkpoint Management

```
scripts/checkpoint_manager.py  [prune|clean|register|fresh]

Keeps: base model (v4) + active model + top 3 scoring checkpoints
Hot-swap: hermes_tool.py checks registry.json on each predict call
```

## Performance (v15)

| Metric | v4 | v6 | v14-SAT | v15-Endogenous |
|--------|----|----|---------|----------------|
| Unique hexagrams | 48/64 | 38/64 | **60/64** | 58/64 |
| Confidence | 0.708 | **0.981** | 0.908 | **0.966** |
| Diversity | 0.240 | 0.190 | **0.300** | 0.290 |
| RL Reward | — | — | 0.984 | 0.982 |

## Project Structure

```
YiCeNet/
├── src/
│   ├── model.py          # YiCeNet model (5.6M params)
│   ├── world_model.py    # Dual-head World Model v2
│   ├── flywheel.py       # 12h autonomous training cron
│   ├── hermes_tool.py    # Hermes integration (hot-swap)
│   ├── yicenet_engine.py # Inference engine
│   ├── rl_train.py       # RL + projection components
│   └── ...
├── scripts/
│   ├── ds_train.py       # DS-supervised training
│   ├── ds_evaluate.py    # Batch DS evaluation
│   └── checkpoint_manager.py
├── checkpoints/
│   ├── registry.json     # Active/ready/fallback model registry
│   ├── yicenet_v4.pt     # Base model (always kept)
│   └── yicenet_v15.pt    # Current best model
└── data/
    └── flywheel_buffer.jsonl  # Conversation buffer
```
