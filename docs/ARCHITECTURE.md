# YiCeNet (易策网络) — Architecture

> 太極生兩儀，兩儀生四象，四象生八卦

## Overview

YiCeNet is a tiny neural network (~5.7M params, 22MB) that learns I-Ching hexagram prediction as a proxy for conversational navigation. It maps user text to one of 64 hexagrams, then selects an orchestration action.

```
Input ─→ TinyEncoder ─→ h (256-dim) ─→ GumbelRouter ─→ hexagram (1/64)
                                             │
                                    ┌───────┴───────┐
                                    ▼               ▼
                              Value Network    Action Decoder
                                    │               │
                                    └───────┬───────┘
                                            ▼
                                     Navigation Decision
```

---

## Core Components

### TinyEncoder

4-layer Transformer encoder (8M params).

```
Input:  BPE-tokenized text → 8000 vocab → 128 max tokens
  → Embedding (256-dim)
  → 4× Transformer blocks (4 heads, 1024 FFN)
  → Mean pool → LayerNorm → StateProjection(256→256)
Output: h (256-dim) — conversational state vector
```

### GumbelRouter

Binary decision network that selects a hexagram from the state vector `h`.

```
h (256-dim) → Linear(256→64) → Gumbel-Softmax → hexagram index (1/64)
```

- Temperature τ anneals from 1.0 → 0.1 during training
- Low τ → near-argmax (exploitation), high τ → near-uniform (exploration)
- Output: hexagram_idx + categorical probabilities over 64 hexagrams

### Hexagram Embedding + Trigram Cross-Attention

Each of the 64 hexagrams has a 256-dim embedding vector. The 8 trigrams (八卦) are represented as learnable prototypes with cross-attention to the hexagram embedding.

- `hexagram_embed: nn.Embedding(64, 256)` — learned hexagram prototypes
- `trigram_prototypes: nn.Parameter(8, 256)` — 8 basic patterns
- Cross-attention computes compatibility between selected hexagram and each trigram

### Value Network

Small MLP that scores each candidate hexagram.

```
256-dim → 128 → 1 (value scalar)
```

Learned via regression to match the World Model's 64-dim prediction reward.

### Action Decoder

Projects hexagram + value to action tokens.

```
hexagram_embed(256) + value(1) → 128 → num_actions(50)
```

Outputs logits over 50 orchestration primitives (search, read, write, delegate, etc.).

---

## Dual-Head World Model (WorldModelV2)

~20K params. The smallest but most critical component — serves as the training critic.

```
Input: probes(ℝ⁹) + hexagram_onehot(ℝ⁶⁴) → ℝ⁷³
  → LayerNorm
  → Linear(73→128) → GELU  (shared layer)
      ├── Head A: Linear(128→64) → Softmax → hexagram distribution ℝ⁶⁴
      └── Head B: Linear(128→3) → Sigmoid → external metrics ℝ³

Head A targets: [hexagram_distribution]  — long-term (τ=30 days)
Head B targets: [token_cost, response_length, satisfaction] — short-term (τ=3 days)
```

### Key Behaviors

- **Power-law forgetting**: Older samples decay as `w(t) = (1 + t/τ)^(-α)` — never reach zero, just become whispers
- **Dual-head sharing**: Short-term fluctuations in Head B feed into Head A's shared layer
- **Endogenous weighting**: KL(WM_pred || actual_target) used as self-confidence during training (see below)

### Training

World Model is trained first (supervised on DS-evaluated samples), then frozen during RL fine-tuning where it serves as the reward function:

```
reward = cosine_similarity(WM_prediction, DS_target) ∈ [0, 1]
```

---

## Noise Adaptation (2 Layers Implemented)

The system has two implemented mechanisms for handling noisy training data. A third layer is designed but not yet coded.

### Layer 1 (Implemented): DS Confidence Temperature Modulation

**Location**: `src/rl_train.py` — `project_to_hexagram_space()`

```
if satisfaction is not None:
    effective_temp = max(0.1, 1.0 - abs(satisfaction))
```

- High |satisfaction| (DS is confident) → low temperature → sharp target → strong learning
- Low |satisfaction| (DS is uncertain) → high temperature → near-uniform → near-zero loss
- **No external noise classifiers** — the DS evaluator's own uncertainty IS the noise detector

Callers that pass `satisfaction`:
- `scripts/ds_train.py` — both WM training and RL fine-tuning phases

### Layer 2 (Implemented): Endogenous Prediction Surprise

**Location**: `src/world_model.py` — `compute_endogenous_weight()`

```python
def compute_endogenous_weight(self, probes, hexagram_id, target_dist):
    pred_dist, _ = self.forward(probes, hexagram_id)
    kl = KL(pred_dist || target_dist)
    weight = 1.0 - sigmoid((kl - 0.03) * 50)
    return weight   # [0, 1]
```

- Low KL (WM predicted correctly) → weight ≈ 1.0 → strong learning
- High KL (WM prediction missed) → weight ≈ 0.1 → noise suppressed
- **Fully endogenous** — no external evaluator needed after initial training

Integrated into:
- `scripts/ds_train.py` — via `--endogenous` flag (passes weight to WM training loop)
- `src/flywheel.py` — default on for 12h autonomous training

### Layer 3 (Designed, Not Implemented): Sampling Stratification

Goal: Classify buffer samples by quality before training, stratified sampling to ensure diverse coverage. Code not yet written — documented here for future implementation.

---

## Evaluation & Training Pipeline

### DS-Supervised Training (`scripts/ds_train.py`)

```
python scripts/ds_train.py \
  --version v15 \
  --buffer data/flywheel_buffer.jsonl \
  --ds-results data/ds_eval_all.jsonl \
  --endogenous
```

Two phases:
1. **WM training**: Each sample → YiCeNet forward (probes + hex) → WM prediction → DS target → KL loss × power-law weight × endogenous weight
2. **RL fine-tuning**: 200 episodes, 64-dim projection reward via WM

### Autonomous Flywheel (`src/flywheel.py`)

Runs via cron every 12h:
1. `_collect_new_messages()` — scan Hermes session DB for new user messages
2. `_update_world_model_v2()` — incremental WM update with power-law + endogenous weighting
3. `_rl_fine_tune_v5()` — 200-ep RL, registers result in registry.json

### Batch DS Evaluation (`scripts/ds_evaluate.py`)

Sends samples to DeepSeek API in batches (default 20/batch), returns satisfaction scores and signal flags. Each sample costs ~200 tokens.

---

## Checkpoint Management

### Registry (`checkpoints/registry.json`)

```json
{
  "active":  {"version": "v15", "path": "...yicenet_v15.pt", "avg_reward": 0.982},
  "ready":   {"version": "v14-sat", "path": "...yicenet_v14_sat.pt", ...},
  "fallback": {...},
  "history": [...]
}
```

### Hot-Swap

`src/hermes_tool.py` checks registry.json on each `yicenet_predict()` call. If `active.version` changed since engine load, calls `engine.switch_model(new_path)` — no restart needed.

### Pruning

`scripts/checkpoint_manager.py prune` keeps:
- Base model (v4) — never deleted
- Active model — never deleted
- Top 3 scoring (reward × 100 + version_number)

---

## Project Structure

```
YiCeNet/
├── src/                    # Implementation (11 files, ~3500 lines)
│   ├── model.py            YiCeNet model — TinyEncoder + GumbelRouter + ValueNet + Decoder
│   ├── world_model.py      WorldModelV2 — dual-head + power-law + endogenous weight
│   ├── flywheel.py         12h autonomous training pipeline
│   ├── hermes_tool.py      Hermes integration with hot-swap
│   ├── yicenet_engine.py   Inference engine with switch_model()
│   ├── rl_train.py         project_to_hexagram_space + compute_hexagram_reward
│   ├── config.py           YiCeNetConfig — all hyperparameters
│   ├── tokenizer.py        BPE → YiCeNet token rebucket
│   ├── encoder.py          TinyEncoder implementation
│   ├── probes.py           probe tensor extraction
│   ├── decoder.py          ActionDecoder
│   ├── value_net.py        ValueNetwork
│   ├── hexagram.py         Hexagram line pattern + candidate generation
│   ├── train.py            Two-stage training entry (pretrain + RL)
│   ├── train_value_net.py  Value net standalone training
│   ├── metrics.py          Evaluation metrics
│   ├── constants.py        Hardcoded lookup tables
│   ├── external_metrics.py External satisfaction computation
│   └── interfaces.py       Type stubs
├── scripts/                # 3 operational scripts
│   ├── ds_train.py         DS-supervised training (--endogenous flag)
│   ├── ds_evaluate.py      Batch DS evaluation via API
│   └── checkpoint_manager.py  Registry management + pruning + hot-swap
├── docs/
│   └── ARCHITECTURE.md     This file
└── README.md
```

## Model Comparision (v4 vs v6 vs v15)

| Metric | v4 | v6 | v15 (endogenous) |
|--------|------|------|--------|
| Architecture | Pretrained | RL fine-tuned | DS-supervised |
| Unique hexagrams | 48/64 | 38/64 | **58/64** |
| Confidence | 0.708 | 0.981 | **0.966** |
| Hexagram diversity | 0.240 | 0.190 | **0.290** |
| RL avg reward | — | — | **0.982** |
| Training samples | 10K synthetic | 200 real | **997 real** |
| Noise adaptation | None | None | **2 layers** |
