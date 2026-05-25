# YiCeNet Architecture

> Version 15.0.0 · 5.6M parameters · 4ms inference

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Architecture Overview](#2-architecture-overview)
3. [Component Details](#3-component-details)
4. [Training Pipeline](#4-training-pipeline)
5. [Flywheel: Online Learning](#5-flywheel-online-learning)
6. [Config & Portability](#6-config--portability)
7. [Testing](#7-testing)

---

## 1. Design Principles

YiCeNet was designed around four core principles:

### 1.1 Structural Over Statistical

Most ML models use statistical smoothing — averaging over many examples to find the "typical" response. YiCeNet instead uses **structural reasoning**: mapping inputs through the 64-hexagram I-Ching state space, applying deterministic mutation operators (opposition, overlap, core, shift) to produce interpretable reasoning chains.

### 1.2 Tiny by Design

With only 5.6M parameters (~22MB FP32), YiCeNet runs on CPU, laptop, or edge devices. This is not a compressed version of a larger model — it is a fundamentally different architecture designed from scratch for its size class.

### 1.3 Self-Evolving

The model never stops training. Each user interaction feeds back into the flywheel loop, improving the model incrementally without requiring a full retrain.

### 1.4 Explainable

Every decision maps to a traceable hexagram path. You can inspect which hexagram was chosen, why, and what alternative paths were considered.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Input (task + context)                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  TinyEncoder (4-layer Transformer, 256-dim)                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Token Emb   │→│ Transformer │→│ Pooling     │──→ 6D vec │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Gumbel Router                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  MLP(6 → 64) → Gumbel-Softmax → discrete hexagram ID │   │
│  └──────────────────────────────────────────────────────┘   │
│  Output: argmax hexagram + full probability distribution    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Hexagram Embedding Table (64 × 256)                        │
│  Maps hexagram ID → 256-dim structured feature vector       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Mutation Engine (fixed logic, 4 operators)                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │Opposition│ │ Overlap  │ │   Core   │ │  Shift   │       │
│  │ 错卦     │ │ 综卦     │ │ 互卦     │ │ 变卦     │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│  Produces: hexagram reasoning chain (4-8 related states)    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Policy Decoder (26K params)                                │
│  Reads hexagram chain → outputs dispatch instruction        │
│  (agent_sequence, confidence, alternative_paths)            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  World Model (dual-head: prediction + value)                │
│  ┌─────────────────────┐  ┌──────────────────────┐          │
│  │ Prediction Head     │  │ Endogenous Weight    │          │
│  │ (next hexagram)     │  │ (KL-based surprise)  │          │
│  └─────────────────────┘  └──────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow (Inference)

1. **Input encoding**: task text → tokenized → TinyEncoder → 6D intent vector
2. **Hexagram selection**: Gumbel Router maps 6D vector → 64-class distribution → sample hexagram ID
3. **Embedding lookup**: ID → 256-dim hexagram feature vector
4. **Mutation**: 4 deterministic operators transform the hexagram into a reasoning chain
5. **Decoding**: Policy decoder reads chain → outputs agent sequence + confidence
6. **World model eval**: optional — predicts next likely hexagram and computes surprise score

---

## 3. Component Details

### 3.1 TinyEncoder (`encoder.py`)

A 4-layer Transformer encoder with 256-dim hidden states.

| Parameter | Value |
|-----------|-------|
| Layers | 4 |
| Hidden dim | 256 |
| Attention heads | 4 |
| FFN dim | 1024 |
| Vocabulary | 8,000 (rebucketed Qwen BPE) |
| Parameters | ~5.3M |
| Output | 6-dim intent vector (mean-pooled) |

### 3.2 Gumbel Router (`model.py`)

| Parameter | Value |
|-----------|-------|
| Input | 6D (from TinyEncoder) |
| Output | 64-class logits |
| Sampling | Gumbel-Softmax (τ=1.0, annealing) |
| Temperature range | 1.0 → 0.1 over training |
| Parameters | ~470 |

### 3.3 Hexagram Embedding (`constants.py` + `model.py`)

| Parameter | Value |
|-----------|-------|
| Table size | 64 × 256 |
| Initialization | Pretrained via K-means on synthetic traces |
| Training | Fine-tuned via RL |
| Parameters | 16,384 |

Each hexagram has a binary encoding (6-bit, yin=0/yang=1) and a learned embedding that captures its "decision archetype":

| ID | Binary | Name | Archetype |
|----|--------|------|-----------|
| 1 | 111111 | 乾 (Qián) | Creative, strong action |
| 2 | 000000 | 坤 (Kūn) | Receptive, supportive |
| 3 | 010001 | 屯 (Zhūn) | Initial difficulty |
| ... | ... | ... | ... |

### 3.4 Mutation Engine (`hexagram.py`)

The four I-Ching structural operators (fixed logic, no learned parameters):

| Operator | Chinese | Description | Effect |
|----------|---------|-------------|--------|
| Opposition | 错卦 | Flip every line (yin↔yang) | Complete reversal — used for contrasting alternatives |
| Overlap | 综卦 | Reverse the hexagram order | Mirror perspective — used for second opinions |
| Core | 互卦 | Extract inner 3-4-5 as new trigram | Hidden structure — used for deeper analysis |
| Shift | 变卦 | Change one line at a time | Gradual transition — used for step-by-step reasoning |

### 3.5 Policy Decoder (`decoder.py`)

| Parameter | Value |
|-----------|-------|
| Architecture | 2-layer MLP (256 → 128 → 32) |
| Input | Hexagram chain embeddings (concatenated) |
| Output | Agent sequence + confidence scores |
| Parameters | ~26K |

### 3.6 World Model (`world_model.py`)

Dual-head architecture that predicts the next hexagram and evaluates decision quality:

| Component | Description |
|-----------|-------------|
| Prediction head | Predicts next hexagram from current + context |
| Value head | Estimates expected reward of current hexagram |
| Endogenous weight | KL(WM_prediction || observation) as surprise metric |
| Forgetting curve | Power-law decay: `weight = (t + 1)^(-0.5)` |
| Parameters | ~125K |

The endogenous weight serves as the model's own "knowledge boundary detector" — high KL divergence indicates out-of-distribution input, causing automatic deweighting in training.

### 3.7 Value Network (`value_net.py`)

| Parameter | Value |
|-----------|-------|
| Architecture | 3-layer MLP (256 → 128 → 64 → 1) |
| Training | MSE on RL trajectory rewards |
| Parameters | ~41K |

---

## 4. Training Pipeline

### Stage 1: Pretraining

```bash
python src/yicenet/train.py
```

- Generates 10K synthetic hexagram assignment traces
- K-means initializes hexagram embeddings from task→hexagram mapping
- Cross-entropy loss with temperature annealing
- Output: `checkpoints/yicenet_pretrained.pt`

### Stage 2: World Model Training

```bash
python src/yicenet/train_value_net.py
```

- Uses real interaction data (from flywheel buffer)
- Trains prediction head + value head jointly
- MSE + binary cross-entropy loss
- Output: `checkpoints/world_model_v*.pt`

### Stage 3: RL Fine-tuning

```bash
python scripts/rl_train.py --version v16 --endogenous
```

- REINFORCE with baseline (learned value network)
- Reward signal from API evaluator (teacher model)
- Endogenous noise weighting enabled
- Output: `checkpoints/yicenet_v16.pt`

### Training Dataset

| Dataset | Size | Source | Content |
|---------|------|--------|---------|
| Synthetic pretrain | 10K | Auto-generated | Random task→hexagram traces |
| API evaluations | 898 | DeepSeek V4 eval | Real hexagram quality ratings |
| Flywheel buffer | ~200 | User interactions | Online RL training data |

---

## 5. Flywheel: Online Learning

The flywheel runs autonomously (every 6h by default) and performs:

```
collect → evaluate → train → register → hot-swap
```

### Pipeline

1. **Collect**: Scans interaction logs, extracts task + chosen hexagram + outcome
2. **Evaluate**: Computes satisfaction score from conversation signals (token cost, response length, continuation patterns)
3. **Train**: 200-episode RL fine-tune with endogenous noise weighting
4. **Register**: New checkpoint saved to registry.json with version number + avg_reward
5. **Hot-swap**: Production engine loads new weights on next inference (no restart)

### Denoising (3 layers)

| Layer | Location | Method |
|-------|----------|--------|
| Sampling filter | Future | Quality-stratified sampling before flywheel |
| DS SAT temperature | `flywheel.py` | Lower satisfaction → higher temperature → flatter target → weak signals ignored |
| Endogenous weight | `world_model.py` | KL(WM_prediction || target) → high surprise = low weight |

---

## 6. Config & Portability

### Environment Variables

All paths follow this priority: `YICENET_HOME` env var > source tree auto-detect.

| Variable | Purpose |
|----------|---------|
| `YICENET_HOME` | Override project root directory |
| `EVAL_API_URL` | Teacher model API endpoint |
| `EVAL_MODEL` | Teacher model name |
| `EVAL_API_KEY` | Teacher API key (from env or .env) |

### Zero Hardcoded Paths

All file paths use `yicenet_home()` from `config.py`, which automatically resolves relative to the package root. The package is fully relocatable with `pip install -e .` — no absolute paths, no user names, no machine-specific references.

---

## 7. Testing

```bash
# Run test suite
cd tests && python -m pytest -v

# Key tests:
# - test_model.py: Forward pass, hexagram output validity
# - test_tokenizer.py: Qwen BPE integration
# - test_flywheel.py: Online training pipeline
# - test_config.py: Path resolution, env var override
# - test_hexagram.py: Mutation engine correctness
```

---

*YiCeNet — 5.6M parameters · 4ms · Fully local · Continuously evolving*
