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
8. [Reward Signal Architecture](#8-reward-signal-architecture)

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
│  │ Token Emb   │→│ Transformer │→│ Pooling     │──→ 256D h │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Gumbel Router                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  MLP(256 → 64) → Gumbel-Softmax → discrete hexagram ID  │   │
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

1. **Input encoding**: task text → tokenized → TinyEncoder → 256-dim state vector h
2. **Hexagram selection**: Gumbel Router maps 256D vector → 64-class distribution → sample hexagram ID
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
| Output | 256-dim state vector (mean-pooled + Tanh projection) |

### 3.2 Gumbel Router (`model.py`)

| Parameter | Value |
|-----------|-------|
| Input | 256D (from TinyEncoder) |
| Output | 64-class logits |
| Sampling | Gumbel-Softmax (τ=1.0, annealing to 0.1) |
| Temperature range | 1.0 → 0.1 over training |
| Parameters | ~16.4K |

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
python -m yicenet.train --stage pretrain  # or: python src/yicenet/train.py
```

- Generates 10K synthetic hexagram assignment traces (or uses real session dataset)
- K-means initializes hexagram embeddings from task→hexagram mapping
- Contrastive loss fine-tunes encoder + hexagram embeddings
- Output: `checkpoints/yicenet_pretrained.pt`

### Stage 2: World Model Training

The World Model trains on real interaction data from the flywheel buffer or API evaluations:

- **Local path** (flywheel, zero-cost): Uses heuristic signals from `external_metrics.py` — regex pattern matching on user follow-up messages to detect continuation, correction, completion, praise, abandonment
- **API path** (higher quality): Uses `scripts/eval_api.py` to send batches to DeepSeek/OpenAI for fine-grained satisfaction scoring (-1.0 to +1.0) with per-sample reasoning

Both paths train the World Model's dual heads:
- Head A: predict hexagram distribution from probes
- Head B: predict external vector [token_cost, response_length, satisfaction]

Output: `checkpoints/world_model_best.pt`

### Stage 3: RL Fine-tuning

```bash
python scripts/rl_train.py --version v16 --endogenous
```

- Reward signal: cosine similarity between WM-predicted hexagram distribution and projected target distribution from actual user behavior signals
- WM serves as **reward proxy** — smoother and more stable than raw heuristic signals
- Endogenous noise weighting: samples where WM is surprised (high KL) are de-weighted
- Output: `checkpoints/yicenet_v{N}.pt`

### Training Dataset

| Dataset | Size | Source | Content |
|---------|------|--------|---------|
| Synthetic pretrain | 10K | Auto-generated | Random task→hexagram traces |
| API evaluations | 898 | DeepSeek V4 eval | Real hexagram quality ratings |
| Flywheel buffer | ~200 | User interactions | Online RL training data |

---

## 5. Flywheel: Online Learning

The flywheel runs autonomously (every 6h by default via Hermes cron `yicenet_flywheel`) and performs:

```
scan → buffer → update WM → RL fine-tune → register ready → evaluate → auto-promote
```

### Pipeline

1. **Scan**: Reads Hermes session DB for new user messages since last checkpoint, extracts reward signals from follow-up messages via `external_metrics.py`
2. **Buffer**: Appends new samples to `flywheel_buffer.jsonl` with timestamps for power-law weighting. Defers if buffer < 20 samples
3. **Update WM**: Incremental World Model v2 training with power-law weighted loss (slow τ=30d for hexagram head, fast τ=3d for external head) + endogenous noise weighting
4. **RL Fine-tune**: 200-episode policy gradient on buffer samples, using WM predictions as reward proxy
5. **Register**: New checkpoint saved as `yicenet_v{N}.pt`, registered as 'ready' in `registry.json`
6. **Evaluate**: Run new model on buffer, compute avg_reward and win_rate, update registry metrics
7. **Auto-promote**: Compare ready vs active win_rate on buffer. If ready outperforms by ≥3%, hot-swap to new checkpoint automatically

### Denoising (3 layers)

| Layer | Location | Method |
|-------|----------|--------|
| Sampling filter | (Future) | Quality-stratified sampling before flywheel |
| DS SAT temperature | `rl_train.py` | Lower |satisfaction| → higher temperature → flatter target → weak signals naturally ignored |
| Endogenous weight | `world_model.py` | KL(WM_prediction || target) → high surprise = low weight. WM learns its own noise perception |

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

## 8. Reward Signal Architecture

### The Core Question

YiCeNet is a reinforcement learning system. Every RL system must answer: **where does the reward come from?** For YiCeNet, the reward is *not* an explicit "this hexagram was correct." There is no labeled dataset of (input, correct_hexagram) pairs. Instead, the reward is **inferred from user behavior after the system acts**.

### Five-Layer Signal Pipeline

```
                    ┌─── Layer 1: Local Heuristics (zero-cost) ───
                    │  external_metrics.py
                    │  Regex patterns on user's NEXT message →
                    │    "thanks" / "great"    → praised    +1.0
                    │    "no" / "wrong" / "不对" → corrected  -1.0
                    │    "ok" / "done" / "好的" → completed  +0.5
                    │    normal continuation     → continued  +0.3
                    │    no follow-up            → abandoned  -0.5
User Message ──────┼─── Layer 2: API Supervision (high quality, has cost) ───
  │                 │  eval_api.py → DeepSeek / OpenAI
  │                 │  LLM evaluates each conversation turn →
  │                 │    satisfaction_score: -1.0 ~ +1.0 (fine-grained, continuous)
  │                 │    signals: {continued, corrected, completed, praised, abandoned}
  │                 │    reasoning: "User corrected the agent's misunderstanding"
  │                 │
  │                 └─── Layer 3: World Model Internalization ───────
  │                       WorldModelV2 (dual-head, ~20K params)
  │                       Input:  (six-probes ℝ⁹, hexagram_id one-hot ℝ⁶⁴) → ℝ⁷³
  │                       Head A: predict outcome hexagram distribution ℝ⁶⁴
  │                       Head B: predict external vector [token_cost,续航,satisfaction] ℝ³
  │
  │                       Loss: KL_divergence(pred_hex_dist, target_dist) × power_law_weight
  │                           + β × MSE(pred_ext_vec, target_ext_vec) × power_law_weight
  │
  ├── YiCeNet ──────── Layer 4: WM as Reward Proxy ──────────
  │  Forward pass         rl_train.py / flywheel._rl_fine_tune_v5()
  │  (text → h → 卦)      target_dist = project_to_hexagram_space(user_signals)
  │     │                 wm_pred_dist = WM(probes, hexagram_id)
  │     │                 reward = cosine_sim(wm_pred_dist, target_dist) → [0, 1]
  │     │                 policy_gradient: loss = -log_prob × reward
  │     │
  │     └── Layer 5: Flywheel Autonomous Loop ─────────────
  │           flywheel.py (every 6h cron)
  │           scan session DB → extract heuristic signals →
  │           update World Model → RL fine-tune YiCeNet →
  │           register as 'ready' in registry.json → A/B auto-promote
  │
  └── Outcome: user continues / corrects / praises / abandons → next flywheel cycle
```

### Why the World Model Is the Bridge

YiCeNet does **not** learn directly from raw heuristic signals. The pipeline is:

```
Raw signal (noisy, delayed, regex-based)
    │
    ▼
World Model (learns: what probe pattern → what outcome distribution)
    │
    ▼
WM prediction vs actual outcome → cosine similarity → smooth reward ∈ [0,1]
    │
    ▼
YiCeNet RL (policy gradient on smooth reward)
```

**Why this indirection?** Raw heuristic signals are too noisy for direct RL. A regex match on "no" cannot distinguish between "no, I meant the other file" (genuine correction) and "no problem, thanks" (satisfaction). But the same "no" produces different probe vectors (ℝ⁹ internal state) depending on context. The World Model learns to associate probe patterns with actual outcomes, effectively denoising the signal before it reaches YiCeNet's RL loop.

### Logical Soundness of the Closed Loop

The system forms a valid reinforcement learning closed loop. Here is the logical verification:

| Component | RL Formalism | YiCeNet Implementation |
|-----------|-------------|----------------------|
| **State** | Observable representation of the environment | Six-probes ℝ⁹ (h_norm, h_entropy, logit_entropy, clan_upper/lower/opposite, q_gap, jump_distance, action_confidence) |
| **Action** | Discrete choice from action space | Hexagram index (0-63) via Gumbel Router |
| **Reward** | Scalar feedback signal | WM-predicted distribution similarity to actual outcome (cosine → [0,1]) |
| **Policy** | π(a\|s) — probability of action given state | Gumbel-Softmax over router logits |
| **Value** | V(s) or Q(s,a) — expected future reward | Value Network scores candidate hexagrams |
| **Transition** | s → s' with reward r | Next-turn probes extracted after user responds |

**Why it is valid:**

1. **Reward is grounded in real user behavior**, not synthetic simulation. Every reward traces back to an actual human decision: continue, correct, praise, or abandon.

2. **The World Model prevents reward hacking.** Since the WM is trained on ground-truth signals (user behavior), not YiCeNet's own judgments, optimizing against WM predictions means optimizing against a learned model of actual user satisfaction — not against a self-referential target.

3. **The endogenous weight mechanism prevents overfitting.** When the WM encounters a probe pattern it cannot predict well (high KL surprise), that sample is automatically de-weighted. This means outlier or noisy interactions don't distort training.

4. **The flywheel prevents distributional shift.** As YiCeNet improves, the distribution of probes changes. But the flywheel continuously retrains both the WM and YiCeNet on the latest session data, so both models co-evolve with the shifting data distribution.

5. **The system degrades gracefully.** If the WM is unavailable (no checkpoint), the flywheel falls back to raw heuristic signals. If the flywheel has too few samples (<20), it defers training. If the registry is missing, the engine uses a default checkpoint. Every component has a fallback path.

**One inherent limitation** (not a logical flaw): the reward is always **delayed by one user turn**. The system cannot know if a hexagram decision was good until the user responds. This is not a bug — it's the nature of learning from interaction. The World Model partially mitigates this by learning to predict outcomes from probes, effectively creating a "simulated next turn" for training. But the ground truth always comes from the user.

### Two Training Paths

| | Flywheel Path (every 6h cron) | API Path (manual trigger) |
|---|---|---|
| **Signal source** | `external_metrics.py` regex heuristics | DeepSeek/OpenAI LLM evaluation |
| **Cost** | Zero (fully local) | API call fees |
| **Signal quality** | Coarse (regex matches surface text only) | High (LLM understands semantics) |
| **Use case** | Continuous autonomous evolution | Batch quality evaluation, v15 training |
| **Training targets** | World Model + YiCeNet | World Model (supervised) + YiCeNet (RL) |
| **Satisfaction** | Discrete (-1.0, -0.5, 0.3, 0.5, 1.0) | Continuous (e.g., 0.37, -0.82) |
| **Per-sample reasoning** | None | One-sentence LLM explanation |

Both paths converge on the same World Model checkpoint (`world_model_best.pt`), which serves as the unified reward proxy for subsequent RL fine-tuning.

---

*YiCeNet — 5.6M parameters · 4ms · Fully local · Continuously evolving*
