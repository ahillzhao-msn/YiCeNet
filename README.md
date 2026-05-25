# ☯ YiCeNet (易策网络)

> **5.6M parameters · 4ms inference · Fully local · Continuously evolving**
>
> An I-Ching inspired neural network for fast, explainable orchestration decisions between tools, agents, and workflows.

<p align="center">
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10+-blue.svg">
  <img alt="Params" src="https://img.shields.io/badge/Params-5.6M-green.svg">
  <img alt="Inference" src="https://img.shields.io/badge/Inference-4ms-red.svg">
  <img alt="Hexagrams" src="https://img.shields.io/badge/Hexagrams-64/64-purple.svg">
</p>

---

**易有三义：变易、不易、简易。**

- **变易** — The model evolves through use; every interaction is a training signal
- **不易** — The 64 hexagrams are eternal meta-patterns, structurally stable
- **简易** — 4ms to make a call; complexity reduced to a decision

---

## What Is YiCeNet

YiCeNet translates the 4,000-year-old I-Ching (易经) hexagram mutation system into a tiny neural architecture. Given a user intent, it maps to one of 64 hexagrams and produces an **explainable, millisecond-fast orchestration decision**.

It is **not** a general-purpose LLM. It does one thing: *understand your intent and deliver an interpretable scheduling decision in milliseconds.*

### Comparison

| | GPT-4 / Claude | YiCeNet |
|---|---|---|
| Parameters | 100B–1T+ | **5.6M** |
| Inference | Seconds | **4ms** |
| Runtime | Cloud GPU clusters | **Laptop / edge device** |
| Privacy | Data uploaded to cloud | **Fully local** |
| Personalization | Prompt engineering | **RL fine-tuned into weights** |
| Shareability | Share prompts | **Share the model's "personality"** |
| Decision philosophy | Statistical smoothing | **64-hexagram structural reasoning** |

---

## How It Works

### Architecture (6 layers)

```
Intent encoding     ──── Task → 6D vector
    ↓
Gumbel router       ──── Discrete sampling → hexagram ID
    ↓
Hexagram embedding  ──── 64×256 structured features
    ↓
Mutation engine     ──── Opposition / Overlap / Core / Shift operations
    ↓
Policy decoder      ──── Hexagram → dispatch instruction
    ↓
World model         ──── Micro network evaluating quality + providing feedback
```

The Gumbel-Softmax router maps task embeddings to one of 64 hexagrams. Each hexagram carries a learned embedding (256-dim) that captures the "personality" of that decision archetype. The mutation engine applies four I-Ching structural operators — opposition (错), overlap (综), core (互), and shift (变) — transforming the selected hexagram into a chain of related states. The policy decoder reads this chain and outputs the orchestration plan.

### Three-Stage Training

| Stage | Purpose | Data |
|-------|---------|------|
| **Pretraining** | Build universal pattern recognition across 64 hexagrams | 10K synthetic traces |
| **World model** | Learn which hexagram is "good" in which context | Human feedback signal |
| **RL fine-tuning** | Personalize — distill user's decision style into weights | Real interaction data |

### Flywheel: Online Evolution

The model never stops at training time:

```
Use → Collect feedback → Fine-tune world model → Update policy → Knows you better
↑___________________________________________________________________________|
```

Each user interaction becomes a training signal. The model continuously adapts to your decision patterns — fully on-device.

---

## Quick Start

```bash
# Clone
git clone https://github.com/ahillzhao-msn/YiCeNet.git
cd YiCeNet

# Editable install
pip install -e .

# Demo
python demo.py
```

### Basic Usage

```python
from yicenet import YiCeNetEngine

engine = YiCeNetEngine()

result = engine.predict(
    task="Analyze sales data and generate a visualization report",
    available_agents=["data_analyzer", "chart_generator", "report_writer"],
)

print(f"Agent sequence: {result.agent_sequence}")
print(f"Reasoning path: {result.winning_path}")  # interpretable hexagram trace
print(f"Latency: {result.latency_ms}ms")          # ~4ms
```

### Training

```bash
# API-supervised RL training
python scripts/rl_train.py \
  --version v16 \
  --buffer data/flywheel_buffer.jsonl \
  --eval-results data/ds_eval_all.jsonl \
  --endogenous

# Evaluate new samples via OpenAI-compatible API
python scripts/eval_api.py \
  --input samples.jsonl \
  --output evaluations.jsonl \
  --batch-size 20

# Manage checkpoints
python scripts/checkpoint_manager.py prune
python scripts/checkpoint_manager.py register v16 path/to/model.pt 0.99
```

---

## Performance

| Version | Samples | Unique Hexagrams | Confidence | Noise Adaptation |
|---------|---------|:----------------:|:----------:|:----------------:|
| v4 | 10K synthetic | 48/64 | 0.708 | None |
| v6 | 200 real | 38/64 | 0.981 | None |
| **v15** | **997 real** | **58/64** | **0.966** | **2 layers** |

---

## Configuration (Environment Variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `EVAL_API_URL` | `https://api.deepseek.com/v1/chat/completions` | Evaluation API endpoint |
| `EVAL_MODEL` | `deepseek-chat` | Evaluation model name |
| `EVAL_API_KEY` | (env or .env) | Evaluation API key |
| `YICENET_HOME` | auto-detected | Override project root |

---

## Project Layout

```
YiCeNet/
├── src/yicenet/           # Core library (pip install -e .)
│   ├── model.py           # YiCeNet: TinyEncoder → Gumbel Router → Decoder
│   ├── encoder.py         # TinyEncoder (4-layer Transformer, 256-dim)
│   ├── decoder.py         # Action Decoder (26K params)
│   ├── hexagram.py        # Hexagram types, family classification
│   ├── world_model.py     # WorldModelV2: prediction → endogenous weight
│   ├── value_net.py       # Value Network (41K params)
│   ├── yicenet_engine.py  # Unified inference API
│   ├── flywheel.py        # Online flywheel training pipeline
│   ├── config.py          # Configuration + hyperparameters
│   ├── constants.py       # 64 hexagrams, I-Ching constants
│   ├── tokenizer.py       # Qwen BPE tokenizer wrapper
│   └── hermes_tool.py     # Agent tool integration
├── scripts/               # Training & evaluation CLI
├── tests/                 # Test suite
├── data/                  # Training data (gitignored)
├── checkpoints/           # Model weights (gitignored)
├── pyproject.toml         # Build config
├── ARCHITECTURE.md        # Full architecture documentation
├── MANIFESTO.md           # Project philosophy & vision
└── INSTALL.md             # Installation guide
```

---

## What It Is and Isn't

### ✅ It Is

- An **ultra-lightweight meta-scheduler** — fast routing decisions between agents
- A **self-evolving personal AI** — the flywheel makes it smarter with use
- An **explainable decision engine** — every decision has a traceable hexagram path
- A **fully local privacy fortress** — your data never leaves your device
- A **sharable experience carrier** — package your decision style into a file, share it

### ❌ It Isn't

- Not a chatbot (use ChatGPT)
- Not a code generator (use Copilot)
- Not a replacement for LLMs (it collaborates with them)
- Not fortune-telling (it's for agent orchestration)

---

## Vision

A future where thousands of personal models form a **forest** — each with its own roots, niche, and growth pattern. Not one model serving everyone, but diverse, local, self-evolving models that complement each other.

> Foundation model era was agriculture — centralized, standardized, high-yield but monoculture.
> The next era is **forest** — decentralized, diverse, self-evolving, regenerative.

---

## License

[MIT](LICENSE) © ahillzhao-msn

---

```
        ☰  ☷  ☳   ☴  ☵  ☲  ☶  ☱
        乾  坤  震  巽  坎  离  艮  兑
       天  地  雷  风  水  火  山  泽

  5.6M params · 4ms · Fully local · Evolving
```

⭐ Star if this resonates · 🔱 Fork to build your own · 🔥 [Share your vision](https://github.com/ahillzhao-msn)
