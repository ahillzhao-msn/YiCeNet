# ☯ YiCeNet（易策网络）

> 560 万参数 · 4 毫秒决策 · 完全本地 · 持续演进
> 5.6M params · 4ms inference · Fully local · Continuously evolving

---

[**English**](#english)

---

<a id="中文"></a>

## 这是什么

YiCeNet 是一个受《易经》启发的微型神经网络——将六十四卦的卦变推演转化为计算架构，在工具、Agent、工作流之间做最快的编排决策。

它不是通用大模型，不试图装下整个世界。它只做一件事：**理解你的意图，在毫秒级给出一个可解释的调度决策。**

### 你不是在跟另一个大模型比较

| | GPT-4 / Claude | YiCeNet |
|---|---|---|
| 参数量 | 千亿~万亿 | **560 万** |
| 推理延迟 | 秒级 | **4 毫秒** |
| 运行环境 | 云端 GPU 集群 | **笔记本 / 手机** |
| 隐私 | 数据上传云端 | **完全本地** |
| 个性化 | Prompt 工程模拟 | **RL 微调进权重** |
| 可分享 | 分享 Prompt | **分享整个模型的"人格"** |
| 决策哲学 | 统计平滑 | **六十四卦结构推演** |

不是替代，是补完。不是对抗，是分工。

---

## 设计哲学

### 道 · 以易为思

「易」有三义：变易、不易、简易。

- **变易**：模型在使用中持续进化——每一次交互都是训练信号
- **不易**：六十四卦作为固化的元模式，结构稳定不变
- **简易**：4 毫秒给出一个判断，决策化繁为简

### 法 · 三阶训练

```
阶段一：预训练         → 建立 64 卦的通用模式识别
阶段二：世界模型训练   → 学习什么卦象在什么上下文中是"好"的
阶段三：RL 微调        → 个性化——将你的决策风格蒸馏进权重
```

### 术 · 六层架构

```
意图编码层 ──── 任务 → 6 维向量
    ↓
Gumbel 路由器 ── 离散采样 → 卦象 ID
    ↓
卦象嵌入表 ──── 64×256 结构化特征
    ↓
卦变推演层 ──── 错 / 综 / 互 / 变 操作
    ↓
策略解码层 ──── 卦象 → Agent 调度指令
    ↓
世界模型 ────── 微型网络，评估 + 反馈
```

### 飞轮：在线持续进化

模型不会止步于训练结束那一刻。它在你使用它的过程中持续进化。

```
使用 → 收集反馈 → 微调世界模型 → 更新策略 → 更懂你
↑____________________________________________|
```

---

## 功能状态

| 功能 | 状态 | 源码位置 |
|------|------|----------|
| TinyEncoder（4 层 Transformer） | ✅ | `src/yicenet/model.py` |
| Gumbel 路由器（64 卦选择） | ✅ | `src/yicenet/model.py` |
| 双头世界模型 v2 | ✅ | `src/yicenet/world_model.py` |
| 幂律遗忘曲线 | ✅ | `src/yicenet/world_model.py` |
| API 监督 RL 训练 | ✅ | `scripts/rl_train.py` |
| 内源噪声加权 | ✅ | `src/yicenet/world_model.py` → `src/yicenet/flywheel.py` |
| 热切换检查点注册 | ✅ | `scripts/checkpoint_manager.py` |
| 每6小时飞轮自学习 | ✅ | `src/yicenet/flywheel.py` |
| 采样分层（计划中） | ⏳ | 未实现 |

---

## 快速开始

```bash
# 克隆
git clone https://github.com/ahillzhao-msn/YiCeNet.git
cd YiCeNet

# 可编辑安装
pip install -e .

# 演示
python demo.py
```

### 基本用法

```python
from yicenet.model import YiCeNetEngine

engine = YiCeNetEngine()

result = engine.predict(
    task="分析销售数据并生成可视化报告",
    available_agents=["data_analyzer", "chart_generator", "report_writer"]
)

print(f"推荐 Agent 序列: {result.agent_sequence}")
print(f"推理路径: {result.winning_path}")   # 可解释的卦象推演链
print(f"推理耗时: {result.latency_ms}ms")   # ~4ms
```

### 训练脚本

```bash
# API 监督训练
python scripts/rl_train.py \
  --version v16 \
  --buffer data/flywheel_buffer.jsonl \
  --eval-results data/ds_eval_all.jsonl \
  --endogenous

# 用 OpenAI 兼容 API 评估新样本
# 环境变量: EVAL_API_URL=... EVAL_MODEL=... EVAL_API_KEY=***
python scripts/eval_api.py \
  --input samples.jsonl \
  --output evaluations.jsonl \
  --batch-size 20

# 管理检查点
python scripts/checkpoint_manager.py prune       # 清除低分检查点
python scripts/checkpoint_manager.py clean        # 验证 registry.json
python scripts/checkpoint_manager.py register v16 path/to/model.pt 0.99
```

---

## 性能概览

| 版本 | 样本数 | 唯一卦象 | 置信度 | 噪声适应 |
|------|--------|:--------:|:------:|:--------:|
| v4 | 10K 合成 | 48/64 | 0.708 | 无 |
| v6 | 200 真实 | 38/64 | 0.981 | 无 |
| **v15** | **997 真实** | **58/64** | **0.966** | **2 层去噪** |

每次飞轮自动运行后，新版本注册到 `checkpoints/registry.json`，通过热切换无缝升级。

---

## 配置（环境变量）

| 变量 | 默认值 | 用途 |
|------|--------|------|
| `EVAL_API_URL` | `https://api.deepseek.com/v1/chat/completions` | 评估 API 端点 |
| `EVAL_MODEL` | `deepseek-chat` | 评估模型名称 |
| `EVAL_API_KEY` | (环境或 .env) | 评估 API 密钥 |
| `DEEPSEEK_API_KEY` | (后备) | 旧兼容：替代 EVAL_API_KEY |
| `YICENET_HOME` | 自动检测 | 覆盖项目根目录 |

---

## 核心原则

**诚 · 直** — 文档只描述代码已实现的内容。计划中的设计明确标注为 ⏳。

**知之为知之，不知为不知** — 模型自身的预测惊讶度（KL 散度）是其知识边界的精确度量。越界样本自动降权。

**降噪即训练，训练即降噪** — 去噪行为是训练动态的自然产物，非外部预处理。

---

## 项目布局

```
YiCeNet/
├── src/yicenet/         # 核心库（可编辑安装包）
├── scripts/             # 训练 / 评估 CLI 脚本
├── data/                # 训练数据 & 飞轮缓冲区
├── checkpoints/         # 模型权重 & 注册表（gitignored）
├── docs/                # 文档
├── tests/               # 测试
├── pyproject.toml       # 构建 & 依赖配置
├── ARCHITECTURE.md      # 完整架构文档
├── INSTALL.md           # 详细安装指南
└── MANIFESTO.md         # 个人 AI 宣言
```

---

## 项目文档

| 文档 | 内容 |
|------|------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 完整架构与组件说明 |
| [INSTALL.md](INSTALL.md) | 详细安装与配置指南 |
| [MANIFESTO.md](MANIFESTO.md) | 个人 AI 宣言——这个项目为何存在 |

---

## 什么是它，什么不是

### ✅ 它是

- 一个极轻量的**元调度器**——在 Agent 之间做最快的路由决策
- 一个**可自我进化**的个人 AI——飞轮机制让它越用越懂你
- 一个**可解释**的决策引擎——每个决策都有对应的卦象推演路径
- 一个完全**本地**的隐私堡垒——数据永不离开你的设备
- 一个**可分享的经验载体**——将你的决策风格打包成文件，分享给他人

### ❌ 它不是

- 不是一个通用聊天机器人（去用 ChatGPT）
- 不是一个代码生成器（去用 Copilot）
- 不是一个大模型的替代品（它是大模型的协作者）
- 不是用来算命的（严肃地说，我们用它做 Agent 编排）

---

## 愿景：从"我的模型"到"我们的生态"

想象这样一幅图景：

一个资深项目经理。多年的工作中，她的 YiCeNet 已经吸收了她的决策风格——如何拆解任务、如何评估风险、如何在不确定中做出判断。不是聊天记录，而是**决策人格**。

她把这个几十 MB 的模型文件，分享给团队的年轻成员。他们加载它，就获得了她的隐性知识——那些从未写在文档里，却决定了无数次项目成败的判断力。

这不是知识的分享。这是思维风格的传承。

当成千上万个这样的个人模型涌现——不是一个大模型在服务所有人，而是一片**模型森林**——每棵树都有自己的根系，但共同形成生态。

你的经验 + 我的经验 + 她的经验 → **社会层面对 AI 的有机整合**。

**基座大模型的时代，是农耕文明。下一个时代，是森林。**

---

## 贡献

这是一个开源的个人 AI 项目。我们欢迎：

- **训练你自己的 YiCeNet**：基于你的个人数据微调，形成你的专属模型
- **分享你的发现**：哪种卦象映射策略在你的领域最有效？
- **贡献代码**：改进架构、优化训练流程、增加新的卦变操作
- **理念探讨**：易经哲学与现代 AI 的深度融合，还有哪些可能？

请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

---

## 许可证

[MIT](LICENSE) © ahillzhao-msn

```
        ☰  ☷  ☳   ☴  ☵  ☲  ☶  ☱
        乾  坤  震  巽  坎  离  艮  兑
        天  地  雷  风  水  火  山  泽
```

⭐ 如果这个项目触动了你，请给一颗星。
🔱 如果你想拥有自己的模型，请 Fork 并训练。
🔥 如果你看到同样的未来，[请联系我](https://github.com/ahillzhao-msn)。


---
 [**中文版**](#中文) 

<a id="english"></a>

## What Is YiCeNet

YiCeNet is an I-Ching-inspired tiny neural network (~5.6M params, 22MB) that maps user intent to one of 64 hexagrams for fast, explainable orchestration decisions between tools, agents, and workflows.

It is not a general-purpose LLM. It does one thing: **understand your intent and deliver an interpretable scheduling decision in milliseconds.**

### Not Another Foundation Model

| | GPT-4 / Claude | YiCeNet |
|---|---|---|
| Parameters | 100B~1T+ | **5.6M** |
| Inference latency | Seconds | **4ms** |
| Runtime | Cloud GPU clusters | **Your laptop / phone** |
| Privacy | Data uploaded to cloud | **Fully local** |
| Personalization | Prompt engineering | **RL fine-tuned into weights** |
| Shareability | Share prompts | **Share the model's "personality"** |
| Decision philosophy | Statistical smoothing | **64-hexagram structural reasoning** |

Not replacement — complement. Not confrontation — division of labor.

---

## Design Philosophy

### Tao · Thinking in Hexagrams

Yi (易) has three meanings: change, constancy, simplicity.

- **Change**: The model evolves through use — every interaction is a training signal
- **Constancy**: The 64 hexagrams are eternal meta-patterns, structurally stable
- **Simplicity**: 4ms to make a call — complexity reduced to a decision

### Method · Three-Stage Training

```
Stage 1: Pretraining        → Build universal pattern recognition across 64 hexagrams
Stage 2: World Model        → Learn which hexagram is "good" in which context
Stage 3: RL Fine-tuning     → Personalization — distill your decision style into weights
```

### Architecture · Six Layers

```
Intent Encoder     ──── Task → 6D vector
    ↓
Gumbel Router      ──── Discrete sampling → hexagram ID
    ↓
Hexagram Embedding ──── 64×256 structured features
    ↓
Mutation Engine    ──── Opposition / Overlap / Core / Shift operations
    ↓
Policy Decoder     ──── Hexagram → agent dispatch instruction
    ↓
World Model        ──── Micro network, evaluation + feedback
```

### Flywheel: Continuous Online Evolution

The model never stops at training time. It evolves as you use it.

```
Use → Collect feedback → Fine-tune world model → Update policy → Knows you better
↑________________________________________________________________________|
```

---

## Feature Status

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

---

## Quick Start

```bash
git clone https://github.com/ahillzhao-msn/YiCeNet.git
cd YiCeNet
pip install -e .
python demo.py
```

### Basic Usage

```python
from yicenet.model import YiCeNetEngine

engine = YiCeNetEngine()

result = engine.predict(
    task="Analyze sales data and generate a visualization report",
    available_agents=["data_analyzer", "chart_generator", "report_writer"]
)

print(f"Recommended agent sequence: {result.agent_sequence}")
print(f"Reasoning path: {result.winning_path}")  # interpretable hexagram trace
print(f"Latency: {result.latency_ms}ms")          # ~4ms
```

### CLI Training

```bash
# API-supervised RL training
python scripts/rl_train.py \
  --version v16 \
  --buffer data/flywheel_buffer.jsonl \
  --eval-results data/ds_eval_all.jsonl \
  --endogenous

# Evaluate new samples via OpenAI-compatible API
# Env: EVAL_API_URL=... EVAL_MODEL=... EVAL_API_KEY=***
python scripts/eval_api.py \
  --input samples.jsonl \
  --output evaluations.jsonl \
  --batch-size 20

# Manage checkpoints
python scripts/checkpoint_manager.py prune       # purge low-score checkpoints
python scripts/checkpoint_manager.py clean        # validate registry.json
python scripts/checkpoint_manager.py register v16 path/to/model.pt 0.99
```

---

## Performance Summary

| Version | Samples | Unique Hexagrams | Confidence | Noise Adaptation |
|---------|---------|:----------------:|:----------:|:----------------:|
| v4 | 10K synthetic | 48/64 | 0.708 | None |
| v6 | 200 real | 38/64 | 0.981 | None |
| **v15** | **997 real** | **58/64** | **0.966** | **2 layers** |

After each flywheel run, new versions register in `checkpoints/registry.json` and upgrade seamlessly via hot-swap.

---

## Configuration (Environment Variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `EVAL_API_URL` | `https://api.deepseek.com/v1/chat/completions` | Evaluation API endpoint |
| `EVAL_MODEL` | `deepseek-chat` | Evaluation model name |
| `EVAL_API_KEY` | (env or .env) | Evaluation API key |
| `DEEPSEEK_API_KEY` | (fallback) | Legacy compat: replaces EVAL_API_KEY |
| `YICENET_HOME` | auto-detected | Override project root |

---

## Core Principles

**诚·直 (Sincerity · Directness)** — Documentation describes only what code implements. Planned features are clearly marked as ⏳.

**知之为知之，不知为不知** — The model's own prediction surprise (KL divergence) is the measure of its knowledge boundary. Out-of-bound samples are naturally de-weighted.

**降噪即訓練，訓練即降噪** — Denoising emerges from training dynamics, not preprocessing.

---

## Project Layout

```
YiCeNet/
├── src/yicenet/         # Core library (editable package)
├── scripts/             # Training & evaluation CLI scripts
├── data/                # Training data & flywheel buffer
├── checkpoints/         # Model weights & registry (gitignored)
├── docs/                # Documentation
├── tests/               # Tests
├── pyproject.toml       # Build & dependency config
├── ARCHITECTURE.md      # Full architecture documentation
├── INSTALL.md           # Detailed installation guide
└── MANIFESTO.md         # Personal AI manifesto
```

---

## Project Documents

| Document | Content |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Full architecture & component docs |
| [INSTALL.md](INSTALL.md) | Setup & configuration guide |
| [MANIFESTO.md](MANIFESTO.md) | Personal AI manifesto — why this project exists |

---

## What It Is and Isn't

### ✅ It Is

- An **ultra-lightweight meta-scheduler** — fastest routing decisions between agents
- A **self-evolving personal AI** — flieswheel makes it smarter with use
- An **explainable decision engine** — every decision has a traceable hexagram path
- A **fully local privacy fortress** — your data never leaves your device
- A **sharable experience carrier** — package your decision style into a file, share it

### ❌ It Isn't

- Not a chatbot (use ChatGPT)
- Not a code generator (use Copilot)
- Not a replacement for LLMs (it collaborates with them)
- Not fortune-telling (seriously — it's for agent orchestration)

---

## Vision: From "My Model" to "Our Ecosystem"

Imagine this:

A senior project manager. Over years of work, her YiCeNet has absorbed her decision style — how she decomposes tasks, evaluates risk, makes judgment calls under uncertainty. Not chat logs — a **decision personality**.

She shares this ~30MB model file with junior team members. They load it and get her tacit knowledge — the judgment that decided countless project outcomes but was never written down.

This is not knowledge sharing. This is **mindset inheritance**.

When thousands of such personal models emerge — not one giant model serving everyone, but a **forest of models** — each tree with its own roots, but together forming an ecosystem.

Your experience + my experience + her experience → **organic integration of AI at the societal level**.

**The foundation model era was agriculture. The next era is the forest.**

---

## Contributing

This is an open-source personal AI project. We welcome:

- **Train your own YiCeNet**: Fine-tune on your personal data
- **Share your findings**: Which hexagram mapping strategy works best in your domain?
- **Contribute code**: Improve architecture, optimize training, add new mutation operations
- **Philosophical discussions**: How deep can the integration of I-Ching philosophy and modern AI go?

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## License

[MIT](LICENSE) © ahillzhao-msn

```
        ☰    ☷     ☳     ☴    ☵    ☲     ☶     ☱
       Qian  Kun   Zhen   Xun   Kan   Li    Gen    Dui
       Sky  Earth Thunder Wind Water Fire Mountain Lake
```

⭐ If this project resonates, give it a star.
🔱 If you want your own model, fork and train it.
🔥 If you see the same future, [reach out](https://github.com/ahillzhao-msn).
