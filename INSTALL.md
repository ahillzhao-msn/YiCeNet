# YiCeNet Installation Guide

## System Dependencies

### Core (mandatory)

YiCeNet itself is a standalone PyTorch model with NO external runtime dependencies beyond pip packages:

| Dependency | Version | Purpose |
|-----------|---------|---------|
| Python | >= 3.10 | Runtime |
| torch | >= 2.0 | Neural network engine |
| numpy | >= 1.24 | Numerical operations |
| transformers | >= 4.35 | Qwen BPE tokenizer |
| sentencepiece | >= 0.1 | Tokenizer backend |
| tqdm | >= 4.60 | Progress bars |
| huggingface-hub | >= 0.20 | Tokenizer download |
| httpx | >= 0.25 | HTTP client (flywheel Hermes DB access) |

Install via pip (pyproject.toml):
```bash
pip install -e ~/YiCeNet
```

### Optional: LLM API (training / evaluation)

`scripts/eval_api.py` and `scripts/rl_train.py` use an OpenAI-compatible API for reward scoring.
Any provider that supports the chat completions format works (DeepSeek, OpenAI, Anthropic via proxy, local llama.cpp, etc.).

**Does NOT require llama.cpp, a local web server, or any running service.**

API configuration:
| Env Variable | Default | Example |
|-------------|---------|---------|
| `EVAL_API_URL` | `https://api.deepseek.com/v1/chat/completions` | `http://localhost:8000/v1/chat/completions` |
| `EVAL_MODEL` | `deepseek-chat` | `gpt-4o-mini`, `qwen2.5-32b` |
| `EVAL_API_KEY` | (see below) | `sk-your-key-here` |

Key resolution order:
1. `EVAL_API_KEY` environment variable
2. `DEEPSEEK_API_KEY` environment variable (backward compatible)
3. `~/YiCeNet/.env` file (project-level)
4. `~/.hermes/.env` file (Hermes Agent)

```
# .env file format:
EVAL_API_KEY=sk-your-key-here
# Or legacy:
DEEPSEEK_API_KEY=sk-your-key-here
```

To use a different provider:
```bash
# OpenAI
export EVAL_API_URL=https://api.openai.com/v1/chat/completions
export EVAL_MODEL=gpt-4o-mini
export EVAL_API_KEY=sk-...

# Local llama.cpp
export EVAL_API_URL=http://localhost:8000/v1/chat/completions
export EVAL_MODEL=qwen2.5-7b-instruct
export EVAL_API_KEY=not-needed
```

### Optional: Hermes Agent Integration

When installed as a Hermes Agent tool:

1. Install the package:
```bash
pip install -e /path/to/YiCeNet
```

2. Symlink the Hermes tool:
```bash
ln -sf /path/to/YiCeNet/src/yicenet/hermes_tool.py ~/.hermes/hermes-agent/tools/yicenet_tool.py
```

3. Restart Hermes. The `yicenet_predict` and `yicenet_switch` tools appear in the `file` toolset.

### Optional: Autonomous Flywheel (Continuous Learning)

The flywheel in `src/yicenet/flywheel.py` can be triggered via any scheduler:

```bash
# Via cron (every 12 hours):
0 */12 * * * cd ~/YiCeNet && python3 -m yicenet.flywheel >> logs/flywheel.log 2>&1

# Via Hermes cron (if monitoring Hermes session DB):
# Scheduled through ~/.hermes/SOUL.md 卦链 step: flywheel_every_12h
```

No systemd service, web server, or long-running daemon is required. The flywheel runs on-demand and exits.

## Configuration

### Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `EVAL_API_KEY` | For training | — | API key for reward scoring (also reads `DEEPSEEK_API_KEY`) |
| `EVAL_API_URL` | Optional | DeepSeek | OpenAI-compatible endpoint URL |
| `EVAL_MODEL` | Optional | `deepseek-chat` | Model name for evaluation |
| `YICENET_HOME` | Optional | auto-detected | Override project root (when installed as pip package) |

### Path Convention

All `.pt` checkpoint paths in `checkpoints/registry.json` are stored **relative** to the `checkpoints/` directory.
This ensures portability across machines. Example:

```json
{
  "active": {
    "version": "v15",
    "path": "yicenet_v15.pt"
  }
}
```

The registry is managed by `scripts/checkpoint_manager.py`:
```bash
python scripts/checkpoint_manager.py fresh   # rebuild from existing .pt files
python scripts/checkpoint_manager.py clean   # validate paths
python scripts/checkpoint_manager.py prune   # remove low-score checkpoints
```

## Project Layout (Portable)

```
YiCeNet/
├── pyproject.toml          # Build config (pip install -e .)
├── src/yicenet/            # Core library
│   ├── hermes_tool.py      # → symlink to Hermes tools/
│   └── flywheel.py         # Optional auto-training
├── scripts/                # CLI training/evaluation
├── checkpoints/            # Model weights (gitignored)
│   ├── registry.json       # Relative paths only
│   └── *.pt                # Generated artifacts
├── data/                   # Training data (gitignored)
└── .env                    # API keys (gitignored, optional)
```

The project has zero hardcoded absolute paths. Root resolution is:
- Scripts: relative to `Path(__file__).parent.parent`
- Installed package: resolved via `yicenet.__file__`
- Override: `YICENET_HOME` env var

## Quick Start

```bash
# 1. Install
pip install -e /path/to/YiCeNet

# 2. Verify
python3 -c "import yicenet; print(yicenet.__version__)"

# 3. Run inference
python3 -c "
from yicenet.model import YiCeNet
from yicenet.config import YiCeNetConfig
model = YiCeNet(YiCeNetConfig())
print(f'Model ready: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params')
"
```
