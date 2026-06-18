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
| mcp | >= 1.0 | MCP server (FastMCP for Claude Code / IDE integration) |

## Install (Wheel Release)

YiCeNet ships as a Python wheel via GitHub Releases. No source tree needed.

```bash
# 1. Download the .whl from latest release
#    https://github.com/ahillzhao-msn/YiCeNet/releases/latest

# 2. Install into your venv
pip install YiCeNet-15.4.0-py3-none-any.whl

# 3. Verify
python -c "import yicenet; print(yicenet.__version__)"
# → 15.4.0

# 4. Initialize data root + config + SOUL + target registration
yicenet-bootstrap --auto
# 或指定自定义 SOUL 模板：
yicenet-bootstrap --auto --soul ~/LOOM/SOUL-template.md
```

What `yicenet-bootstrap --auto` does:

| Phase | Action |
|:-----:|--------|
| 1 | Detect target environments (Hermes, Claude Code, PyTorch) |
| 2 | Install YiCeNet to target venv (pip install in editable mode) |
| 3 | Verify dependencies |
| 4 | Download model checkpoints |
| **5** | **Create ~/.yicenet/ + config.yaml + SOUL.md** |
| 6 | Register Hermes tools / Claude Code MCP server |
| 7 | Register flywheel cron |

### Resulting file tree:

```
~/.yicenet/
├── config.yaml          # Runtime-tunable parameters (Flywheel, Inference, SOUL)
├── SOUL.md              # YiCeNet identity — the "易之魂"
├── checkpoints/         # Model weights (registry.json + .pt files)
├── data/                # Flywheel training buffer
└── logs/                # Training logs
```

## Install (Source — Development)

```bash
git clone https://github.com/ahillzhao-msn/YiCeNet.git
cd YiCeNet
pip install -e .
yicenet-bootstrap --auto           # Same initialization
```

## Upgrading

> **Stop the MCP server before upgrading.**
>
> When YiCeNet runs as an MCP server (`yicenet-serve`), the host application
> (Claude Code, any MCP-compatible IDE or client) holds an open file lock on
> the `yicenet-serve` executable for the entire session. `pip install
> --force-reinstall` will fail with a file-in-use error if the process is
> still running, and may leave the package partially uninstalled.

**Steps:**

1. Stop the MCP server by closing the host application, or terminate the
   process directly:

   ```powershell
   # Windows
   taskkill /f /im yicenet-serve.exe
   ```

   ```bash
   # Linux / macOS
   pkill -f yicenet-serve
   ```

2. Install the new wheel:

   ```bash
   pip install YiCeNet-X.Y.Z-py3-none-any.whl --force-reinstall
   ```

3. Restart the host application. The MCP server will pick up the new version
   automatically on next launch.

**If the install already failed** (partial uninstall, `No module named 'yicenet'`):
stop `yicenet-serve` as above, then re-run step 2.

## Uninstall

```bash
# 1. Remove target registrations + (optionally) all data
yicenet-uninstall                  # Removes Hermes plugin + Claude Code MCP entry
yicenet-uninstall --clean-data     # + deletes ~/.yicenet/ entirely

# 2. Close all Claude Code sessions first (see Upgrading above), then:
pip uninstall yicenet -y
```

## SOUL Template Integration

YiCeNet has its own identity template (`SOUL-template.md`) that defines
its core philosophy — 心·道 / 骨·法 / 皮·儒 / 用·兵 — mapped to the
64-hexagram routing framework.

During `yicenet-bootstrap`, `~/.yicenet/SOUL.md` is created from this template.
You can customize it before bootstrap:

```bash
# Use your custom SOUL during initialization
yicenet-bootstrap --soul /path/to/your/SOUL.md
```

The SOUL influences YiCeNet in two ways:
1. **Hexagram prior** — Each SOUL layer (道/法/儒/兵) biases the router
   toward structurally aligned hexagrams
2. **Environment injection** — SOUL summaries are injected into
   Hermes system prompts / Claude Code context via the configured
   `injection_level` in `~/.yicenet/config.yaml`

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

### Optional: Hermes Agent Plugin Integration (Recommended)

For zero-effort setup where YiCeNet runs on **every turn without explicit tool calls**,
install the Hermes plugin. It wires all 7 hooks as native lifecycle callbacks:

```bash
# From the YiCeNet repo
bash scripts/install/install-yicenet-hooks.sh
```

This replaces the manual symlink approach below. The plugin:
- Auto-injects hexagram context into every LLM call (pre_llm_call)
- Calibrates tool direction against hexagram (pre_tool_call, observe only)
- Sends tool-level reward signals (post_tool_call)
- Accumulates first-hand token usage data (post_api_request)
- Writes reward signals to flywheel training buffer (post_llm_call)
- Works standalone **or** alongside loom-hooks (auto-skip via _loom_hooks_active)
- Self-suppresses when LOOM is installed — no duplicate predictions

**Requirements:** Hermes Agent, YiCeNet pip-installed (`pip install -e ~/YiCeNet`).

### Legacy: Manual Tool Symlink

If you prefer not to use the plugin, you can manually symlink the tool:

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
    "version": "v18",
    "path": "yicenet_v18.pt"
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
