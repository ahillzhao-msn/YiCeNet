# YiCeNet (易策网络)

## Project Structure

```
YiCeNet/
├── src/                    # Package root (editable install)
│   ├── __init__.py         # Package description
│   └── yicenet/            # Core library
│       ├── __init__.py
│       ├── config.py       # YiCeNetConfig + hyperparameters
│       ├── constants.py    # 64 hexagrams, I-Ching constants
│       ├── model.py        # YiCeNet: TinyEncoder → Gumbel Router → Decoder
│       ├── encoder.py      # TinyEncoder (4-layer Transformer, 256-dim)
│       ├── decoder.py      # Action Decoder (26K params)
│       ├── hexagram.py     # Hexagram types, family classification
│       ├── tokenizer.py    # Qwen BPE tokenizer wrapper
│       ├── world_model.py  # WorldModelV2: prediction → endogenous weight
│       ├── value_net.py    # Value Network (41K params)
│       ├── yicenet_engine.py  # YiCeNetEngine: unified inference API
│       ├── hermes_tool.py  # Hermes tool integration (symlink)
│       ├── flywheel.py     # Online flywheel training pipeline
│       ├── external_metrics.py  # External vector extraction (token_cost, satisfaction, etc.)
│       ├── metrics.py      # Training metrics
│       ├── train.py        # Pretraining (synthetic data)
│       └── train_value_net.py  # Value network training
├── scripts/                # Standalone training/evaluation scripts
│   ├── rl_train.py         # API-supervised RL training
│   ├── eval_api.py         # Batch evaluation via any OpenAI-compatible API
│   └── checkpoint_manager.py  # Checkpoint registry management
├── data/                   # Training data & artifacts
│   ├── flywheel_buffer.jsonl  # Online RL buffer
│   ├── ds_eval_all_898.jsonl  # API evaluation results (898 pairs)
│   ├── metrics.db          # SQLite metrics DB
│   └── qwen_to_yicenet.json   # Embedding vocabulary mapping
├── checkpoints/            # Model weights (gitignored)
│   ├── yicenet_v15.pt      # Main network (22MB, ~10M params)
│   ├── world_model_v15.pt  # World model (125KB)
│   ├── registry.json       # Version registry (relative paths)
│   └── v14_v15_metrics.json
├── pyproject.toml          # Build config (pip install -e .)
├── INSTALL.md              # Installation & external dependency guide
├── README.md
└── .gitignore
```

## Dependencies

Core: torch, numpy, transformers, sentencepiece, tqdm, huggingface-hub, httpx
Dev: pytest, pytest-cov

## Installation

```bash
pip install -e /path/to/YiCeNet
```

Then: `from yicenet.model import YiCeNet`

All paths are relative. No hardcoded absolute paths or personal information.
