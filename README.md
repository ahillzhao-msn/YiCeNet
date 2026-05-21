# YiCeNet (易策网络)

I-Ching-inspired tiny neural network for orchestration task decomposition.

**~5.6M parameters, ~21MB FP32, ~4ms GPU inference.**

## Quick Start

```bash
git clone https://github.com/ahillzhao-msn/YiCeNet.git
cd YiCeNet
pip install torch transformers tqdm numpy
```

### Option A: Download pre-trained weights (recommended)

```bash
# Download v4 release
curl -L -o yicenet_v4.tar.gz \
  https://github.com/ahillzhao-msn/YiCeNet/releases/download/v4/yicenet_v4_release.tar.gz
tar xzf yicenet_v4.tar.gz

# Verify files exist
ls checkpoints/yicenet_v4.pt    # main model (~22MB)
ls checkpoints/world_model_best.pt  # world model (~200KB)
ls data/qwen_to_yicenet.json    # BPE token mapping

# Build registry.json for inference
echo '{"active":{"version":"v4","path":"'$(pwd)'/checkpoints/yicenet_v4.pt"}}' \
  > checkpoints/registry.json

# Run demo
python scripts/demo.py --scenario "search knowledge base for SAP PM"
```

### Option B: Train from scratch

```bash
# Build vocabulary from your own Hermes session data
python src/tokenizer.py

# Full training pipeline (pre-train + RL)
python src/train.py --dataset session --stage all --episodes 500 --pretrain_epochs 50

# Or step by step:
python src/train.py --dataset session --stage pretrain --pretrain_epochs 80   # K-means + contrastive
python src/world_model.py                                                      # Train world model
python src/rl_train.py --episodes 1000                                         # RL fine-tune
```

## Architecture

```
Input (Qwen BPE tokens, 8000 vocab) → TinyEncoder (4×Transformer, 256-dim)
  → Gumbel Router → hexagram index (0-63)
  → Hexagram Embedding (64×256)
  → Structural Reasoning (错/综/互/变) → 8 candidates
  → Value Network → Q-values
  → Action Decoder → orchestration action
```

## Training

The model is trained in three stages:

| Stage | What | Data |
|-------|------|------|
| **1. Pre-train** | K-means + contrastive loss on encoder | 665 real session messages |
| **2. World Model** | Supervised regression: (h, hex) → reward | 5,320 (h, hex, reward) pairs |
| **3. RL** | REINFORCE with world model as reward signal | 1,000 episodes |

**Pre-trained weights** ([v4 release](https://github.com/ahillzhao-msn/YiCeNet/releases/tag/v4)) 
are trained on the author's personal Hermes session data (~665 Chinese/English technical 
conversations about SAP ABAP, system administration, and ML training). The tokenizer 
(Qwen BPE) is universal, but the hexagram preferences are personalized. 
**For best results on your own data, retrain Stage 2+3 with your session logs.**

## Inference

```python
from src.yicenet_engine import YiCeNetEngine

engine = YiCeNetEngine(project_root=".")
result = engine.predict("search knowledge base", temperature=0.1)
print(result["hexagram_name"], result["action_name"])
```

### Hermes Integration

When used as a Hermes agent tool:

```
yicenet_predict(task_brief="search knowledge base")
→ {"hexagram_id": 35, "hexagram_name": "晋", "action_name": "route_to_service", ...}
```

Supports A/B model switching via `checkpoints/registry.json`.

## Files

```
src/
├── tokenizer.py       Qwen BPE → 8000 vocab mapping
├── model.py           YiCeNet full model
├── encoder.py         4-layer TinyTransformer
├── decoder.py         Action decoder
├── value_net.py       Q-value MLP
├── hexagram.py        错/综/互/变 structural reasoning
├── world_model.py     Reward predictor (18K params)
├── train.py           Training pipeline (pre-train + RL)
├── rl_train.py        World model driven RL fine-tuning
├── train_value_net.py Value network supervised training
├── data/
│   └── dataset.py     SessionDataset + DataDrivenEnv
├── yicenet_engine.py  Inference engine
└── hermes_tool.py     Hermes agent tool integration
scripts/
├── demo.py            Inference demo
└── *.sh               Training scripts
```

## License

MIT
