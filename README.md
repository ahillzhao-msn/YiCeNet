# YiCeNet (易策网络)

**~5.6M params | ~22MB | ~4ms inference | I-Ching-inspired orchestration engine**

A lightweight neural network that encodes the I Ching (易经) philosophical framework — controlled randomness + strong logical reasoning — as a fast decomposition engine for AI orchestration systems. Designed for **Hermes Agent** but deployable as a standalone service.

---

## Philosophy → Code

| I Ching | Engineering |
|---|---|
| 太极 (Taiji) | 256-dim state vector encoding user intent |
| 两仪 (Yin-Yang) | Binary decision nodes |
| 八卦 (8 Trigrams) | 8 prototype orchestration capabilities |
| 六十四卦 (64 Hexagrams) | 64 orchestration scenario patterns (learned) |
| 起卦 (Divination) | Gumbel-Softmax sampling for controlled exploration |
| 错综互变 | Deterministic structural reasoning (0 params, fixed logic) |
| 卦爻辞 (Judgment) | Value network scoring candidates |

---

## Architecture

```
Input → TinyEncoder (5.3M) → h (256-dim)
    → GumbelRouter → hexagram (0-63)
    → Hexagram Embedding (64×256)
    → 错/综/互/变 (8 candidates, fixed logic)
    → Value Network → Q-values → select best
    → Action Decoder → action (1 of 50)
```

## Quick Start

### Docker (cross-host deployment)

```bash
# Clone
git clone https://github.com/<your-org>/YiCeNet.git
cd YiCeNet

# Build & start
docker compose up -d

# Service is now available at:
#   http://localhost:8001/v1/health    — health check
#   http://localhost:8001/v1/predict   — inference
#   http://localhost:8501              — dashboard

# Train a model first
docker exec yicenet-api python scripts/training_worker.py --once
```

### Local (Hermes integration)

```bash
# Install
pip install -r requirements.txt

# Setup Hermes tool
ln -sf ~/YiCeNet/src/hermes_tool.py ~/.hermes/hermes-agent/tools/yicenet_tool.py

# Start dashboard
streamlit run dashboard.py --server.port 8501

# Register auto-training cron (from Hermes session)
# Follow prompts from:
python scripts/register_hermes_cron.py
```

---

## Project Structure

```
YiCeNet/
├── api.py                    # FastAPI HTTP service (cross-host Plan A)
├── dashboard.py              # Streamlit monitoring dashboard
├── docker-compose.yml        # Full-stack deployment
├── Dockerfile                # API + training container
├── Dockerfile.dashboard      # Dashboard container
├── requirements.txt          # Python dependencies
│
├── src/
│   ├── model.py              # YiCeNet full model (5.6M params)
│   ├── encoder.py            # TinyTransformer (4 layers, 256-dim)
│   ├── hexagram.py           # 错/综/互/变 transformations
│   ├── config.py             # All hyperparameters
│   ├── value_net.py          # Value network (41K params)
│   ├── decoder.py            # Action decoder (26K params)
│   ├── train.py              # Training pipeline (pretrain + RL)
│   ├── yicenet_engine.py     # In-process inference engine (Plan B)
│   ├── hermes_tool.py        # Hermes tool registration
│   ├── metrics.py            # SQLite metrics logger
│   └── data/
│       └── dataset.py        # Synthetic data + RL environment
│
├── scripts/
│   ├── training_worker.py    # CPU PPO training + A/B switch
│   ├── demo.py               # Interactive inference demo
│   ├── export_onnx.py        # ONNX export (limited by bit ops)
│   └── register_hermes_cron.py  # Cron job setup
│
├── tests/
│   └── test_model.py         # 9 tests (all passing)
│
└── checkpoints/              # Trained model weights
    └── registry.json         # A/B model registry
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/v1/predict` | POST | Inference: task → hexagram + action + Q-values |
| `/v1/switch` | POST | Hot-switch to different checkpoint |
| `/v1/health` | GET | System health + model status |
| `/v1/check-switch` | GET | Check A/B registry for better model |
| `/v1/train` | POST | Trigger one PPO training cycle |
| `/v1/metrics` | GET | Trajectory + success stats |

### Predict Example

```bash
curl -X POST http://localhost:8001/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"task_brief": "search knowledge base", "deterministic": true}'
```

Response:
```json
{
  "hexagram_id": 27,
  "hexagram_name": "颐",
  "hexagram_number": 28,
  "action_id": 0,
  "action_name": "route_to_service",
  "candidates": [
    {"index": 0, "hexagram_name": "颐", "q_value": 0.0342},
    ...
  ],
  "deterministic": true
}
```

---

## Two Deployment Modes

| Aspect | Plan B (Local, default) | Plan A (Cross-host) |
|---|---|---|
| Hermes integration | In-process tool (symlink) | HTTP API call |
| Latency | ~4ms (zero network) | ~10ms (network) |
| Dependencies | PyTorch + Hermes tools | Docker + Docker Compose |
| Setup | `ln -sf` tool file | `docker compose up` |

For local Hermes + YiCeNet on same host → **Plan B** (faster, simpler).
For YiCeNet on separate host → **Plan A** (HTTP via docker-compose).

---

## Training Pipeline

Two-stage training:
1. **Unsupervised pre-train**: K-means on orchestration traces → initialize 64 hexagram prototypes
2. **RL fine-tune (PPO)**: REINFORCE in simulated "fortune teller-customer" environment

Reward signals (disambiguated):
| Terminal Type | Reward | Meaning |
|---|---|---|
| `success` | +1.7 | Task completed naturally |
| `abandoned` | -2.0 | User left after failure |
| `timeout` | -0.5 | Max steps reached |

Auto-training via Hermes cron (every 2h):
1. Check trajectory count in SQLite
2. If ≥500 new → run PPO training (CPU, ~7s)
3. Evaluate: compare win_rate vs active model
4. If +5% better → A/B switch (hot, zero downtime)

---

## License

MIT
