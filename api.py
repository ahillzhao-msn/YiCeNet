"""
YiCeNet FastAPI HTTP service — Plan A for cross-host deployment.

When YiCeNet runs on a different host from Hermes, expose the model
via this lightweight HTTP API. Hermes calls it via HTTP instead of
the in-process tool.

Usage (dev):
    pip install fastapi uvicorn
    python api.py

Usage (Docker):
    docker compose up yicenet-api
"""

import os
import sys
from pathlib import Path

# Add project root to path
_root = Path(__file__).parent
sys.path.insert(0, str(_root))

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.yicenet_engine import YiCeNetEngine, get_engine
from src.metrics import MetricsLogger


app = FastAPI(
    title="YiCeNet (易策网络)",
    description="I-Ching-inspired lightweight orchestration engine",
    version="1.0.0",
)


# ── Request/Response models ──

class PredictRequest(BaseModel):
    task_brief: str
    temperature: float = 0.1
    deterministic: bool = False


class PredictResponse(BaseModel):
    hexagram_id: int
    hexagram_name: str
    hexagram_number: int
    hexagram_pattern: str
    best_candidate: int
    selected_hexagram_id: int
    selected_hexagram_name: str
    candidates: list
    action_id: int
    action_name: str
    q_values: list
    temperature: float
    deterministic: bool


class SwitchRequest(BaseModel):
    checkpoint: str


class SwitchResponse(BaseModel):
    success: bool
    active: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    active_checkpoint: str
    total_trajectories: int


class TrainResponse(BaseModel):
    status: str
    version: str | None = None
    avg_reward: float | None = None
    duration_sec: float | None = None


# ── Endpoints ──

@app.post("/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Run YiCeNet inference: encode → divine → evaluate → act."""
    try:
        engine = get_engine()
        result = engine.predict(
            req.task_brief,
            temperature=req.temperature,
            deterministic=req.deterministic,
        )
        # Log to metrics
        try:
            MetricsLogger().log_trajectory(
                session_id="api",
                hexagram_id=result["hexagram_id"],
                candidate_values=result["q_values"],
                action_id=result["action_id"],
                reward=0.0,
                terminal_type="active",
            )
            MetricsLogger().log_hexagram_usage(
                result["hexagram_id"],
                max(result["q_values"]) if result["q_values"] else 0.0,
            )
        except Exception:
            pass
        return PredictResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/switch", response_model=SwitchResponse)
def switch_model(req: SwitchRequest):
    """Hot-switch to a different checkpoint."""
    try:
        engine = get_engine()
        resolved = os.path.join(_root, req.checkpoint)
        engine.switch_model(resolved if os.path.exists(resolved) else req.checkpoint)
        return SwitchResponse(success=True, active=engine.active_checkpoint)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/v1/health", response_model=HealthResponse)
def health():
    """System health check."""
    engine = get_engine()
    stats = MetricsLogger().get_stats()
    return HealthResponse(
        status="ok" if engine.is_loaded else "loading",
        model_loaded=engine.is_loaded,
        active_checkpoint=engine.active_checkpoint,
        total_trajectories=stats["total_trajectories"],
    )


@app.get("/v1/check-switch")
def check_switch():
    """Check registry for ready model and auto-switch if better."""
    engine = get_engine()
    result = engine.check_for_switch()
    return result or {"should_switch": False, "reason": "No ready model"}


@app.post("/v1/train", response_model=TrainResponse)
def trigger_training():
    """Trigger one training cycle."""
    try:
        from scripts.training_worker import run_once
        result = run_once()
        return TrainResponse(status="ok", **result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/metrics")
def get_metrics():
    """Quick metrics summary."""
    try:
        return MetricsLogger().get_stats()
    except Exception as e:
        return {"error": str(e)}


# ── Main ──

if __name__ == "__main__":
    port = int(os.environ.get("YICENET_PORT", 8001))
    host = os.environ.get("YICENET_HOST", "0.0.0.0")
    print(f"YiCeNet API starting on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
