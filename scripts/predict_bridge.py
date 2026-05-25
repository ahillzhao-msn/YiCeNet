#!/usr/bin/env python3
"""
YiCeNet Predict Bridge — called via subprocess from the Hermes tool.

Reads a JSON request from stdin, runs inference in the venv (which has torch),
prints JSON result to stdout.

Usage (from hermes_tool.py):
    echo '{"task_brief": "..."}' | /path/to/venv/python scripts/predict_bridge.py
"""
import json
import os
import sys

# Resolve project root: this script lives at yicenet/scripts/predict_bridge.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# yicenet is pip-installed as editable (pip install -e ~/yicenet)
# Imports use the installed package name, not 'src'


def main():
    raw = sys.stdin.read()
    try:
        params = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON input: {e}"}))
        sys.exit(1)

    task_brief = params.get("task_brief", "")
    temperature = float(params.get("temperature", 0.1))
    deterministic = bool(params.get("deterministic", False))
    mode = params.get("mode", "predict")  # "predict" | "switch" | "check"

    if mode == "check":
        # Dependency check from inside the venv
        try:
            import torch
            ckpt = _resolve_checkpoint()
            if ckpt and os.path.exists(ckpt):
                result = {
                    "ready": True,
                    "torch": True,
                    "checkpoint": ckpt,
                    "cuda": torch.cuda.is_available(),
                }
            else:
                result = {"ready": False, "torch": True, "checkpoint": ckpt or None}
        except ImportError as e:
            result = {"ready": False, "torch": False, "error": str(e)}
        print(json.dumps(result, ensure_ascii=False))
        return

    if mode == "switch":
        from yicenet.yicenet_engine import YiCeNetEngine

        checkpoint = params.get("checkpoint", "")
        if not checkpoint or not os.path.exists(checkpoint):
            print(json.dumps({"error": f"Checkpoint not found: {checkpoint}"}))
            sys.exit(1)

        engine = YiCeNetEngine(project_root=_PROJECT_ROOT)
        engine.switch_model(checkpoint)
        print(json.dumps({"success": True, "active": checkpoint}))
        return

    if mode == "predict":
        from contextlib import redirect_stdout
        from io import StringIO
        from yicenet.yicenet_engine import YiCeNetEngine

        _stderr = sys.stderr
        with redirect_stdout(_stderr):
            engine = YiCeNetEngine(project_root=_PROJECT_ROOT)
            result = engine.predict(task_brief, temperature, deterministic)

        # Metrics logging (non-fatal if it fails)
        try:
            from yicenet.metrics import MetricsLogger
            MetricsLogger().log_trajectory(
                session_id="bridge",
                hexagram_id=result["hexagram_id"],
                candidate_values=result["q_values"],
                action_id=result["action_id"],
                reward=0.0,
                terminal_type="active",
                latency_ms=0.0,
            )
            MetricsLogger().log_hexagram_usage(
                result["hexagram_id"],
                max(result["q_values"]) if result["q_values"] else 0.0,
            )
        except Exception:
            pass

        print(json.dumps(result, ensure_ascii=False))


def _resolve_checkpoint() -> str:
    """Try registry.json first, then fallback to default checkpoint."""
    reg_path = os.path.join(_PROJECT_ROOT, "checkpoints", "registry.json")
    if os.path.exists(reg_path):
        try:
            with open(reg_path) as f:
                reg = json.load(f)
            ckpt = reg.get("active", {}).get("path", "")
            if ckpt and os.path.exists(ckpt):
                return ckpt
        except Exception:
            pass

    # Fallback
    for candidate in ["yicenet_v4.pt", "yicenet_rl_best.pt", "yicenet_final.pt"]:
        ckpt = os.path.join(_PROJECT_ROOT, "checkpoints", candidate)
        if os.path.exists(ckpt):
            return ckpt
    return ""


if __name__ == "__main__":
    main()
