"""
YiCeNet Hermes Tool — in-process inference.

Deployment: editable package in Hermes env (pip install -e ~/YiCeNet).

Usage from Hermes session:
  > yicenet_predict(task_brief="search knowledge base")
  → {"hexagram_id": 35, "hexagram_name": "晋", "action_name": "route_to_service", ...}
"""

import json
import os
import sys

# Hermes tools discovery: when loaded from ~/.hermes/hermes-agent/tools/
try:
    from tools.registry import registry
except ModuleNotFoundError:
    # Not running inside Hermes — this module is a Hermes tool, not a library.
    # The actual functions (yicenet_predict, yicenet_switch) work fine via
    # direct import when called with the right context.
    registry = None

# Resolve YiCeNet project root from hermes-agent/tools/ → ~/YiCeNet
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_YICENET_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", "..", "YiCeNet"))

# Lazy engine import (avoids torch load at Hermes startup)
_engine = None
_active_version = None  # tracks registry.json['active']['version'] for hot-reload

def _get_engine():
    global _engine, _active_version
    if _engine is None:
        from yicenet.yicenet_engine import YiCeNetEngine
        # Read active checkpoint from registry
        reg_path = os.path.join(_YICENET_ROOT, "checkpoints", "registry.json")
        ckpt = ""
        if os.path.exists(reg_path):
            try:
                with open(reg_path) as f:
                    reg = json.load(f)
                ckpt = reg.get("active", {}).get("path", "")
                _active_version = reg.get("active", {}).get("version", "")
                # Resolve relative path (relative to checkpoints/)
                if ckpt and not os.path.isabs(ckpt):
                    ckpt = os.path.join(_YICENET_ROOT, "checkpoints", ckpt)
            except Exception:
                pass
        if not ckpt or not os.path.exists(ckpt):
            ckpt = os.path.join(_YICENET_ROOT, "checkpoints", "yicenet_v15.pt")
        _engine = YiCeNetEngine(checkpoint=ckpt, project_root=_YICENET_ROOT)
    return _engine


def _check_registry_switch():
    """Check if registry.json active version changed since engine load.
    If so, hot-switch the engine to the new checkpoint."""
    global _active_version, _engine

    reg_path = os.path.join(_YICENET_ROOT, "checkpoints", "registry.json")
    if not os.path.exists(reg_path):
        return

    try:
        with open(reg_path) as f:
            reg = json.load(f)
        active = reg.get("active", {})
        new_version = active.get("version", "")
        new_path = active.get("path", "")

        # Resolve relative path
        if new_path and not os.path.isabs(new_path):
            new_path = os.path.join(_YICENET_ROOT, "checkpoints", new_path)

        # No change, skip
        if new_version == _active_version or not new_path:
            return

        # Version changed — hot-switch
        if not os.path.exists(new_path):
            return

        if _engine is not None:
            _engine.switch_model(new_path)
        else:
            # Lazy load will pick it up
            _engine = None
            _get_engine()

        _active_version = new_version
        print(f"[YiCeNet] Hot-switched to {new_version}: {new_path}")
    except Exception:
        pass


def yicenet_predict(task_brief: str, temperature: float = 0.1,
                    deterministic: bool = False) -> str:
    """
    Predict orchestration skeleton for a task description.

    Uses YiCeNet (易策网络), a ~5.6M parameter I-Ching-inspired tiny model.
    Returns hexagram, action, and Q-values for the given task.

    When deterministic=True, bypasses Gumbel exploration noise for
    rigid/fixed workflows. The model outputs pure argmax.
    """
    try:
        # Check if registry.json active changed since last call
        _check_registry_switch()
        engine = _get_engine()
        result = engine.predict(task_brief, temperature, deterministic)
        # Log to metrics DB
        try:
            from yicenet.metrics import MetricsLogger
            MetricsLogger().log_trajectory(
                session_id="hermes_tool",
                hexagram_id=result["hexagram_id"],
                candidate_values=result["q_values"],
                action_id=result["action_id"],
                reward=0.0,  # updated post-hoc when terminal_type known
                terminal_type="active",
                latency_ms=0.0,
            )
            MetricsLogger().log_hexagram_usage(
                result["hexagram_id"],
                max(result["q_values"]) if result["q_values"] else 0.0,
            )
        except Exception as log_err:
            pass  # metrics logging is non-critical

        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def yicenet_switch(checkpoint: str) -> str:
    """
    Hot-switch YiCeNet to a different checkpoint (A/B swap).
    """
    try:
        engine = _get_engine()
        engine.switch_model(checkpoint)
        return json.dumps({"success": True, "active": checkpoint})
    except Exception as e:
        return json.dumps({"error": str(e)})


def check_yicenet_requirements() -> bool:
    """Check if YiCeNet can run."""
    try:
        import torch
        # Check registry first, then fallback
        reg_path = os.path.join(_YICENET_ROOT, "checkpoints", "registry.json")
        if os.path.exists(reg_path):
            with open(reg_path) as f:
                reg = json.load(f)
            active_path = reg.get("active", {}).get("path", "")
            if active_path:
                if not os.path.isabs(active_path):
                    active_path = os.path.join(_YICENET_ROOT, "checkpoints", active_path)
                if os.path.exists(active_path):
                    return True
        ckpt = os.path.join(_YICENET_ROOT, "checkpoints", "yicenet_v15.pt")
        return os.path.exists(ckpt)
    except ImportError:
        return False


# ── Schema ──
YICENET_SCHEMA = {
    "type": "function",
    "function": {
        "name": "yicenet_predict",
        "description": (
            "Use YiCeNet (易策网络) — a tiny I-Ching-inspired neural network — "
            "to generate an orchestration skeleton for a given task. "
            "Returns hexagram (0-63), action, and Q-values for 8 structural variants. "
            "Call this when you need a fast, lightweight decomposition of a complex task."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_brief": {
                    "type": "string",
                    "description": "Brief description of the task to decompose"
                },
                "temperature": {
                    "type": "number",
                    "description": "Exploration temperature (0.0=deterministic, 1.0=exploratory)",
                    "default": 0.1,
                },
                "deterministic": {
                    "type": "boolean",
                    "description": "If True, bypass Gumbel noise entirely. Pure argmax. Use for rigid/fixed workflows where exploration must not interfere.",
                    "default": False,
                },
            },
            "required": ["task_brief"],
        },
    },
}

YICENET_SWITCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "yicenet_switch",
        "description": "Hot-switch YiCeNet to a different checkpoint for A/B model comparison.",
        "parameters": {
            "type": "object",
            "properties": {
                "checkpoint": {
                    "type": "string",
                    "description": "Path to checkpoint .pt file"
                },
            },
            "required": ["checkpoint"],
        },
    },
}


# ── Register tools (Hermes only) ──
if registry is not None:
    registry.register(
        name="yicenet_predict",
        toolset="file",
        schema=YICENET_SCHEMA,
        handler=lambda args, **kw: yicenet_predict(
            task_brief=args.get("task_brief", ""),
            temperature=float(args.get("temperature", 0.1)),
            deterministic=bool(args.get("deterministic", False)),
        ),
        check_fn=check_yicenet_requirements,
        emoji="☯",
    )

    registry.register(
        name="yicenet_switch",
        toolset="file",
        schema=YICENET_SWITCH_SCHEMA,
        handler=lambda args, **kw: yicenet_switch(
            checkpoint=args.get("checkpoint", ""),
        ),
        check_fn=check_yicenet_requirements,
        emoji="🔄",
    )
