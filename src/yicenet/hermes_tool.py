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
    registry = None

from yicenet.engine_provider import EngineProvider


def yicenet_predict(task_brief: str, temperature: float = 0.1,
                    deterministic: bool = False,
                    return_prescription: bool = False,
                    session_id: str = "",
                    turn_id: int = 0,
                    turn_summary: str = "") -> str:  # noqa: D417
    """
    Predict orchestration skeleton for a task description.

    Uses YiCeNet (易策网络), a ~5.6M parameter I-Ching-inspired tiny model.
    Returns hexagram, action, and Q-values for the given task.

    When deterministic=True, bypasses Gumbel exploration noise for
    rigid/fixed workflows. The model outputs pure argmax.

    When return_prescription=True, also runs cross-attention against
    historical turns in the same session. Requires session_id.
    Default False preserves existing behavior.
    """
    try:
        EngineProvider.check_switch()
        engine = EngineProvider.get_engine()
        result = engine.predict(
            task_brief, temperature, deterministic,
            return_prescription=return_prescription,
            session_id=session_id,
            turn_id=turn_id,
            turn_summary=turn_summary,
        )
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
    """Hot-switch YiCeNet to a different checkpoint (A/B swap)."""
    try:
        EngineProvider.switch(checkpoint)
        return json.dumps({"success": True, "active": checkpoint})
    except Exception as e:
        return json.dumps({"error": str(e)})


def check_yicenet_requirements() -> bool:
    """Check if YiCeNet can run."""
    try:
        import torch  # noqa: F401
        EngineProvider._resolve_checkpoint()
        return True
    except Exception:
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
