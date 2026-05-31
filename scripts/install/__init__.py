"""
YiCeNet (易策网络) — standalone Hermes lifecycle hooks.

Three-channel flywheel data flow:

  1. Session DB scan (scan_new_messages) — intrinsic foundation.
     The model "understands" context through embedding-based
     learning, not pattern matching.  Works even when predict()
     is called standalone without an immediate response.

  2. LOOM solidify → _loom_to_yicenet() → flywheel buffer
     (when LOOM is present; indirect via LOOM's architecture).

  3. This plugin: post_llm_call → feedback() → flywheel buffer.
     First-hand data that Session DB cannot infer: accurate
     token costs from post_api_request, real-time response
     length, model identity.

All learning is embedding-based.  The I-Ching principle is
chaos and change (易) — there are no absolute patterns.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logging.getLogger("yicenet-hooks").setLevel(logging.INFO)
logger = logging.getLogger("yicenet-hooks")

# YiCeNet source path (for yicenet_predict import)
_YICENET_SRC = os.path.expanduser("~/YiCeNet/src")
if _YICENET_SRC not in sys.path:
    sys.path.insert(0, _YICENET_SRC)

# Flywheel buffer path
_YICENET_BUFFER = str(Path.home() / "YiCeNet" / "data" / "flywheel_buffer.jsonl")

# Per-session token usage accumulator
_session_usage: dict[str, dict[str, float]] = {}

# Lazy-loaded YiCeNet
_yicenet_predict = None


def _get_yicenet():
    global _yicenet_predict
    if _yicenet_predict is None:
        try:
            from yicenet.hermes_tool import yicenet_predict as yp
            _yicenet_predict = yp
        except ImportError as e:
            logger.warning("yicenet_predict not available: %s", e)
            _yicenet_predict = False
    return _yicenet_predict if _yicenet_predict else None


def feedback(session_id: str, response_chars: int,
              input_chars: int, n_turns: int,
              model: str, platform: str) -> None:
    """Write first-hand reward signal to flywheel training buffer.

    Enhancement to YiCeNet's intrinsic Session DB scan.
    Supplies first-hand data: accurate token costs, response length,
    model identity — things the Session DB cannot provide directly.
    """
    if not _YICENET_BUFFER:
        return

    usage = _session_usage.pop(session_id, {})
    token_cost = usage.get("total_tokens", 0) or int(response_chars * 0.25)
    token_efficiency = usage.get("efficiency", 0)
    if not token_efficiency:
        total = input_chars + response_chars + 1
        token_efficiency = response_chars / total if total > 0 else 0.5

    # Model cost multiplier
    cost_factors = {
        "deepseek-v4-flash": 0.00015,
        "deepseek-v4-pro": 0.0015,
    }
    model_key = model.split("/")[-1] if "/" in model else model
    cost_per_char = cost_factors.get(model_key, 0.0003)

    sample = {
        "user_text": f"[yicenet-hooks] sid={session_id[:12]}",
        "producer": "yicenet-hooks",
        "conversation_id": session_id,
        "hexagram_evolution": [],
        "timestamp": time.time(),
        "token_cost": int(token_cost),
        "token_efficiency": round(token_efficiency, 4),
        "continued": False,
        "corrected": False,
        "completed": n_turns > 0,
        "praised": False,
        "abandoned": False,
        "satisfaction": round(
            min(1.0, token_efficiency * 1.5)
            * (1.0 - min(0.3, cost_per_char * 100)),
            4
        ),
    }

    try:
        os.makedirs(os.path.dirname(_YICENET_BUFFER), exist_ok=True)
        with open(_YICENET_BUFFER, "a") as f:
            f.write(json.dumps(sample) + "\n")
    except Exception as e:
        logger.debug("feedback write failed: %s", e)


def _record_api_usage(session_id: str, usage_data: dict | None) -> None:
    if not session_id or not usage_data:
        return
    if session_id not in _session_usage:
        _session_usage[session_id] = {"total_tokens": 0, "api_calls": 0, "total_input": 0, "total_output": 0}
    acc = _session_usage[session_id]
    acc["total_tokens"] += usage_data.get("total_tokens", 0) or 0
    acc["api_calls"] += 1
    in_tok = usage_data.get("input_tokens", 0) or usage_data.get("prompt_tokens", 0) or 0
    out_tok = usage_data.get("output_tokens", 0) or usage_data.get("completion_tokens", 0) or 0
    acc["total_input"] += in_tok
    acc["total_output"] += out_tok
    total = acc["total_input"] + acc["total_output"]
    acc["efficiency"] = acc["total_output"] / total if total > 0 else 0.5


# ── Hook Handlers ─────────────────────────────────────────


def _loom_hooks_active() -> bool:
    """检查 loom-hooks 插件是否已加载（此时 yicenet-hooks 应自我抑制）。"""
    try:
        from hermes_cli.plugins import get_plugin_manager
        pm = get_plugin_manager()
        return 'loom-hooks' in pm._plugins
    except Exception:
        return False


def on_session_start(**kw: Any) -> None:
    """Establish hexagram baseline at session start.
    当 loom-hooks 活跃时跳过（LOOM 的 on_session_start + recommend 已处理）。"""
    if _loom_hooks_active():
        return
    yp = _get_yicenet()
    if not yp:
        return
    session_id = kw.get("session_id", "?")
    platform = kw.get("platform", "?")
    try:
        yp(f"Session start: {platform}", temperature=0.1, deterministic=True)
        logger.debug("yicenet baseline: session %s", session_id[:12])
    except Exception as e:
        logger.debug("yicenet baseline skipped: %s", e)


def pre_llm_call(**kw: Any) -> dict | str | None:
    """YiCeNet context sensing — inject hexagram before every turn.

    检测到 loom-hooks 插件已加载时跳过（LOOM 已处理），
    避免每轮算两卦。yicenet-hooks 仅用于无 LOOM 的独立部署。
    """
    if _loom_hooks_active():
        logger.debug("loom-hooks active — yicenet-hooks pre_llm_call skipped")
        return None
    yp = _get_yicenet()
    if not yp:
        return None

    user_message = kw.get("user_message", "")
    session_id = kw.get("session_id", "")
    is_first = kw.get("is_first_turn", False)

    if not user_message or not user_message.strip():
        return None

    try:
        hx = yp(user_message[:200], temperature=0.1,
                 deterministic=is_first)
        if hx and len(hx) > 10:
            return {"context": f"[YiCeNet hexagram]\n{hx[:300]}"}
    except Exception as e:
        logger.debug("yicenet predict skipped: %s", e)

    return None


def post_api_request(**kw: Any) -> None:
    """Accumulate token usage for accurate reward computation."""
    usage = kw.get("usage")
    if usage and isinstance(usage, dict):
        _record_api_usage(kw.get("session_id", ""), usage)


def post_llm_call(**kw: Any) -> None:
    """Send reward signal to YiCeNet flywheel training buffer."""
    user_message = kw.get("user_message", "")
    assistant_response = kw.get("assistant_response", "")
    session_id = kw.get("session_id", "")
    model = kw.get("model", "unknown")
    platform = kw.get("platform", "")
    history = kw.get("conversation_history", [])
    n_turns = sum(1 for m in history if isinstance(m, dict) and m.get("role") == "assistant")

    if not assistant_response or not assistant_response.strip():
        return

    feedback(
        session_id=session_id,
        response_chars=len(assistant_response),
        input_chars=len(user_message or ""),
        n_turns=n_turns,
        model=model,
        platform=platform,
    )


def on_session_end(**kw: Any) -> None:
    """Log session end (buffer metadata)."""
    session_id = kw.get("session_id", "?")
    logger.debug("yicenet session ended: %s", session_id[:12])


# ── Plugin Registration ──────────────────────────────────


def register(ctx) -> None:
    """Register all YiCeNet lifecycle hooks."""
    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("pre_llm_call", pre_llm_call)
    ctx.register_hook("post_api_request", post_api_request)
    ctx.register_hook("post_llm_call", post_llm_call)
    ctx.register_hook("on_session_end", on_session_end)
    logger.info("yicenet-hooks: registered 5 hooks (3-channel flywheel)")
