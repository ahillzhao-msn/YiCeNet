"""
YiCeNet Hermes plugin hook implementations.

Called from ~/.hermes/plugins/yicenet-hooks/__init__.py:
  pre_llm_call(context)   — inject hexagram framing before LLM sees the prompt
  post_tool_call(context) — record tool result in MemoryBank
  post_llm_call(context)  — submit flywheel reward signal

Display injection:
  TerminalDisplay(mode=compact) via ProviderRegistry.default().display.
  If hexagram_chain is enabled in config and session_id is present,
  the turn-history chain prefix is prepended automatically.
"""
from __future__ import annotations

import os
import sys

# Suppress tokenizer/model progress bars and HF Hub network calls.
# Must run before any sentence-transformers or transformers import.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def pre_llm_call(context: dict) -> dict | None:
    """Inject YiCeNet hexagram context before the LLM call."""
    try:
        from yicenet.engine_provider import EngineProvider
        from yicenet.providers import ProviderRegistry

        prompt = (context.get("messages") or [{}])[-1].get("content", "")
        session_id = context.get("session_id", "")

        engine = EngineProvider.get_engine()
        result = engine.predict(prompt or "general task", temperature=0.1)

        display = ProviderRegistry.default().display

        # Fetch chain only when the display is configured to use it
        chain = None
        if session_id and display.needs_chain:
            from yicenet.memory_bank import get_memory_bank
            chain = get_memory_bank().get_hexagram_history(session_id)

        rendered = display.render(result, chain=chain)
        sys.stderr.write(rendered + "\n")
        sys.stderr.flush()
        return {"yicenet_context": rendered}
    except Exception:
        return None


def post_tool_call(context: dict) -> None:
    """Record tool invocation turn in MemoryBank."""
    try:
        session_id = context.get("session_id", "")
        turn_id = context.get("turn_id", 0)
        tool_name = context.get("tool_name", "")
        if not session_id:
            return
        from yicenet.engine_provider import EngineProvider
        engine = EngineProvider.get_engine()
        engine.prescribe(
            task_text=f"tool:{tool_name}",
            session_id=session_id,
            turn_id=turn_id,
            turn_summary=f"tool_call:{tool_name}",
        )
    except Exception:
        pass


def post_llm_call(context: dict) -> None:
    """Submit flywheel reward from conversation outcome signals."""
    try:
        from yicenet.flywheel import submit_trajectory

        session_id = context.get("session_id", "")
        signals = context.get("signals", {})
        submit_trajectory({
            "producer": "hermes-hook",
            "version": 1,
            "conversation_id": session_id,
            "trajectory": {
                "continued": bool(signals.get("continued")),
                "corrected": bool(signals.get("corrected")),
                "completed": bool(signals.get("completed")),
                "praised": bool(signals.get("praised")),
                "abandoned": bool(signals.get("abandoned")),
                "token_cost": int(signals.get("token_cost", 0)),
                "token_efficiency": float(signals.get("token_efficiency", 0.0)),
            },
        })
    except Exception:
        pass
