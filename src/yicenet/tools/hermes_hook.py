"""
YiCeNet Hermes plugin hook implementations.

Called from ~/.hermes/plugins/yicenet-hooks/__init__.py:
  pre_llm_call(context)   — inject hexagram framing before LLM sees the prompt
  post_tool_call(context) — record tool result in MemoryBank
  post_llm_call(context)  — submit flywheel reward signal
"""
from __future__ import annotations


def pre_llm_call(context: dict) -> dict | None:
    """Inject YiCeNet hexagram context before the LLM call."""
    try:
        from yicenet.engine_provider import EngineProvider
        from yicenet.display import format_prediction

        prompt = (context.get("messages") or [{}])[-1].get("content", "")
        session_id = context.get("session_id", "")
        engine = EngineProvider.get_engine()
        result = engine.predict(prompt or "general task", temperature=0.1)

        hexagram_str = format_prediction(result, mode="compact")
        return {"yicenet_context": hexagram_str}
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
