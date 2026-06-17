"""
YiCeNet Hermes plugin hook implementations.

Called from ~/.hermes/plugins/yicenet-hooks/__init__.py:
  pre_llm_call(context)   — inject hexagram framing + consume prior feedback
  post_tool_call(context) — record tool invocation in MemoryBank
  post_llm_call(context)  — store response metadata for next turn's feedback

Hermes is a long-lived daemon: MemoryBank stays in memory between calls.
FileBackend is attached only if persist_daemon_sessions=True in config.

HermesAdapter.platform_signals() computes corrected/praised from
conversation_history so the orchestrator uses real signals, not text guesses.
"""
from __future__ import annotations

import sys


# ── Platform adapter ──────────────────────────────────────────────────────────

class HermesAdapter:
    """Translates Hermes context dicts to domain primitives."""

    platform_id = "hermes"
    process_model = "daemon"

    def session_id(self, payload: dict) -> str:
        return payload.get("session_id", "")

    def turn_id(self, payload: dict) -> int:
        tid = payload.get("turn_id")
        if tid is not None:
            return int(tid)
        # Fallback: count from conversation_history length
        history = payload.get("conversation_history") or payload.get("messages") or []
        return max(0, len(history) - 1)

    def prompt(self, payload: dict) -> str:
        messages = payload.get("messages") or [{}]
        last = messages[-1]
        return last.get("content", "") if isinstance(last, dict) else ""

    def assistant_response(self, payload: dict) -> str:
        """Extract assistant response text from Hermes post_llm_call context."""
        resp = payload.get("assistant_response") or payload.get("response", "")
        if isinstance(resp, dict):
            return resp.get("content", "")
        return str(resp) if resp else ""

    def platform_signals(self, payload: dict):
        """Derive real feedback signals from conversation_history.

        Hermes supplies the full conversation at post_llm_call time, so we
        can compare the current user message against the previous assistant
        response to infer corrected / praised / continued directly.
        Returns None at pre_llm_call (called before the response exists).
        """
        history = payload.get("conversation_history") or []
        if len(history) < 2:
            return None

        # Find the last user message and the assistant message before it
        user_msg, prev_asst = "", ""
        for i in range(len(history) - 1, -1, -1):
            entry = history[i]
            role = entry.get("role", "")
            content = entry.get("content", "")
            if role == "user" and not user_msg:
                user_msg = content
            elif role == "assistant" and user_msg and not prev_asst:
                prev_asst = content
                break

        if not user_msg or not prev_asst:
            return None

        from yicenet.external_metrics import (
            compute_satisfaction, estimate_token_cost,
            _check_patterns,
            _CORRECTION_PATTERNS, _COMPLETION_PATTERNS,
            _PRAISE_PATTERNS, _ABANDON_PATTERNS,
        )
        corrected = _check_patterns(user_msg, _CORRECTION_PATTERNS)
        completed = _check_patterns(user_msg, _COMPLETION_PATTERNS)
        praised   = _check_patterns(user_msg, _PRAISE_PATTERNS)
        abandoned = _check_patterns(user_msg, _ABANDON_PATTERNS) or not user_msg.strip()
        continued = bool(user_msg.strip()) and not abandoned
        satisfaction = compute_satisfaction(user_msg, prev_asst)
        token_cost = estimate_token_cost(prev_asst)

        return {
            "continued":   continued,
            "corrected":   corrected,
            "completed":   completed,
            "praised":     praised,
            "abandoned":   abandoned,
            "satisfaction": satisfaction,
            "token_cost":  token_cost,
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_adapter = HermesAdapter()
_orchestrator = None


def _orch():
    global _orchestrator
    if _orchestrator is None:
        from yicenet.hook_engine import HookOrchestrator
        _orchestrator = HookOrchestrator(_adapter)
    return _orchestrator


# ── Hook entry points ─────────────────────────────────────────────────────────

def pre_llm_call(context: dict) -> dict | None:
    """Inject YiCeNet hexagram context; consume previous turn's feedback."""
    # Submit feedback for the previous turn first
    try:
        _orch().before_prediction(context)
    except Exception:
        pass

    try:
        from yicenet.engine_provider import EngineProvider
        from yicenet.providers import ProviderRegistry

        prompt = _adapter.prompt(context)
        session_id = _adapter.session_id(context)

        engine = EngineProvider.get_engine()
        turn_id = _adapter.turn_id(context)
        result = engine.predict(
            prompt or "general task",
            temperature=0.1,
            session_id=session_id,
            turn_id=turn_id,
        )

        display = ProviderRegistry.default().display
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
        session_id = _adapter.session_id(context)
        turn_id = _adapter.turn_id(context)
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
    """Record response metadata for next turn's real feedback extraction."""
    try:
        _orch().on_turn_complete(context)
    except Exception:
        pass
