"""
YiCeNet Hermes plugin hook adapter.

HermesAdapter implements only what is specific to Hermes:
  - session_id: directly from context dict
  - turn_id: checks direct turn_id field + conversation_history fallback
  - assistant_response: from context['assistant_response'] or context['response']
  - platform_signals: derived from full conversation_history (real feedback, not guessed)
  - process_model: "daemon" (Hermes is long-lived)

All shared prediction and hook lifecycle logic lives in HooksAdapter.

Called from ~/.hermes/plugins/yicenet-hooks/__init__.py:
  pre_llm_call(context)   — inject hexagram framing + consume prior feedback
  post_tool_call(context) — record tool invocation in MemoryBank
  post_llm_call(context)  — store response metadata for next turn's feedback
"""
from __future__ import annotations

import sys
from typing import Optional

from yicenet.tools.hooks_adapter import HooksAdapter


class HermesAdapter(HooksAdapter):
    """Platform adapter for Hermes daemon hooks."""

    _platform_id = "hermes"

    @property
    def platform_id(self) -> str:
        return self._platform_id

    @property
    def process_model(self) -> str:
        return "daemon"

    def session_id(self, payload: dict) -> str:
        return payload.get("session_id", "")

    def turn_id(self, payload: dict) -> int:
        tid = payload.get("turn_id")
        if tid is not None:
            return int(tid)
        history = payload.get("conversation_history") or payload.get("messages") or []
        return max(0, len(history) - 1)

    def prompt(self, payload: dict) -> str:
        messages = payload.get("messages") or [{}]
        last = messages[-1]
        return last.get("content", "") if isinstance(last, dict) else ""

    def assistant_response(self, payload: dict) -> str:
        resp = payload.get("assistant_response") or payload.get("response", "")
        if isinstance(resp, dict):
            return resp.get("content", "")
        return str(resp) if resp else ""

    def platform_signals(self, payload: dict) -> Optional[dict]:
        """Derive real feedback signals from conversation_history.

        Hermes supplies the full conversation at post_llm_call time, so we
        compare the current user message against the previous assistant response
        to infer corrected / praised / continued directly.
        Returns None at pre_llm_call (called before the response exists).
        """
        history = payload.get("conversation_history") or []
        if len(history) < 2:
            return None

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
            "continued":    continued,
            "corrected":    corrected,
            "completed":    completed,
            "praised":      praised,
            "abandoned":    abandoned,
            "satisfaction": satisfaction,
            "token_cost":   token_cost,
        }


# ── Module-level singleton (Hermes plugin interface) ──────────────────────────

_adapter = HermesAdapter()


def pre_llm_call(context: dict) -> "dict | None":
    """Inject YiCeNet hexagram context; consume previous turn's feedback."""
    result = _adapter.predict_for_turn_payload(context)
    if result is None:
        return None
    label = result["yicenet"]["label"]
    sys.stderr.write(label + "\n")
    sys.stderr.flush()
    return {"yicenet_context": label}


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
        engine.predict(
            f"tool:{tool_name}",
            temperature=0.1,
            session_id=session_id,
            turn_id=turn_id,
            return_prescription=True,
        )
    except Exception:
        pass


def post_llm_call(context: dict) -> None:
    """Record response metadata for next turn's real feedback extraction."""
    _adapter.stop(context)
