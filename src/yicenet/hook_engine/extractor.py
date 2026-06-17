"""
hook_engine.extractor — Pure feedback extraction functions.

No side effects, no I/O, no platform imports.
Input: text strings + TurnRecord metadata.
Output: FeedbackSignals (frozen dataclass).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yicenet.memory_bank import TurnRecord


@dataclass(frozen=True)
class FeedbackSignals:
    """Immutable feedback inferred for a completed turn."""
    continued: bool
    corrected: bool
    completed: bool
    praised: bool
    abandoned: bool
    satisfaction: float    # [-1.0, 1.0]
    token_cost: float      # [0.0, 1.0]  normalised


def extract_feedback(next_prompt: str, last_turn: "TurnRecord") -> FeedbackSignals:
    """Infer Turn N's feedback from Turn N+1's opening user prompt.

    Pure function — no I/O, no state.

    next_prompt: the new user message (Turn N+1).
    last_turn:   TurnRecord with metadata["response_snippet"] and
                 metadata["response_char_count"] populated by on_turn_complete().
    """
    from yicenet.external_metrics import (
        compute_satisfaction,
        _check_patterns,
        _CORRECTION_PATTERNS,
        _COMPLETION_PATTERNS,
        _PRAISE_PATTERNS,
        _ABANDON_PATTERNS,
    )

    response_snippet = last_turn.metadata.get("response_snippet", "")
    char_count = int(last_turn.metadata.get("response_char_count", 0))

    corrected = _check_patterns(next_prompt, _CORRECTION_PATTERNS)
    completed = _check_patterns(next_prompt, _COMPLETION_PATTERNS)
    praised   = _check_patterns(next_prompt, _PRAISE_PATTERNS)
    abandoned = _check_patterns(next_prompt, _ABANDON_PATTERNS) or not next_prompt.strip()
    continued = bool(next_prompt.strip()) and not abandoned

    satisfaction = compute_satisfaction(next_prompt or None, response_snippet)
    # Normalise char_count to [0, 1] using same 4-char/token, 512-token scale
    token_cost = min(1.0, (char_count / 4.0) / 512.0)

    return FeedbackSignals(
        continued=continued,
        corrected=corrected,
        completed=completed,
        praised=praised,
        abandoned=abandoned,
        satisfaction=satisfaction,
        token_cost=token_cost,
    )


def signals_from_platform(raw: dict) -> FeedbackSignals:
    """Lift a platform-provided signal dict to FeedbackSignals.

    Used when the platform (e.g. Hermes) already computed signals
    from conversation_history and supplies them via platform_signals().
    """
    return FeedbackSignals(
        continued=bool(raw.get("continued", False)),
        corrected=bool(raw.get("corrected", False)),
        completed=bool(raw.get("completed", False)),
        praised=bool(raw.get("praised", False)),
        abandoned=bool(raw.get("abandoned", False)),
        satisfaction=float(raw.get("satisfaction", 0.0)),
        token_cost=float(raw.get("token_cost", 0.0)),
    )


def build_trajectory(
    signals: FeedbackSignals,
    last_turn: "TurnRecord",
    session_id: str,
    platform: str,
) -> dict:
    """Assemble the submit_trajectory() payload dict."""
    return {
        "producer": platform,
        "version": 1,
        "conversation_id": session_id,
        "user_text": last_turn.metadata.get("response_snippet", ""),
        "trajectory": {
            "continued":          signals.continued,
            "corrected":          signals.corrected,
            "completed":          signals.completed,
            "praised":            signals.praised,
            "abandoned":          signals.abandoned,
            "token_cost":         signals.token_cost,
            "satisfaction":       signals.satisfaction,
            "hexagram_evolution": [last_turn.hexagram_id],
        },
    }
