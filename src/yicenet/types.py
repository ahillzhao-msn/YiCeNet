"""
YiCeNet public type contracts — single source of truth for all shared types.

Import from here, not from individual modules.
"""
from __future__ import annotations

from typing import TypedDict

# Re-export from canonical homes so callers only import from types.py
from .cross_attention import Prescription          # noqa: F401
from .memory_bank import TurnRecord                # noqa: F401
from .datasource import Sample                     # noqa: F401


class PredictionResult(TypedDict, total=False):
    """Contract for YiCeNetEngine.predict() return value.

    Keys marked optional (total=False) may be absent depending on call flags:
      - context_hint: absent when context_status == "sufficient"
      - context_prescription: present only when return_prescription=True
    """
    hexagram_id: int
    hexagram_name: str
    hexagram_number: int
    hexagram_pattern: str            # multi-line "—" / "- -" string
    best_candidate: int
    selected_hexagram_id: int
    selected_hexagram_name: str
    candidates: list[dict]           # [{index, hexagram_id, hexagram_name, q_value}]
    action_id: int
    action_name: str
    q_values: list[float]            # 8 elements
    temperature: float
    deterministic: bool
    probes: list[float]              # 9 elements
    env_confidence: float
    context_status: str              # "sufficient" | "partial" | "thin"
    context_hint: str
    context_prescription: dict


class EnvAnalysis(TypedDict, total=False):
    """Contract for YiCeNetEngine.analyze() return value.

    Fast path (~3ms): encode + probes only, no hexagram routing.
    """
    probes: list[float]
    env_confidence: float
    context_status: str
    context_hint: str
