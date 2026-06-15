"""
YiCeNet environment context utilities — platform-agnostic structural signals.

Converts platform-specific runtime state into a normalized 7-dim structural
vector that conditions the EnvironmentProjector inside YiCeNet, sharpening
hexagram routing without embedding any semantic content into model weights.

DESIGN PRINCIPLE (道 not 技):
  Allowed signal types — structural, timing, rate-based:
    hour_of_day, session_turn, last_hexagram_id,
    correction_rate, satisfaction_ema, attention_entropy, last_tool_success

  Forbidden — semantic content that would seep into model weights:
    project_type, domain, current_file, language, framework, user_name, ...

  Unknown keys in the env dict are silently ignored so the API stays
  forward-compatible as platforms add richer signals over time.

Usage (Hermes plugin):
    env = {
        "hour_of_day": datetime.now().hour,
        "session_turn": session_store.get_turn(session_id),
        "last_hexagram_id": memory_bank.get_last_hexagram(session_id),
        "correction_rate": compute_recent_correction(session_id),
    }
    result = engine.predict(text, environment=env)

Usage (Claude Code MCP):
    yicenet_predict(
        task_brief="search knowledge base",
        session_id="abc123",
        environment={"session_turn": 7, "last_tool_success": True}
    )
"""

from __future__ import annotations

import math
from typing import Optional

import torch

ENV_DIM = 7
"""Dimensionality of the structural environment vector fed to EnvironmentProjector."""

# ── 7-dim vector layout ────────────────────────────────────────────────────
# [0]  sin(2π × hour / 24)      — time-of-day phase (sine component)
# [1]  cos(2π × hour / 24)      — time-of-day phase (cosine component)
# [2]  session_turn / 50.0      — conversation depth, capped at 1.0
# [3]  last_hexagram_id / 63.0  — previous hexagram state (0.0 when unknown)
# [4]  correction_rate          — fraction of recent turns that were corrected (0-1)
# [5]  satisfaction_ema         — smoothed tool/turn success signal (0-1)
# [6]  attention_entropy / 4.0  — MemoryBank focus level (0=sharp, 1=diffuse)
# ──────────────────────────────────────────────────────────────────────────

_LOG64 = math.log(64)  # ≈ 4.158 — max entropy for 64-way routing distribution


def build_env_vec(context: Optional[dict]) -> Optional[torch.Tensor]:
    """Convert environment context dict → normalized 7-dim structural vector.

    Returns None when context is absent or empty; the EnvironmentProjector
    in YiCeNet treats None as a no-op (zero residual), preserving backward
    compatibility with existing checkpoints.

    Args:
        context: dict with any subset of structural signal keys.
                 Unknown keys are silently ignored.

    Returns:
        Float32 tensor of shape (7,), or None.
    """
    if not context:
        return None

    hour = float(context.get("hour_of_day", 12.0))
    turn = float(context.get("session_turn", 0.0))
    last_hx = float(context.get("last_hexagram_id", -1.0))
    corr = float(context.get("correction_rate", 0.0))
    sat = float(context.get("satisfaction_ema", 0.5))
    attn_e = float(context.get("attention_entropy", 2.0))
    tool_ok = context.get("last_tool_success")

    sin_h = math.sin(2.0 * math.pi * hour / 24.0)
    cos_h = math.cos(2.0 * math.pi * hour / 24.0)
    turn_n = min(turn / 50.0, 1.0)
    hx_n = last_hx / 63.0 if last_hx >= 0.0 else 0.0
    corr_n = max(0.0, min(1.0, corr))
    if tool_ok is not None:
        sat_n = 1.0 if bool(tool_ok) else 0.0
    else:
        sat_n = max(0.0, min(1.0, sat))
    attn_n = max(0.0, min(1.0, attn_e / 4.0))

    return torch.tensor(
        [sin_h, cos_h, turn_n, hx_n, corr_n, sat_n, attn_n],
        dtype=torch.float32,
    )


def compute_env_confidence(
    probe_list: Optional[list],
    q_values: Optional[list],
) -> tuple[float, str, str]:
    """Derive routing confidence from existing probe + Q-value structural signals.

    This function reads probe[2] (logit_entropy) and probe[7] (q_gap) — both
    purely structural, zero semantic content — to estimate how certain the
    hexagram router is given the current input.

    Low confidence indicates the model lacks structural context to route
    confidently; the hint suggests which structural signal types to add.
    Note: the hint describes *signal types*, not content recommendations.

    Args:
        probe_list: 9-element probe vector from predict() (may be None).
        q_values:   8 Q-values from candidate evaluation (may be None/empty).

    Returns:
        (confidence, status, hint)
            confidence: float 0-1
            status:     "sufficient" | "partial" | "thin"
            hint:       empty string when sufficient, else short suggestion
    """
    if not probe_list or not q_values:
        return 0.5, "partial", ""

    # probe_list[2] = logit_entropy (high = diffuse = uncertain routing)
    # probe_list[7] = q_gap (small = two candidates nearly tied = uncertain)
    logit_entropy = float(probe_list[2]) if len(probe_list) > 2 else 2.0
    q_gap = float(probe_list[7]) if len(probe_list) > 7 else 0.0

    # q_conf: sigmoid of gap×3 — large gap → confident
    q_conf = 1.0 / (1.0 + math.exp(-q_gap * 3.0))
    # e_conf: fraction of entropy headroom used (low entropy = high conf)
    e_conf = 1.0 - (logit_entropy / _LOG64)

    confidence = round(q_conf * 0.6 + e_conf * 0.4, 3)
    confidence = max(0.0, min(1.0, confidence))

    if confidence >= 0.65:
        return confidence, "sufficient", ""
    elif confidence >= 0.40:
        return (
            confidence,
            "partial",
            "session_turn and correction_rate would sharpen this prediction",
        )
    else:
        return (
            confidence,
            "thin",
            "low routing confidence — consider providing: "
            "session_turn, last_hexagram_id, correction_rate",
        )
