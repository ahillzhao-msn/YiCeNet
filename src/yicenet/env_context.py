"""
YiCeNet environment context utilities — platform-agnostic structural signals.

Converts platform-specific runtime state into a normalized 16-dim structural
vector that conditions the EnvironmentProjector inside YiCeNet, sharpening
hexagram routing without embedding any semantic content into model weights.

DESIGN PRINCIPLE (道 not 技):
  Allowed signal types — structural, timing, rate-based, chain-dynamic:
    hour_of_day, day_of_week, session_turn, memory_bank_depth,
    last_hexagram_id, hexagram_stability, hexagram_velocity,
    clan_diversity, hexagram_entropy, last_tool_success,
    correction_rate, completed_rate, praised_rate,
    satisfaction_ema, attention_entropy

  Forbidden — semantic content that would seep into model weights:
    project_type, domain, current_file, language, framework, user_name, ...

  Unknown keys are silently ignored; missing keys produce 0.0 in their slot
  (which the zero-initialized EnvironmentProjector treats as no contribution).
  This means any platform can pass a partial dict and degrade gracefully.

16-Dim Vector Layout
────────────────────────────────────────────────────────────────────────────
Group            Slot  Signal                       Notes
────────────────────────────────────────────────────────────────────────────
Time phase        [0]  sin(2π × hour / 24)          (0,0) when not provided
                  [1]  cos(2π × hour / 24)
                  [2]  sin(2π × day_of_week / 7)    (0,0) when not provided
                  [3]  cos(2π × day_of_week / 7)
────────────────────────────────────────────────────────────────────────────
Session depth     [4]  session_turn / 50             how deep in the session
                  [5]  memory_bank_depth / 100       turns stored in MemoryBank
                  [6]  last_hexagram_id / 63         most recent hexagram (0=unknown)
                  [7]  hexagram_stability / 20       consecutive same-hexagram turns
────────────────────────────────────────────────────────────────────────────
Hexagram chain    [8]  hexagram_velocity             jump_distance EMA / 63 (0-1)
dynamics          [9]  clan_diversity                visited clans / 8 (0-1)
(卦链动态)       [10]  hexagram_entropy              Shannon entropy of recent dist (0-1)
                 [11]  last_tool_success_bin         1.0 success / 0.5 unknown / 0.0 fail
────────────────────────────────────────────────────────────────────────────
Quality signals  [12]  correction_rate               corrected turns / total
                 [13]  completed_rate                completed turns / total
                 [14]  praised_rate                  praised turns / total
────────────────────────────────────────────────────────────────────────────
Attention        [15]  attention_entropy / 4         MemoryBank focus level (0-1)
────────────────────────────────────────────────────────────────────────────

Auto-computed by the engine when session_id is provided (callers do not need
to supply these manually):  memory_bank_depth, hexagram_stability,
hexagram_velocity, clan_diversity, hexagram_entropy.

Everything else can be supplied by the calling platform.  Slots that are
left at 0 produce zero gradient contribution from the zero-initialized
EnvironmentProjector until training teaches them to be meaningful.

Usage (Hermes plugin — partial dict, engine fills the rest):
    env = {
        "hour_of_day": datetime.now().hour,
        "day_of_week": datetime.now().weekday(),
        "correction_rate": compute_recent_correction(session_id),
    }
    result = engine.predict(text, session_id=session_id, environment=env)

Usage (Claude Code MCP — minimal dict):
    yicenet_predict(task_brief="...", session_id="abc",
                    environment={"last_tool_success": True})
"""

from __future__ import annotations

import math
from typing import Optional

import torch

ENV_DIM = 16
"""Dimensionality of the structural environment vector fed to EnvironmentProjector."""

_LOG64 = math.log(64)


def env_vec_dropout(
    env_vec: torch.Tensor,
    p: float = 0.3,
    training: bool = True,
) -> torch.Tensor:
    """Randomly zero individual env_vec slots during training.

    Each slot is independently masked with probability p so the
    EnvironmentProjector learns to operate robustly on any subset of
    provided signals.  At inference time (training=False) returns env_vec
    unchanged, preserving the full structural context.

    Works on both 1-D (16,) and batched (B, 16) tensors.
    """
    if not training or p <= 0.0:
        return env_vec
    mask = (torch.rand_like(env_vec) > p).float()
    return env_vec * mask


def build_env_vec(context: Optional[dict]) -> Optional[torch.Tensor]:
    """Convert environment context dict -> normalized 16-dim structural vector.

    Returns None when context is absent or empty; the EnvironmentProjector
    treats None as a no-op (h += 0), preserving full backward compatibility.

    Unknown keys are silently ignored.  Missing keys default to 0.0 (or 0.5
    for satisfaction/tool-success where neutral differs from absent).

    Args:
        context: dict with any subset of the structural signal keys documented
                 at the module level.

    Returns:
        Float32 tensor of shape (16,), or None.
    """
    if not context:
        return None

    # ── Time phase (slots 0-3) ─────────────────────────────────────────────
    hour = context.get("hour_of_day")
    if hour is not None:
        h = float(hour)
        sin_h = math.sin(2.0 * math.pi * h / 24.0)
        cos_h = math.cos(2.0 * math.pi * h / 24.0)
    else:
        sin_h = cos_h = 0.0  # unknown → zero contribution

    dow = context.get("day_of_week")
    if dow is not None:
        d = float(dow)
        sin_d = math.sin(2.0 * math.pi * d / 7.0)
        cos_d = math.cos(2.0 * math.pi * d / 7.0)
    else:
        sin_d = cos_d = 0.0  # unknown → zero contribution

    # ── Session depth (slots 4-7) ──────────────────────────────────────────
    turn_n    = min(float(context.get("session_turn", 0.0)) / 50.0, 1.0)
    depth_n   = min(float(context.get("memory_bank_depth", 0.0)) / 100.0, 1.0)
    last_hx   = float(context.get("last_hexagram_id", -1.0))
    hx_n      = last_hx / 63.0 if last_hx >= 0.0 else 0.0
    stab_n    = min(float(context.get("hexagram_stability", 0.0)) / 20.0, 1.0)

    # ── Hexagram chain dynamics (slots 8-11) ───────────────────────────────
    vel_n     = max(0.0, min(1.0, float(context.get("hexagram_velocity", 0.0))))
    clan_n    = max(0.0, min(1.0, float(context.get("clan_diversity", 0.0))))
    ent_n     = max(0.0, min(1.0, float(context.get("hexagram_entropy", 0.0))))
    tool_ok   = context.get("last_tool_success")
    if tool_ok is None:
        tool_bin = 0.5
    else:
        tool_bin = 1.0 if bool(tool_ok) else 0.0

    # ── Quality signals (slots 12-14) ─────────────────────────────────────
    corr_n    = max(0.0, min(1.0, float(context.get("correction_rate", 0.0))))
    comp_n    = max(0.0, min(1.0, float(context.get("completed_rate", 0.0))))
    praise_n  = max(0.0, min(1.0, float(context.get("praised_rate", 0.0))))

    # ── Attention entropy (slot 15) ────────────────────────────────────────
    attn_n    = max(0.0, min(1.0, float(context.get("attention_entropy", 0.0)) / 4.0))

    return torch.tensor(
        [
            sin_h, cos_h, sin_d, cos_d,          # 0-3 time
            turn_n, depth_n, hx_n, stab_n,        # 4-7 session
            vel_n, clan_n, ent_n, tool_bin,        # 8-11 chain
            corr_n, comp_n, praise_n,              # 12-14 quality
            attn_n,                                # 15 attention
        ],
        dtype=torch.float32,
    )


def compute_env_confidence(
    probe_list: Optional[list],
    q_values: Optional[list],
) -> tuple[float, str, str]:
    """Derive routing confidence from the existing 9-dim probe vector.

    Reads probe[2] (logit_entropy) and probe[6] (q_gap) — purely structural
    signals extracted after inference — to estimate how certain the hexagram
    router is given the current input.  No semantic content is used.

    Args:
        probe_list: 9-element probe vector from predict() (may be None).
        q_values:   8 Q-values from candidate evaluation (may be None/empty).

    Returns:
        (confidence, status, hint)
            confidence: float 0-1
            status:     "sufficient" | "partial" | "thin"
            hint:       empty string when sufficient; type suggestion otherwise
    """
    if not probe_list or not q_values:
        return 0.5, "partial", ""

    logit_entropy = float(probe_list[2]) if len(probe_list) > 2 else 2.0
    q_gap         = float(probe_list[6]) if len(probe_list) > 6 else 0.0  # [6]=q_gap, [7]=jump_distance

    q_conf     = 1.0 / (1.0 + math.exp(-q_gap * 3.0))
    e_conf     = 1.0 - (logit_entropy / _LOG64)
    confidence = round(q_conf * 0.6 + e_conf * 0.4, 3)
    confidence = max(0.0, min(1.0, confidence))

    if confidence >= 0.65:
        return confidence, "sufficient", ""
    elif confidence >= 0.40:
        return (
            confidence,
            "partial",
            "session_turn, correction_rate, or last_tool_success would sharpen this prediction",
        )
    else:
        return (
            confidence,
            "thin",
            "low routing confidence — consider providing: "
            "session_turn, last_hexagram_id, correction_rate, last_tool_success",
        )
