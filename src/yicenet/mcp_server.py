"""
YiCeNet MCP Server — context window management for Claude Code and MCP-compatible IDEs.

Exposes YiCeNetEngine as MCP tools so Claude can actively manage its own context:
  - yicenet_attend:   3ms lightweight context prescription (no hexagram overhead)
  - yicenet_predict:  full hexagram reasoning + context prescription
  - yicenet_feedback: submit reward signal to flywheel for continuous learning
  - yicenet_switch:   hot-swap model checkpoint (A/B testing)

Transport: stdio (Claude Code spawns this process; no port to manage).

Claude Code setup (~/.claude/settings.json):
  {
    "mcpServers": {
      "yicenet": {
        "command": "yicenet-serve",
        "env": {"YICENET_HOME": "/path/to/YiCeNet"}
      }
    }
  }

Or run install-claudecode-hooks.sh to configure automatically.
"""

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from yicenet.config import yicenet_checkpoint_dir, yicenet_home

mcp = FastMCP("yicenet")

_engine = None
_active_version: str = ""


def _get_engine():
    global _engine, _active_version
    if _engine is not None:
        return _engine
    from yicenet.yicenet_engine import YiCeNetEngine

    checkpoint_dir = yicenet_checkpoint_dir()
    reg_path = checkpoint_dir / "registry.json"
    ckpt = ""
    if reg_path.exists():
        try:
            reg = json.loads(reg_path.read_text(encoding="utf-8"))
            active = reg.get("active", {})
            ckpt = active.get("path", "")
            _active_version = active.get("version", "")
            if ckpt and not Path(ckpt).is_absolute():
                ckpt = str(checkpoint_dir / ckpt)
        except Exception:
            pass
    if not ckpt or not Path(ckpt).exists():
        ckpt = str(checkpoint_dir / "yicenet_v15.pt")
    _engine = YiCeNetEngine(checkpoint=ckpt, project_root=str(yicenet_home()))
    return _engine


@mcp.tool()
def yicenet_attend(
    task_text: str,
    session_id: str,
    turn_id: int = 0,
    turn_summary: str = "",
    environment: dict = None,
) -> dict:
    """
    Lightweight context prescription (~3ms, no hexagram routing).

    Encodes task_text, stores it in the session MemoryBank, then runs
    cross-attention against all prior turns to classify each as:
      - retain_turns:    keep full in context
      - summarize_turns: compress to 1-3 key facts each
      - discard_turns:   drop from reasoning entirely

    Call this when the conversation is growing long and you want guidance on
    which prior turns are most relevant. Up to 70% of old context can be pruned
    when attention entropy is low (highly focused session).

    session_id: stable identifier for this coding session.
                Recommended: sha256(cwd + YYYYMMDD)[:12]
    turn_id:    sequential turn number within the session (0-indexed).
    environment: optional structural signals to sharpen encoding.
                 Allowed keys: hour_of_day, session_turn, last_hexagram_id,
                 correction_rate, satisfaction_ema, attention_entropy,
                 last_tool_success (bool).
    """
    engine = _get_engine()
    result = engine.attend(
        task_text,
        session_id=session_id,
        turn_id=turn_id,
        turn_summary=turn_summary,
        environment=environment,
    )
    return result.get("context_prescription", result)


@mcp.tool()
def yicenet_predict(
    task_brief: str,
    session_id: str = "",
    temperature: float = 0.1,
    deterministic: bool = False,
    environment: dict = None,
    turn_id: int = 0,
) -> dict:
    """
    Full YiCeNet inference: hexagram structural reasoning + context prescription.

    Maps task_brief to one of 64 I-Ching hexagrams (structural archetypes),
    evaluates 8 candidate hexagrams by Q-value, and returns the recommended
    action sequence plus an optional context prescription if session_id is given.

    Use at the start of a complex multi-step task when you want both:
      1. Structural framing (which decision archetype applies — e.g. 乾=strong action,
         坤=receptive support, 屯=initial difficulty, 解=release of tension)
      2. Context management guidance (which prior turns to retain vs. discard)

    deterministic=True bypasses Gumbel exploration noise — use for rigid workflows
    where reproducibility matters more than exploration.

    environment: optional structural signals (16-dim) that sharpen hexagram routing.
        All keys are optional; unknown keys are silently ignored.
        The engine auto-computes hexagram chain signals when session_id is set:
          memory_bank_depth, hexagram_stability, hexagram_velocity,
          clan_diversity, hexagram_entropy, attention_entropy (from previous turn).

        Caller-supplied keys (override auto-computed where overlap):
          hour_of_day (int 0-23)         — time-of-day
          day_of_week (int 0-6)          — Mon=0 … Sun=6
          session_turn (int)             — conversation depth
          last_hexagram_id (int 0-63)    — previous hexagram ID
          correction_rate (float 0-1)    — recent correction fraction
          completed_rate (float 0-1)     — recent completion fraction
          praised_rate (float 0-1)       — recent praise fraction
          satisfaction_ema (float 0-1)   — smoothed satisfaction
          last_tool_success (bool)       — last tool call outcome (overrides satisfaction slot)

    Returns (in addition to standard hexagram fields):
        env_confidence (float 0-1): how certain the router is given current context
        context_status (str): "sufficient" | "partial" | "thin"
        context_hint (str): suggestion for improving confidence (absent when sufficient)
    """
    engine = _get_engine()
    return engine.predict(
        task_brief,
        temperature=temperature,
        deterministic=deterministic,
        return_prescription=bool(session_id),
        session_id=session_id,
        turn_id=turn_id,
        turn_summary="",
        environment=environment,
    )


@mcp.tool()
def yicenet_feedback(
    session_id: str,
    event: str,
    token_cost: int = 0,
) -> dict:
    """
    Submit a reward signal to the YiCeNet flywheel for continuous learning.

    event must be one of:
      - continued:  user sent a follow-up (positive signal, weight +0.3)
      - corrected:  user said the response was wrong (negative, weight -1.0)
      - completed:  task finished successfully (positive, weight +0.5)
      - praised:    user explicitly praised the response (strong positive, +1.0)
      - abandoned:  user left without follow-up (negative, weight -0.5)

    Call this after each meaningful turn outcome. The flywheel accumulates these
    signals and uses them to incrementally fine-tune YiCeNet every 6 hours.
    token_cost: total tokens used in this turn (input + output).
    """
    valid_events = {"continued", "corrected", "completed", "praised", "abandoned"}
    if event not in valid_events:
        return {"error": f"event must be one of {sorted(valid_events)}"}

    from yicenet.flywheel import submit_trajectory

    submit_trajectory({
        "producer": "claude-code",
        "version": 1,
        "conversation_id": session_id,
        "trajectory": {
            "continued": event == "continued",
            "corrected": event == "corrected",
            "completed": event == "completed",
            "praised": event == "praised",
            "abandoned": event == "abandoned",
            "token_cost": token_cost,
            "token_efficiency": 0.0,
        },
    })
    return {"ok": True, "session_id": session_id, "event": event}


@mcp.tool()
def yicenet_switch(checkpoint: str) -> dict:
    """
    Hot-swap YiCeNet to a different checkpoint without restarting the server.

    checkpoint: absolute path to a .pt checkpoint file, or a filename
                relative to the checkpoints/ directory.
    Used for A/B model comparison after the flywheel auto-trains a new version.
    """
    engine = _get_engine()
    success = engine.switch_model(checkpoint)
    return {"success": success, "active": checkpoint}


@mcp.resource("registry://active")
def registry_active() -> str:
    """Current active YiCeNet checkpoint and win_rate from registry.json."""
    reg_path = yicenet_checkpoint_dir() / "registry.json"
    if not reg_path.exists():
        return json.dumps({"error": "registry.json not found"})
    try:
        return reg_path.read_text(encoding="utf-8")
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
