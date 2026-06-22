"""
YiCeNet MCP Server — context window management for Claude Code and MCP-compatible IDEs.

Three operating modes:
  Mode 2 (pure MCP):   Claude explicitly calls yicenet_predict / yicenet_attend.
                       HookOrchestrator feedback loop runs via yicenet_turn_complete.
  Mode 3 (hybrid):     Same MCP tools available PLUS an HTTP side-channel so the
                       UserPromptSubmit / Stop hooks can reach the warm engine
                       without cold-starting a subprocess.

MCP tools:
  yicenet_attend        — 3ms lightweight context prescription
  yicenet_predict       — full hexagram routing + prescription
  yicenet_turn_complete — record response for next-turn feedback (replaces Stop hook)
  yicenet_feedback      — explicit reward signal to flywheel
  yicenet_switch        — hot-swap model checkpoint

Transport: stdio (Claude Code spawns this process; no port to manage).

Claude Code setup (~/.claude/settings.json):
  {
    "mcpServers": {
      "yicenet": { "command": "yicenet-serve" }
    }
  }

Hybrid mode additionally registers UserPromptSubmit / Stop hooks with
YICENET_MODE=hybrid — use ClaudeCodeInstaller.register_hybrid() to configure.
"""

import json

from mcp.server.fastmcp import FastMCP

from yicenet.config import yicenet_checkpoint_dir
from yicenet.engine_provider import EngineProvider

mcp = FastMCP("yicenet")

# ── Module-level orchestrator (MCP / daemon path) ─────────────────────────────

_mcp_orchestrator = None


def _mcp_orch():
    global _mcp_orchestrator
    if _mcp_orchestrator is None:
        from yicenet.tools.mcp_adapter import MCPAdapter
        from yicenet.hook_engine import HookOrchestrator
        _mcp_orchestrator = HookOrchestrator(MCPAdapter())
    return _mcp_orchestrator


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def yicenet_attend(
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
    import anyio

    def _run():
        engine = EngineProvider.get_engine()
        result = engine.attend(
            task_text,
            session_id=session_id,
            turn_id=turn_id,
            turn_summary=turn_summary,
            environment=environment,
        )
        return result.get("context_prescription", result)

    return await anyio.to_thread.run_sync(_run)


@mcp.tool()
async def yicenet_predict(
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
    import anyio

    def _run():
        payload = {
            "session_id": session_id,
            "turn_id": turn_id,
            "task_brief": task_brief,
        }
        if session_id:
            try:
                _mcp_orch().before_prediction(payload)
            except Exception:
                pass

        engine = EngineProvider.get_engine()
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

    return await anyio.to_thread.run_sync(_run)


@mcp.tool()
async def yicenet_turn_complete(
    session_id: str,
    response_snippet: str = "",
) -> dict:
    """
    Record that the assistant's turn is complete (equivalent to the Stop hook).

    Call this after each of your responses so YiCeNet can extract feedback
    signals for the next-turn prediction.  response_snippet is the first ~300
    characters of your reply — used for implicit feedback extraction.

    In hybrid mode this is called automatically by the Stop hook via IPC;
    in pure MCP mode you should call it yourself after each turn.
    """
    import anyio

    def _run():
        payload = {
            "session_id": session_id,
            "response_snippet": response_snippet,
        }
        try:
            _mcp_orch().on_turn_complete(payload)
        except Exception:
            pass
        return {"ok": True, "session_id": session_id}

    return await anyio.to_thread.run_sync(_run)


@mcp.tool()
async def yicenet_feedback(
    session_id: str,
    event: str,
    token_cost: int = 0,
) -> dict:
    """
    Submit an explicit reward signal to the YiCeNet flywheel for continuous learning.

    Use this alongside yicenet_turn_complete for richer feedback.  The flywheel
    accumulates these signals and uses them to incrementally fine-tune YiCeNet.

    event must be one of:
      - continued:  user sent a follow-up (positive signal, weight +0.3)
      - corrected:  user said the response was wrong (negative, weight -1.0)
      - completed:  task finished successfully (positive, weight +0.5)
      - praised:    user explicitly praised the response (strong positive, +1.0)
      - abandoned:  user left without follow-up (negative, weight -0.5)

    token_cost: total tokens used in this turn (input + output).
    """
    valid_events = {"continued", "corrected", "completed", "praised", "abandoned"}
    if event not in valid_events:
        return {"error": f"event must be one of {sorted(valid_events)}"}

    import anyio
    from yicenet.flywheel import submit_trajectory

    def _run():
        submit_trajectory({
            "producer": "claude-code-mcp",
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

    return await anyio.to_thread.run_sync(_run)


@mcp.tool()
async def yicenet_switch(checkpoint: str) -> dict:
    """
    Hot-swap YiCeNet to a different checkpoint without restarting the server.

    checkpoint: absolute path to a .pt checkpoint file, or a filename
                relative to the checkpoints/ directory.
    Used for A/B model comparison after the flywheel auto-trains a new version.
    """
    import anyio

    def _run():
        success = EngineProvider.switch(checkpoint)
        return {"success": success, "active": checkpoint}

    return await anyio.to_thread.run_sync(_run)


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
    import os
    import threading

    # 1. Start HTTP hook side-channel FIRST — before heavy imports — so the port
    #    file is written immediately and hook subprocesses can find the daemon.
    #    The hook server lazy-loads the engine on its first IPC request.
    if os.environ.get("YICENET_DISABLE_HOOK_SERVER", "").lower() not in ("1", "true"):
        try:
            from yicenet.daemon.hook_server import start_hook_server
            start_hook_server()
        except Exception:
            pass

    # 2. Configure MemoryBank for daemon mode.
    from yicenet.tools.mcp_adapter import MCPAdapter
    from yicenet.memory_bank import configure_memory_bank_for
    configure_memory_bank_for(MCPAdapter())

    # 3. Pre-warm heavy imports in a BACKGROUND thread so mcp.run() can start
    #    immediately and respond to the MCP initialize handshake without delay.
    #    Stdout/stderr are suppressed inside the thread to avoid deadlocking on
    #    the IOCP pipe that FastMCP's stdio transport sets up on Windows.
    def _prewarm():
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            try:
                from transformers import AutoTokenizer as _  # noqa: F401
                from yicenet.yicenet_engine import YiCeNetEngine as _  # noqa: F401
            except Exception:
                pass

    threading.Thread(target=_prewarm, name="yicenet-prewarm", daemon=True).start()

    mcp.run()


if __name__ == "__main__":
    main()
