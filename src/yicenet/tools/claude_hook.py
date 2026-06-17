"""
YiCeNet Claude Code hook implementations.

Called by the hook script installed by ClaudeCodeInstaller:
  pre_message_send()  — UserPromptSubmit: prescribe + inject hexagram context
  post_tool_use()     — PostToolUse: submit flywheel reward signal
  stop()              — Stop: flush MemoryBank for the session

Hook input arrives on stdin as JSON (Claude Code hook payload format).
|Output to stdout is injected into <user-prompt-submit-hook> context.
|  Line 1: compact label (plain text, user-visible)
|  Line 2: JSON metadata (Claude reads prescription, env_confidence, etc.)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import datetime

# Windows cp1252 stdout/stderr rejects CJK characters; reconfigure to UTF-8 globally.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _session_id(payload: dict) -> str:
    """Derive a stable 12-char session_id from the hook payload."""
    cc_id = payload.get("session_id", "")
    if cc_id:
        return cc_id.replace("-", "")[:12]
    cwd = payload.get("cwd", os.getcwd())
    date = datetime.datetime.now().strftime("%Y%m%d")
    return hashlib.sha256(f"{cwd}{date}".encode()).hexdigest()[:12]


def _read_payload() -> dict:
    try:
        raw = sys.stdin.read().strip()
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def pre_message_send() -> None:
    """UserPromptSubmit hook: prescribe context, inject hexagram framing."""
    payload = _read_payload()
    session_id = _session_id(payload)
    prompt = payload.get("prompt", "")

    # Infer turn_id from number of existing messages (each pair = 1 turn)
    messages = payload.get("messages", [])
    turn_id = max(0, len(messages) - 1)

    try:
        from yicenet.engine_provider import EngineProvider
        from yicenet.providers import ProviderRegistry

        engine = EngineProvider.get_engine()

        env = {
            "hour_of_day": datetime.datetime.now().hour,
            "session_turn": turn_id,
        }

        prescription = engine.prescribe(
            task_text=prompt or "general task",
            session_id=session_id,
            turn_id=turn_id,
            environment=env,
        ).to_dict()  # Prescription → dict for JSON serialization

        result = engine.predict(
            prompt or "general task",
            temperature=0.1,
            environment=env,
        )

        # Compact human-readable label via display layer
        display = ProviderRegistry.default().display
        chain = None
        if session_id and display.needs_chain:
            from yicenet.memory_bank import get_memory_bank
            chain = get_memory_bank().get_hexagram_history(session_id)
        label = display.render(result, chain=chain)

        # ── Line 1: stdout → compact label (plain text, user-visible) ──
        print(label, flush=True)

        # ── Line 2: stdout → JSON metadata (Claude reads from context) ──
        output = {
            "yicenet": {
                "session_id": session_id,
                "turn_id": turn_id,
                "label": label,
                "hexagram": result.get("selected_hexagram_name", ""),
                "action": result.get("action_name", ""),
                "env_confidence": result.get("env_confidence", 0.0),
                "context_status": result.get("context_status", "thin"),
                "prescription": prescription,
            }
        }
        print(json.dumps(output, ensure_ascii=False), flush=True)

    except Exception:
        # Never block the user — fail silently
        sys.exit(0)


def post_tool_use() -> None:
    """PostToolUse hook: submit continued signal to flywheel."""
    payload = _read_payload()
    session_id = _session_id(payload)
    try:
        from yicenet.flywheel import submit_trajectory
        submit_trajectory({
            "producer": "claude-code-hook",
            "version": 1,
            "conversation_id": session_id,
            "trajectory": {
                "continued": True,
                "corrected": False,
                "completed": False,
                "praised": False,
                "abandoned": False,
                "token_cost": 0,
                "token_efficiency": 0.0,
            },
        })
    except Exception:
        pass


def stop() -> None:
    """Stop hook: flush MemoryBank for this session."""
    payload = _read_payload()
    session_id = _session_id(payload)
    try:
        from yicenet.memory_bank import get_memory_bank
        get_memory_bank().flush_session(session_id)
    except Exception:
        pass
