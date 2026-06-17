"""
YiCeNet Claude Code hook implementations.

Entry points (called by _claude_runner.py):
  pre_message_send()  — UserPromptSubmit
  post_tool_use()     — PostToolUse
  stop()              — Stop

stdout → JSON injected into Claude's context.
stderr → visible to the user in the terminal.

ClaudeCodeAdapter is also imported by configure_memory_bank_for() in the
runner so the MemoryBank singleton is set up before any engine code runs.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys

# Reconfigure to UTF-8 so CJK characters print on Windows.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ── Platform adapter ──────────────────────────────────────────────────────────

class ClaudeCodeAdapter:
    """Translates Claude Code hook payloads to domain primitives."""

    platform_id = "claude-code"
    process_model = "subprocess"

    def session_id(self, payload: dict) -> str:
        cc_id = payload.get("session_id", "")
        if cc_id:
            return cc_id.replace("-", "")[:12]
        cwd = payload.get("cwd", os.getcwd())
        date = datetime.datetime.now().strftime("%Y%m%d")
        return hashlib.sha256(f"{cwd}{date}".encode()).hexdigest()[:12]

    def turn_id(self, payload: dict) -> int:
        return max(0, len(payload.get("messages", [])) - 1)

    def prompt(self, payload: dict) -> str:
        return payload.get("prompt", "")

    def assistant_response(self, payload: dict) -> str:
        return _read_last_assistant(payload)

    def platform_signals(self, payload: dict):
        return None  # CC hooks have no pre-computed signals


# ── Private transcript helper ─────────────────────────────────────────────────

def _read_last_assistant(payload: dict) -> str:
    """Read the last assistant message from the Claude Code transcript."""
    from pathlib import Path

    # Claude Code Stop hook provides transcript_path directly
    tp = payload.get("transcript_path", "")
    if tp and Path(tp).exists():
        return _last_asst_from_file(Path(tp))

    # Fallback: locate transcript by session UUID
    cc_id = payload.get("session_id", "")
    if cc_id:
        projects = Path.home() / ".claude" / "projects"
        for jsonl in projects.glob(f"**/{cc_id}.jsonl"):
            return _last_asst_from_file(jsonl)

    return ""


def _last_asst_from_file(path) -> str:
    from pathlib import Path
    last = ""
    try:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                obj = json.loads(line.strip())
                msg = obj.get("message", {})
                role = (msg.get("role", "") if isinstance(msg, dict)
                        else obj.get("role", ""))
                if role != "assistant":
                    continue
                content = (msg.get("content", "") if isinstance(msg, dict)
                           else obj.get("content", ""))
                if isinstance(content, list):
                    text = " ".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                else:
                    text = str(content)
                if text.strip():
                    last = text
            except (json.JSONDecodeError, AttributeError):
                continue
    except OSError:
        pass
    return last


# ── Module-level singleton ────────────────────────────────────────────────────

_adapter = ClaudeCodeAdapter()
_orchestrator = None  # lazy-init to avoid import at module load


def _orch():
    global _orchestrator
    if _orchestrator is None:
        from yicenet.hook_engine import HookOrchestrator
        _orchestrator = HookOrchestrator(_adapter)
    return _orchestrator


# ── Hook entry points ─────────────────────────────────────────────────────────

def _read_payload() -> dict:
    try:
        raw = sys.stdin.read().strip()
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def pre_message_send() -> None:
    """UserPromptSubmit: extract prior-turn feedback, then prescribe hexagram."""
    payload = _read_payload()
    session_id = _adapter.session_id(payload)
    prompt = _adapter.prompt(payload)
    turn_id = _adapter.turn_id(payload)

    # Submit feedback for the previous turn before doing this turn's prediction
    try:
        _orch().before_prediction(payload)
    except Exception:
        pass

    try:
        from yicenet.engine_provider import EngineProvider
        from yicenet.providers import ProviderRegistry

        engine = EngineProvider.get_engine()
        env = {
            "hour_of_day": datetime.datetime.now().hour,
            "session_turn": turn_id,
        }

        result = engine.predict(
            prompt or "general task",
            temperature=0.1,
            environment=env,
            session_id=session_id,
            turn_id=turn_id,
            return_prescription=True,
        )
        prescription = result.get("context_prescription", {})

        display = ProviderRegistry.default().display
        chain = None
        if session_id and display.needs_chain:
            from yicenet.memory_bank import get_memory_bank
            chain = get_memory_bank().get_hexagram_history(session_id)
        label = display.render(result, chain=chain)

        sys.stderr.write(label + "\n")
        sys.stderr.flush()

        print(json.dumps({
            "yicenet": {
                "session_id":    session_id,
                "turn_id":       turn_id,
                "label":         label,
                "hexagram":      result.get("selected_hexagram_name", ""),
                "action":        result.get("action_name", ""),
                "env_confidence": result.get("env_confidence", 0.0),
                "context_status": result.get("context_status", "thin"),
                "prescription":  prescription,
            }
        }, ensure_ascii=False), flush=True)

    except Exception:
        sys.exit(0)


def post_tool_use() -> None:
    """PostToolUse: no-op for feedback (handled at turn boundaries).

    Feedback signals require the next user message to be present.
    before_prediction() in pre_message_send() handles this correctly.
    We keep this hook registered so Claude Code fires it, but take no action.
    """
    pass


def stop() -> None:
    """Stop: record response metadata for next turn's feedback extraction."""
    payload = _read_payload()
    try:
        _orch().on_turn_complete(payload)
    except Exception:
        pass
    # MemoryBank.flush_session() is called inside on_turn_complete() for subprocesses.
