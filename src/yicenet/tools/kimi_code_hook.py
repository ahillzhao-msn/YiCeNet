"""
YiCeNet Kimi Code hook adapter.

KimiCodeAdapter implements only what is specific to Kimi Code CLI:
  - session_id: derived from Kimi Code's session_id or cwd+date hash
  - assistant_response: read from payload if Kimi Code provides it
  - process_model: "subprocess" (python hook script) or "daemon" (native yicenet-hook client)

All shared prediction and hook lifecycle logic lives in HooksAdapter.

Entry points (called by the plugin hook script):
  pre_message_send([payload])  — UserPromptSubmit
  stop([payload])              — Stop
  post_tool_use([payload])     — PostToolUse reward signal
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

from yicenet.tools.hooks_adapter import HooksAdapter

# Reconfigure to UTF-8 so CJK characters print on Windows.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


class KimiCodeAdapter(HooksAdapter):
    """Platform adapter for Kimi Code CLI hooks."""

    _platform_id = "kimi-code"

    def __init__(self, process_model: str = "subprocess") -> None:
        self._process_model = process_model

    @property
    def platform_id(self) -> str:
        return self._platform_id

    @property
    def process_model(self) -> str:
        return self._process_model

    def session_id(self, payload: dict) -> str:
        """Derive a stable session id from Kimi Code payload."""
        sid = payload.get("session_id", "")
        if sid:
            return sid.replace("-", "")[:12]
        cwd = payload.get("cwd", os.getcwd())
        date = datetime.datetime.now().strftime("%Y%m%d")
        return hashlib.sha256(f"{cwd}{date}".encode()).hexdigest()[:12]

    def turn_id(self, payload: dict) -> int:
        """Monotonically increasing turn counter (0-indexed).

        Kimi Code does not expose a direct turn_id. We derive it from the
        messages list when available; otherwise we fall back to 0.
        """
        tid = payload.get("turn_id")
        if tid is not None:
            return int(tid)
        history = (
            payload.get("messages")
            or payload.get("conversation_history")
            or []
        )
        return max(0, len(history) - 1)

    def prompt(self, payload: dict) -> str:
        """Current user message text (available at UserPromptSubmit)."""
        # Kimi Code UserPromptSubmit likely provides the raw prompt text
        if "prompt" in payload:
            return str(payload["prompt"])
        messages = payload.get("messages") or [{}]
        last = messages[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)

    def assistant_response(self, payload: dict) -> str:
        """Full assistant response text (available at Stop / PostToolUse).

        Kimi Code's Stop event payload is not publicly documented; we read
        it if present and otherwise return an empty string.  Empty response
        means the next turn's feedback extraction will be weaker, but it is
        not fatal.
        """
        resp = payload.get("assistant_response") or payload.get("response", "")
        if isinstance(resp, dict):
            return str(resp.get("content", ""))
        return str(resp) if resp else ""

    def platform_signals(self, payload: dict) -> Optional[dict]:
        """Pre-computed feedback signals are not available from Kimi Code."""
        return None


# ── Module-level singleton ────────────────────────────────────────────────────

_adapter = KimiCodeAdapter()


def pre_message_send(payload: "dict | None" = None) -> None:
    """UserPromptSubmit entry point.

    Reads the hook payload from stdin if not supplied, runs
    predict_for_turn_payload, and prints the result JSON to stdout so Kimi
    Code appends it to the context window.
    """
    if payload is None:
        payload = _adapter._read_payload()
    result = _adapter.predict_for_turn_payload(payload)
    if result is None:
        sys.exit(0)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def stop(payload: "dict | None" = None) -> None:
    """Stop / post-LLM entry point."""
    if payload is None:
        payload = _adapter._read_payload()
    _adapter.stop(payload)


def _submit_tool_trajectory(payload: dict, success: bool) -> None:
    """Submit a lightweight trajectory for tool outcomes."""
    session_id = _adapter.session_id(payload)
    if not session_id:
        return
    try:
        from yicenet.flywheel import submit_trajectory

        submit_trajectory({
            "producer": "kimi-code",
            "version": 1,
            "conversation_id": session_id,
            "trajectory": {
                "continued": False,
                "corrected": False,
                "completed": success,
                "praised": False,
                "abandoned": not success,
                "token_cost": 0,
                "token_efficiency": 0.0,
            },
        })
    except Exception:
        pass


def post_tool_use(payload: "dict | None" = None) -> None:
    """PostToolUse reward signal entry point."""
    if payload is None:
        payload = _adapter._read_payload()
    _submit_tool_trajectory(payload, success=True)


def post_tool_use_failure(payload: "dict | None" = None) -> None:
    """PostToolUseFailure reward signal entry point."""
    if payload is None:
        payload = _adapter._read_payload()
    _submit_tool_trajectory(payload, success=False)


def pre_tool_use(payload: "dict | None" = None) -> None:
    """PreToolUse entry point.

    YiCeNet does not block tools by default; it runs a lightweight predict
    on the tool intent and prints a structural hint to stderr.  The hook
    always exits 0 (allow) so the agent workflow is not interrupted.
    """
    if payload is None:
        payload = _adapter._read_payload()

    tool_name = payload.get("tool_name", "")
    task = f"PreToolUse: {tool_name}"
    try:
        result = _adapter.predict_for_turn_payload({
            **_merge_payload(payload),
            "prompt": task,
        })
    except Exception:
        result = None

    # Print nothing to stdout => allow; stderr already has the hexagram label.
    if result is None:
        sys.exit(0)
    # Kimi Code ignores additionalContext from PreToolUse today; keep stdout clean.
    sys.exit(0)


def session_start(payload: "dict | None" = None) -> None:
    """SessionStart entry point: initialise the MemoryBank session."""
    if payload is None:
        payload = _adapter._read_payload()
    session_id = _adapter.session_id(payload)
    if not session_id:
        return
    try:
        from yicenet.memory_bank import get_memory_bank
        bank = get_memory_bank()
        bank.init_session(session_id)
        bank.flush_session(session_id)
    except Exception:
        pass


def session_end(payload: "dict | None" = None) -> None:
    """SessionEnd entry point: flush any remaining turn metadata."""
    if payload is None:
        payload = _adapter._read_payload()
    session_id = _adapter.session_id(payload)
    if not session_id:
        return
    try:
        from yicenet.memory_bank import get_memory_bank
        bank = get_memory_bank()
        bank.init_session(session_id)
        bank.flush_session(session_id)
    except Exception:
        pass


def pre_compact(payload: "dict | None" = None) -> None:
    """PreCompact entry point: run attend for context prescription.

    Kimi Code ignores return values for PreCompact, so this is observation-only.
    The prescription is logged to stderr for debugging.
    """
    if payload is None:
        payload = _adapter._read_payload()

    session_id = _adapter.session_id(payload)
    if not session_id:
        return

    try:
        from yicenet.engine_provider import EngineProvider
        engine = EngineProvider.get_engine()
        result = engine.attend(
            "context compaction",
            session_id=session_id,
            turn_id=_adapter.turn_id(payload),
            turn_summary="pre-compact attention",
        )
        # Log compact prescription to stderr for observability.
        sys.stderr.write(json.dumps(result.get("context_prescription", result),
                                    ensure_ascii=False) + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def post_compact(payload: "dict | None" = None) -> None:
    """PostCompact entry point: record compaction event."""
    if payload is None:
        payload = _adapter._read_payload()
    _submit_event_trajectory(payload, "compact")


def subagent_start(payload: "dict | None" = None) -> None:
    """SubagentStart entry point."""
    if payload is None:
        payload = _adapter._read_payload()
    _record_prediction(f"subagent_start:{payload.get('subagent_name', '')}", payload)


def subagent_stop(payload: "dict | None" = None) -> None:
    """SubagentStop entry point."""
    if payload is None:
        payload = _adapter._read_payload()
    _submit_event_trajectory(payload, "subagent_stop")


def stop_failure(payload: "dict | None" = None) -> None:
    """StopFailure entry point."""
    if payload is None:
        payload = _adapter._read_payload()
    _submit_event_trajectory(payload, "stop_failure")


def interrupt(payload: "dict | None" = None) -> None:
    """Interrupt entry point."""
    if payload is None:
        payload = _adapter._read_payload()
    _submit_event_trajectory(payload, "interrupt")


def permission_request(payload: "dict | None" = None) -> None:
    """PermissionRequest observation hook."""
    if payload is None:
        payload = _adapter._read_payload()
    _record_prediction(f"permission_request:{payload.get('tool_name', '')}", payload)


def permission_result(payload: "dict | None" = None) -> None:
    """PermissionResult observation hook."""
    if payload is None:
        payload = _adapter._read_payload()
    _submit_event_trajectory(payload, "permission_result")


def notification(payload: "dict | None" = None) -> None:
    """Notification observation hook."""
    if payload is None:
        payload = _adapter._read_payload()
    _record_prediction(f"notification:{payload.get('notification_type', '')}", payload)


# ── Helpers for the new observation hooks ─────────────────────────────────────

def _merge_payload(payload: dict) -> dict:
    """Return a copy of payload augmented with prompt/task keys if absent."""
    merged = dict(payload)
    if "messages" not in merged:
        merged["messages"] = []
    return merged


def _record_prediction(task: str, payload: dict) -> None:
    """Run a lightweight predict for an observation-only event."""
    try:
        _adapter.predict_for_turn_payload({**_merge_payload(payload), "prompt": task})
    except Exception:
        pass


def _submit_event_trajectory(payload: dict, event_type: str) -> None:
    """Submit a neutral trajectory marker for an observation-only event."""
    session_id = _adapter.session_id(payload)
    if not session_id:
        return
    try:
        from yicenet.flywheel import submit_trajectory
        submit_trajectory({
            "producer": "kimi-code",
            "version": 1,
            "conversation_id": session_id,
            "trajectory": {
                "continued": False,
                "corrected": False,
                "completed": False,
                "praised": False,
                "abandoned": False,
                "token_cost": 0,
                "token_efficiency": 0.0,
                "event_type": event_type,
            },
        })
    except Exception:
        pass


# ── CLI dispatcher ────────────────────────────────────────────────────────────

_COMMANDS = {
    "pre_message_send": pre_message_send,
    "stop": stop,
    "post_tool_use": post_tool_use,
    "post_tool_use_failure": post_tool_use_failure,
    "pre_tool_use": pre_tool_use,
    "session_start": session_start,
    "session_end": session_end,
    "pre_compact": pre_compact,
    "post_compact": post_compact,
    "subagent_start": subagent_start,
    "subagent_stop": subagent_stop,
    "stop_failure": stop_failure,
    "interrupt": interrupt,
    "permission_request": permission_request,
    "permission_result": permission_result,
    "notification": notification,
}


def main(argv: "list[str] | None" = None) -> None:
    """Dispatch to the right hook handler based on argv[1]."""
    if argv is None:
        argv = sys.argv

    cmd = argv[1] if len(argv) > 1 else ""
    handler = _COMMANDS.get(cmd)
    if handler is None:
        print(
            f"Usage: {argv[0]} <{'|'.join(_COMMANDS)}>",
            file=sys.stderr,
        )
        sys.exit(2)

    # Ensure YICENET_HOME has a sane default for hook subprocesses
    if not os.environ.get("YICENET_HOME"):
        os.environ.setdefault("YICENET_HOME", str(Path.home() / ".yicenet"))

    handler()


if __name__ == "__main__":
    main()
