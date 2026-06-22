"""Thin IPC client for daemon mode.

Zero engine dependency — pure stdlib.  Connects to the YiCeNet daemon,
sends the hook payload, receives the prediction JSON, and outputs it
to stdout for Claude context injection.

Auto-spawn: if the daemon is not running, spawns one via daemon.launcher
and retries.  This makes daemon mode self-healing — no external process
manager required.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

_PORT_FILE = Path(tempfile.gettempdir()) / "yicenet-daemon.port"
_DEFAULT_PORT = 7788
_TIMEOUT = 10.0


def _get_port() -> int:
    try:
        if _PORT_FILE.exists():
            return int(_PORT_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        pass
    return int(os.environ.get("YICENET_DAEMON_PORT", _DEFAULT_PORT))


def _post(path: str, payload: dict, port: int = 0) -> "dict | None":
    if port == 0:
        port = _get_port()
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8", errors="replace")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _ensure_daemon() -> int:
    """Ensure daemon is running; spawn if needed. Returns port or 0."""
    from yicenet.daemon.launcher import ensure_daemon
    return ensure_daemon()


def pre_message_send_ipc(payload: dict) -> bool:
    """Send pre-turn payload to daemon; inject result into Claude context.

    Auto-spawns daemon if not running. Returns True if successful.
    """
    result = _post("/hook/pre", payload)

    if result is None:
        port = _ensure_daemon()
        if port == 0:
            return False
        result = _post("/hook/pre", payload, port=port)
        if result is None:
            return False

    label = result.get("yicenet", {}).get("label", "")
    if label:
        sys.stderr.write(label + "\n")
        sys.stderr.flush()
    # Write directly to fd 1 as raw UTF-8 — avoids TextIOWrapper encoding
    # issues and ensures data reaches Claude Code's pipe on Windows.
    os.write(1, json.dumps(result, ensure_ascii=False).encode("utf-8"))
    return True


def post_tool_ipc(payload: dict) -> bool:
    """Send post-tool payload to daemon for context collector."""
    result = _post("/hook/post_tool", payload)
    if result is None:
        port = _ensure_daemon()
        if port == 0:
            return False
        result = _post("/hook/post_tool", payload, port=port)
    return result is not None


def stop_ipc(payload: dict) -> bool:
    """Send stop payload to daemon for turn-complete bookkeeping."""
    result = _post("/hook/stop", payload)
    if result is None:
        # Don't spawn daemon just for stop — if it's gone, it's gone.
        return False
    return True
