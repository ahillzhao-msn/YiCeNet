"""Thin IPC client for hybrid mode (Mode 3).

Zero engine dependency — pure stdlib.  Connects to the YiCeNet daemon HTTP
server, sends the hook payload, receives the prediction JSON, and outputs it
to stdout for Claude context injection.

Falls back gracefully to False if the daemon is unreachable so the runner can
decide how to proceed (fall back to subprocess mode or exit cleanly).
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
_TIMEOUT = 10.0  # seconds — accommodate cold start (~1.3s observed)


def _daemon_available() -> bool:
    """Fast check: port file or explicit env var signals a daemon is expected."""
    if os.environ.get("YICENET_DAEMON_PORT"):
        return True
    try:
        return _PORT_FILE.exists()
    except Exception:
        return False


def _get_port() -> int:
    try:
        if _PORT_FILE.exists():
            return int(_PORT_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        pass
    return int(os.environ.get("YICENET_DAEMON_PORT", _DEFAULT_PORT))


def _post(path: str, payload: dict) -> "dict | None":
    """Send a JSON POST to the daemon. Returns parsed response or None."""
    port = _get_port()
    url = f"http://127.0.0.1:{port}{path}"
    # errors="replace": lone surrogates in transcript text won't crash encode
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


def pre_message_send_ipc(payload: dict) -> bool:
    """Send pre-turn payload to daemon; inject result into Claude context.

    Returns True if the daemon responded, False if unreachable.
    Writes the hexagram label to stderr (terminal-visible) and the full
    yicenet JSON to stdout (injected into Claude's context window).
    """
    if not _daemon_available():
        return False
    result = _post("/hook/pre", payload)
    if result is None:
        return False
    label = result.get("yicenet", {}).get("label", "")
    if label:
        sys.stderr.write(label + "\n")
        sys.stderr.flush()
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return True


def post_tool_ipc(payload: dict) -> bool:
    """Send post-tool payload to daemon for context collector."""
    if not _daemon_available():
        return False
    return _post("/hook/post_tool", payload) is not None


def stop_ipc(payload: dict) -> bool:
    """Send stop payload to daemon for turn-complete bookkeeping.

    Returns True if daemon acknowledged, False if unreachable.
    """
    if not _daemon_available():
        return False
    result = _post("/hook/stop", payload)
    return result is not None
