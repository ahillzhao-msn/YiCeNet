"""YiCeNet hook IPC client — Python twin of the native `yicenet-hook` binary.

Forwards the agent's raw hook payload to the daemon
(`POST /hook/<event>?platform=<platform>`) and returns the daemon's reply,
which is the exact stdout for the hook.  No platform logic here: that lives in
yicenet.daemon.platforms.  Pure stdlib so the hook process stays cheap.

Auto-spawn: if the daemon is not running, spawns one via daemon.launcher
and retries once.
"""
from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import quote

_PORT_FILE = Path(tempfile.gettempdir()) / "yicenet-daemon.port"
_DEFAULT_PORT = 7788
_TIMEOUT = 30.0  # the first request after a spawn loads the model


def _get_port() -> int:
    env = os.environ.get("YICENET_DAEMON_PORT")
    if env:
        return int(env)
    try:
        return int(_PORT_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        return _DEFAULT_PORT


def _post(platform: str, event: str, body: bytes, port: int = 0) -> "bytes | None":
    """One request. None when the daemon is unreachable or rejects the request."""
    port = port or _get_port()
    url = f"http://127.0.0.1:{port}/hook/{quote(event)}?platform={quote(platform)}"
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return resp.read()
    except Exception:
        return None


def _ensure_daemon() -> int:
    """Ensure daemon is running; spawn if needed. Returns port or 0."""
    from yicenet.daemon.launcher import ensure_daemon
    return ensure_daemon()


def forward(platform: str, event: str, body: bytes) -> "bytes | None":
    """Deliver one hook event, spawning the daemon once if it is not up."""
    out = _post(platform, event, body)
    if out is None:
        port = _ensure_daemon()
        if port:
            out = _post(platform, event, body, port=port)
    return out
