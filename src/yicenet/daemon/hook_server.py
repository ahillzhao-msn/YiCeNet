"""YiCeNet daemon — independent HTTP server for hook IPC.

Runs as a standalone background process, NOT a thread inside MCP.
Spawned by the hook client (native `yicenet-hook` or daemon.launcher) on the
first request it cannot deliver; self-terminates after IDLE_TIMEOUT_S idle.

Endpoints:
  GET  /health                          — liveness probe ({"ok": true, "pid": ...})
  POST /hook/<event>?platform=<id>      — body: the agent's raw hook payload (JSON);
                                          response body: exact bytes for the hook's stdout.
                                          platform defaults to claude-code.

Platform semantics live in daemon.platforms; clients stay dumb.
Port: YICENET_DAEMON_PORT env var → config → DEFAULT_PORT (7788).
Port written to PORT_FILE; PID written to PID_FILE.
"""
from __future__ import annotations

import atexit
import json
import os
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

DEFAULT_PORT = 7788
IDLE_TIMEOUT_S = 1800  # 30 minutes
PORT_FILE = Path(tempfile.gettempdir()) / "yicenet-daemon.port"
PID_FILE = Path(tempfile.gettempdir()) / "yicenet-daemon.pid"

_last_request_time: float = time.monotonic()
_request_lock = threading.Lock()
# Threaded server so /health answers while a hook runs; hooks themselves run one at a time
# (adapters keep per-turn state and share one model).
_hook_lock = threading.Lock()


def _touch_activity() -> None:
    global _last_request_time
    with _request_lock:
        _last_request_time = time.monotonic()


class _HookHandler(BaseHTTPRequestHandler):

    def do_GET(self) -> None:
        if self.path == "/health":
            _touch_activity()
            self._send_json(200, {"ok": True, "pid": os.getpid()})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        _touch_activity()
        url = urlsplit(self.path)
        if not url.path.startswith("/hook/"):
            self._send_json(404, {"error": "not found"})
            return
        event = url.path[len("/hook/"):]
        platform = parse_qs(url.query).get("platform", [""])[0]

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length > 0 else b""
            text = body.decode("utf-8", errors="replace").strip()
            payload = json.loads(text) if text else {}
            if not isinstance(payload, dict):
                payload = {}
        except Exception:
            self._send_json(400, {"error": "bad request"})
            return

        from yicenet.daemon import platforms
        try:
            handler = platforms.handler_for(platform or platforms.DEFAULT_PLATFORM, event)
        except platforms.UnknownRoute as exc:
            self._send_json(404, {"error": str(exc)})
            return
        try:
            with _hook_lock:
                out = handler(payload)
        except Exception as exc:
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        self._send(200, out, "application/octet-stream")

    def _send_json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _send(self, code: int, data: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        pass


# ── Idle watchdog ────────────────────────────────────────────────────────────


def _idle_watchdog(timeout_s: int) -> None:
    """Background thread: exit the process after idle timeout."""
    while True:
        time.sleep(60)
        with _request_lock:
            idle = time.monotonic() - _last_request_time
        if idle >= timeout_s:
            sys.stderr.write(
                f"[YiCeNet daemon] idle {idle:.0f}s >= {timeout_s}s, shutting down\n"
            )
            _cleanup()
            os._exit(0)


# ── Lifecycle ────────────────────────────────────────────────────────────────


def _resolve_port() -> int:
    port = int(os.environ.get("YICENET_DAEMON_PORT", 0))
    if port:
        return port
    try:
        from yicenet.config import get_platform_config
        cfg = get_platform_config("claude-code")
        port = int(cfg.get("daemon", {}).get("port", 0))
    except Exception:
        pass
    return port or DEFAULT_PORT


def _read_port_file() -> int:
    try:
        return int(PORT_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        return 0


def _healthy(port: int) -> bool:
    if not port:
        return False
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as r:
            return r.status == 200
    except Exception:
        return False


def _cleanup() -> None:
    for f in (PORT_FILE, PID_FILE):
        try:
            f.unlink(missing_ok=True)
        except Exception:
            pass


def _opt_out_of_power_throttling() -> None:
    """Windows 11 runs windowless background processes (pythonw) under EcoQoS, which
    roughly quadruples model latency. Hooks sit on the user's critical path: opt out."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class _PowerThrottlingState(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                        ("StateMask", wintypes.ULONG)]

        PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
        ProcessPowerThrottling = 4
        state = _PowerThrottlingState(1, PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 0)
        kernel32 = ctypes.windll.kernel32
        kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), ProcessPowerThrottling,
                                       ctypes.byref(state), ctypes.sizeof(state))
    except Exception:
        pass


def run_standalone(port: int = 0, idle_timeout: int = IDLE_TIMEOUT_S) -> None:
    """Run the hook server as a standalone daemon process.

    Writes PID and port files, starts idle watchdog, serves until killed
    or idle timeout expires.
    """
    if port == 0:
        port = _resolve_port()

    # Several hook clients may spawn a daemon at the same moment: the first one to
    # bind wins, the rest see a healthy daemon and leave.
    if _healthy(_read_port_file()) or _healthy(port):
        sys.exit(0)
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), _HookHandler)
    except OSError:
        if _healthy(port):
            sys.exit(0)
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", 0), _HookHandler)
        except OSError:
            sys.exit(1)

    actual_port = srv.server_address[1]
    _opt_out_of_power_throttling()

    # Write identity files
    try:
        PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
        PORT_FILE.write_text(str(actual_port), encoding="utf-8")
    except Exception:
        pass

    atexit.register(_cleanup)

    # Start idle watchdog
    wd = threading.Thread(
        target=_idle_watchdog,
        args=(idle_timeout,),
        name="yicenet-idle-watchdog",
        daemon=True,
    )
    wd.start()

    sys.stderr.write(
        f"[YiCeNet daemon] pid={os.getpid()} port={actual_port} "
        f"idle_timeout={idle_timeout}s\n"
    )

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        _cleanup()


if __name__ == "__main__":
    # Reconfigure stdout/stderr to UTF-8 for CJK on Windows.
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            try:
                _s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    run_standalone()
