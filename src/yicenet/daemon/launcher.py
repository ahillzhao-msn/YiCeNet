"""Daemon launcher — spawn and manage an independent YiCeNet daemon process.

The daemon is a first-class background process (not a thread inside MCP).
It runs the HTTP hook server, serves IPC requests from hooks, and
self-terminates after an idle timeout.

Lifecycle:
  1. First hook call discovers no daemon → ensure_daemon() spawns one
  2. Daemon writes PORT_FILE + PID_FILE, serves requests
  3. Idle watchdog: no requests for IDLE_TIMEOUT_S → clean exit
  4. Multiple Claude Code sessions share the same daemon (zero cold-start)
  5. Next session's first hook re-spawns if daemon has exited

Platform: Windows (pythonw.exe for headless), with POSIX fallback.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

PID_FILE = Path(tempfile.gettempdir()) / "yicenet-daemon.pid"
PORT_FILE = Path(tempfile.gettempdir()) / "yicenet-daemon.port"

_SPAWN_WAIT_S = 8.0
_SPAWN_POLL_S = 0.3
_HEALTH_TIMEOUT_S = 2.0


def _read_pid() -> int | None:
    try:
        if PID_FILE.exists():
            return int(PID_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        pass
    return None


def _read_port() -> int | None:
    try:
        if PORT_FILE.exists():
            return int(PORT_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        pass
    return None


def _is_process_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        SYNCHRONIZE = 0x00100000
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _health_check(port: int) -> bool:
    """Ping the daemon's health endpoint."""
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/health",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=_HEALTH_TIMEOUT_S) as resp:
            return resp.status == 200
    except Exception:
        return False


def is_daemon_running() -> int:
    """Check if the daemon is alive and responsive.

    Returns the port number if running, 0 otherwise.
    Cleans up stale PID/port files when the process is dead.
    """
    port = _read_port()
    pid = _read_pid()

    if port and _health_check(port):
        return port

    # Stale files — clean up
    if pid and not _is_process_alive(pid):
        _cleanup_files()
    elif port and not pid:
        _cleanup_files()

    return 0


def ensure_daemon() -> int:
    """Ensure the daemon is running. Spawns one if needed.

    Returns the port number, or 0 if spawn failed.
    """
    port = is_daemon_running()
    if port:
        return port
    return _spawn_daemon()


def _spawn_daemon() -> int:
    """Spawn a detached daemon process running hook_server standalone."""
    _cleanup_files()

    python = _daemon_python()
    if not python:
        return 0

    # Run hook_server as __main__ in a detached, headless process.
    daemon_module = "yicenet.daemon.hook_server"

    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = (
            subprocess.CREATE_NO_WINDOW
            | subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
        )

    log_dir = Path.home() / ".yicenet" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "daemon.log"

    try:
        with open(log_file, "a", encoding="utf-8") as log:
            subprocess.Popen(
                [python, "-m", daemon_module],
                stdout=log,
                stderr=log,
                stdin=subprocess.DEVNULL,
                creationflags=creation_flags,
                close_fds=True,
                start_new_session=(sys.platform != "win32"),
            )
    except Exception:
        return 0

    # Wait for the daemon to write its port file and become responsive.
    deadline = time.monotonic() + _SPAWN_WAIT_S
    while time.monotonic() < deadline:
        time.sleep(_SPAWN_POLL_S)
        port = _read_port()
        if port and _health_check(port):
            return port

    return 0


def stop_daemon() -> bool:
    """Stop the daemon process gracefully."""
    pid = _read_pid()
    if pid and _is_process_alive(pid):
        try:
            if sys.platform == "win32":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                PROCESS_TERMINATE = 0x0001
                handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
                if handle:
                    kernel32.TerminateProcess(handle, 0)
                    kernel32.CloseHandle(handle)
            else:
                os.kill(pid, signal.SIGTERM)
            for _ in range(20):
                time.sleep(0.2)
                if not _is_process_alive(pid):
                    break
        except Exception:
            pass
    _cleanup_files()
    return True


def _cleanup_files() -> None:
    for f in (PID_FILE, PORT_FILE):
        try:
            f.unlink(missing_ok=True)
        except Exception:
            pass


def _daemon_python() -> str | None:
    """Find the Python interpreter for the daemon.

    Prefers pythonw.exe (headless) on Windows; falls back to python.exe.
    Uses the same venv as the current process.
    """
    current = Path(sys.executable)

    if sys.platform == "win32":
        pythonw = current.parent / "pythonw.exe"
        if pythonw.exists():
            return str(pythonw)

    return str(current)
