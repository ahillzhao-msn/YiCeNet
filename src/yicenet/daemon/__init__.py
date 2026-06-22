"""YiCeNet daemon — independent background process for hook IPC.

The daemon is a first-class process, not a thread inside MCP or any host.
Spawned on demand by the first hook call, shared across sessions,
self-terminates after idle timeout.
"""
from .hook_server import DEFAULT_PORT, PORT_FILE, PID_FILE, run_standalone
from .launcher import ensure_daemon, is_daemon_running, stop_daemon

__all__ = [
    "DEFAULT_PORT",
    "PORT_FILE",
    "PID_FILE",
    "run_standalone",
    "ensure_daemon",
    "is_daemon_running",
    "stop_daemon",
]
