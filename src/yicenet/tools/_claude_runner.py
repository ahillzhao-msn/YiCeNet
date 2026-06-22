"""YiCeNet Claude Code hook runner."""
import os, sys
# Prime stdout pipe immediately — Claude Code on Windows requires early data.
if len(sys.argv) > 1 and sys.argv[1] == "pre":
    os.write(1, b"[YiCeNet] ")
import json

# CRITICAL (Windows): Do NOT call sys.stdout.reconfigure() — it breaks
# Claude Code's pipe capture. Do NOT redirect sys.stdout to StringIO —
# delayed pipe data causes Claude Code to discard the output.
# Only reconfigure stderr for CJK label display.
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

event = os.environ.get("YICENET_HOOK_EVENT") or (sys.argv[1] if len(sys.argv) > 1 else "pre")
mode = os.environ.get("YICENET_MODE", "daemon")

# Read stdin once and pass explicitly.
_raw = ""
try:
    _raw = sys.stdin.read().strip()
except Exception:
    pass
_payload: dict = {}
try:
    _payload = json.loads(_raw) if _raw else {}
except Exception:
    pass

# ── Daemon mode (default): IPC to independent daemon, auto-spawn ────────────

if mode == "daemon":
    from yicenet.tools.ipc_hook import pre_message_send_ipc, stop_ipc, post_tool_ipc

    if event == "pre":
        if not pre_message_send_ipc(_payload):
            _degraded = {
                "yicenet": {
                    "session_id": "",
                    "turn_id": 0,
                    "label": "[YiCeNet daemon unavailable]",
                    "hexagram": "",
                    "action": "",
                    "env_confidence": 0.0,
                    "context_status": "thin",
                    "prescription": {},
                }
            }
            # Write directly to fd 1 as raw UTF-8 bytes
            os.write(1, json.dumps(_degraded, ensure_ascii=False).encode("utf-8"))
            sys.stderr.write("[YiCeNet] daemon unavailable\n")
            sys.stderr.flush()
    elif event == "post_tool":
        post_tool_ipc(_payload)
    elif event == "stop":
        stop_ipc(_payload)

    sys.exit(0)

# ── Subprocess mode: cold-start engine in-process ───────────────────────────

from yicenet.tools.claude_hook import ClaudeCodeAdapter
from yicenet.memory_bank import configure_memory_bank_for

_adapter = ClaudeCodeAdapter()
configure_memory_bank_for(_adapter)

if event == "pre":
    _adapter.pre_message_send(_payload)
elif event == "post_tool":
    from yicenet.hook_engine.collector.subprocess import SubprocessContextCollector
    sid = _adapter.session_id(_payload)
    tid = _adapter.turn_id(_payload)
    col = SubprocessContextCollector(sid, tid)
    col.sniff_tool(
        name=_payload.get("tool_name", ""),
        exit_code=_payload.get("exit_code", 0),
        duration_ms=_payload.get("duration_ms", 0),
        result_size_bytes=_payload.get("result_size", 0),
    )
elif event == "stop":
    from yicenet.hook_engine.collector.subprocess import SubprocessContextCollector
    sid = _adapter.session_id(_payload)
    tid = _adapter.turn_id(_payload)
    _adapter._ctx = SubprocessContextCollector(sid, tid)
    _adapter.stop(_payload)
