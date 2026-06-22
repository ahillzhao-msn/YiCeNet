# Phase 2 — CC PostToolUse Hook + IPC Completion

> Goal: Close the CC signal gap. Register PostToolUse hook, add `/hook/post_tool`
> endpoint, switch CC default to daemon/IPC mode, complete the 27-dim signal flow.
>
> Verified: IPC `/hook/pre` and `/hook/stop` work correctly. Adapter is singleton
> in daemon — same `ctx` survives across IPC calls. See `verified-ipc-2026-06-22.md`.

---

## 1. Current State (Verified)

```
Hermes:  ✅ full 27-dim (plugin hooks, daemon)
CC:      ⚠️ pre + stop registered, PostToolUse missing
          ⚠️ yicenet_claude_hook.py default mode = "auto" (tries IPC, falls back to subprocess)
          ⚠️ ipc_hook._TIMEOUT = 2.0s — too short for cold start (~1.3s)
          ⚠️ hook_server missing /hook/post_tool endpoint
Pure MCP: ✅ tools implemented, not applied
```

### IPC Test Results (2026-06-22)

| Test | Result |
|------|--------|
| `POST /hook/pre` → hexagram + label | ✅ ䷪ 夬 |
| `POST /hook/stop` → `{"ok": true}` | ✅ |
| Same daemon adapter across pre→stop | ✅ ctx singleton, different hexagram per turn |
| Cold start within IPC timeout | ⚠️ 1.3s — ipc_hook._TIMEOUT(2.0) barely works, needs bump to 10s |
| `POST /hook/post_tool` | ❌ endpoint doesn't exist |

---

## 2. Changes Required

### 2.1 `daemon/hook_server.py` — Add `/hook/post_tool`

```python
# In do_POST, add:
elif self.path == "/hook/post_tool":
    self._handle_post_tool(payload)

# New handler method:
def _handle_post_tool(self, payload: dict) -> None:
    try:
        adapter = _hook_adapter()  # singleton — same ctx as pre
        if adapter.ctx is not None:
            adapter.ctx.sniff_tool(
                name=payload.get("tool_name", ""),
                exit_code=payload.get("exit_code", 0),
                duration_ms=payload.get("duration_ms", 0),
                result_size_bytes=payload.get("result_size", 0),
            )
        self._send(200, {"ok": True})
    except Exception as exc:
        self._send(500, {"error": str(exc)})
```

### 2.2 `tools/ipc_hook.py` — Increase timeout, add `post_tool_ipc()`

```python
# Change:
_TIMEOUT = 2.0   →   _TIMEOUT = 10.0  # accommodate cold start (~1.3s)

# Add:
def post_tool_ipc(payload: dict) -> bool:
    """Send post-tool payload to daemon for context collector."""
    return _post("/hook/post_tool", payload) is not None
```

### 2.3 `tools/yicenet_claude_hook.py` — Wire post_tool handler

The installed hook script is at `~/.claude/hooks/yicenet_claude_hook.py`.
Its source of truth is this file (wherever it lives in the CC tools).
Currently the `post_tool` event is a no-op:

```python
elif event == "post_tool":
    pass  # ← change this
```

Replace with:

```python
elif event == "post_tool":
    from yicenet.tools.ipc_hook import post_tool_ipc
    if not post_tool_ipc(_payload):
        # IPC unreachable → subprocess fallback
        from yicenet.hook_engine.collector.subprocess import SubprocessContextCollector
        sid = _payload.get("session_id", "")
        tid = int(_payload.get("turn_id", 0))
        col = SubprocessContextCollector(sid, tid)
        col.sniff_tool(
            name=_payload.get("tool_name", ""),
            exit_code=_payload.get("exit_code", 0),
            duration_ms=_payload.get("duration_ms", 0),
            result_size_bytes=_payload.get("result_size", 0),
        )
```

### 2.4 CC `settings.json` — Register PostToolUse hook

Current settings only have UserPromptSubmit and Stop. Add PostToolUse:

```json
{
  "hooks": {
    "UserPromptSubmit": [{ "hooks": [{ "type": "command", "command": "<python> <script> pre" }] }],
    "PostToolUse": [{ "hooks": [{ "type": "command", "command": "<python> <script> post_tool" }] }],
    "Stop": [{ "hooks": [{ "type": "command", "command": "<python> <script> stop" }] }]
  }
}
```

### 2.5 Default to daemon mode

In `yicenet_claude_hook.py`, change default mode from `"auto"` to daemon-first:

```python
# Before:
mode = os.environ.get("YICENET_MODE", "auto")

# After:  
mode = os.environ.get("YICENET_MODE", "daemon")
```

The daemon path (IPC via hook_server) is preferred. Subprocess/file fallback only
when the daemon is unreachable.

### 2.6 `yicenet_claude_hook.py` — Subprocess stop reads file

The stop handler in subprocess mode currently doesn't build context_vector because
`ctx` is from a fresh process (new_turn() never called). Fix:

```python
elif event == "stop":
    from yicenet.tools.ipc_hook import stop_ipc
    if not stop_ipc(_payload):
        # IPC unreachable → subprocess: build context_vector from file
        from yicenet.hook_engine.collector.subprocess import SubprocessContextCollector
        # ... (see full implementation below)
```

---

## 3. Full Event Dispatch Table

```
event      | IPC path               | Purpose                  | Required?
-----------|------------------------|--------------------------|----------
pre        | /hook/pre              | sniff_user + hexagram    | YES
post_tool  | /hook/post_tool        | sniff_tool(name,exit,dur)| YES — NEW
stop       | /hook/stop             | sniff_response + build   | YES
```

Each event tries IPC first (fast, daemon-mode, same ctx). Falls back to
subprocess/file-based SubprocessContextCollector.

---

## 4. File Changes Summary

| File | Change |
|------|--------|
| `src/yicenet/daemon/hook_server.py` | Add `/hook/post_tool` endpoint + `_handle_post_tool()` |
| `src/yicenet/tools/ipc_hook.py` | `_TIMEOUT: 2.0 → 10.0`, add `post_tool_ipc()` |
| `src/yicenet/tools/yicenet_claude_hook.py` | Wire post_tool, fix stop ctx, default mode=daemon |
| `~/.claude/settings.json` | Register PostToolUse hook |

---

## 5. Acceptance Criteria

1. `POST /hook/post_tool` returns `{"ok": true}` and feeds data into daemon ctx
2. `ipc_hook.post_tool_ipc()` returns True when daemon is running
3. CC settings.json has PostToolUse hook registered
4. `yicenet_claude_hook.py` default mode = daemon (YICENET_MODE not set → daemon)
5. Cold-start IPC call succeeds (10s timeout is sufficient)
6. Subprocess fallback still works when daemon is down
7. All existing tests pass: `python -m pytest tests/ -x`
8. Verified: context_vector written to MemoryBank after stop (via orchestrator)

---

## 6. Not In Scope

| Item | Reason |
|------|--------|
| Pure MCP signal integration | Phase 3 (ExplicitContextCollector) |
| site-packages deployment | Separate `pip install -e . --force-reinstall` step |
| ClaudeCodeInstaller script update | CC should design the installer update |
| Hermes plugin re-deployment | Not part of CC phase — separate `cp _hermes_stub.py` |
