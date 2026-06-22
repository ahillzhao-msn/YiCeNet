"""
YiCeNet Claude Code hook adapter.

ClaudeCodeAdapter implements only what is specific to Claude Code:
  - session_id: derived from Claude Code's session UUID or cwd+date hash
  - assistant_response: read from the Claude Code transcript JSONL file
  - process_model: "subprocess" for hook subprocess, "daemon" for daemon process

All shared prediction and hook lifecycle logic lives in HooksAdapter.

Entry points (called by _claude_runner.py):
  adapter.pre_message_send([payload])  — UserPromptSubmit
  adapter.stop([payload])              — Stop
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys

from yicenet.tools.hooks_adapter import HooksAdapter

# Reconfigure to UTF-8 so CJK characters print on Windows.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


class ClaudeCodeAdapter(HooksAdapter):
    """Platform adapter for Claude Code hooks.

    process_model defaults to "subprocess" for the installed hook runner.
    Pass process_model="daemon" when constructing an instance inside the
    daemon process to suppress MemoryBank session flushing between calls.
    """

    _platform_id = "claude-code"

    def __init__(self, process_model: str = "subprocess") -> None:
        self._process_model = process_model

    def create_collector(self, payload: dict):
        if self._process_model == "daemon":
            from yicenet.hook_engine.collector.daemon import DaemonContextCollector
            return DaemonContextCollector()
        from yicenet.hook_engine.collector.subprocess import SubprocessContextCollector
        return SubprocessContextCollector(
            session_id=self.session_id(payload),
            turn_id=self.turn_id(payload),
        )

    @property
    def platform_id(self) -> str:
        return self._platform_id

    @property
    def process_model(self) -> str:
        return self._process_model

    def session_id(self, payload: dict) -> str:
        cc_id = payload.get("session_id", "")
        if cc_id:
            return cc_id.replace("-", "")[:12]
        cwd = payload.get("cwd", os.getcwd())
        date = datetime.datetime.now().strftime("%Y%m%d")
        return hashlib.sha256(f"{cwd}{date}".encode()).hexdigest()[:12]

    def assistant_response(self, payload: dict) -> str:
        """Read last assistant message from the Claude Code transcript file."""
        return _read_last_assistant(payload)


# ── Transcript helpers ────────────────────────────────────────────────────────

def _read_last_assistant(payload: dict) -> str:
    from pathlib import Path

    tp = payload.get("transcript_path", "")
    if tp and Path(tp).exists():
        return _last_asst_from_file(Path(tp))

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
