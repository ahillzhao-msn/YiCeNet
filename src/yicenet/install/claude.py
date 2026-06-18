"""ClaudeCodeInstaller — all three YiCeNet integration modes for Claude Code.

Mode 1  register_hooks()   — subprocess hook (cold-start; works without MCP)
Mode 2  register_mcp()     — MCP server only (pure tool-call; no auto-injection)
Mode 3  register_hybrid()  — MCP daemon + thin IPC hooks (auto-injection, warm engine)

Registered hooks (settings.json):
  UserPromptSubmit → predict_for_turn_payload [extract prior feedback + prescribe]
  Stop             → on_turn_complete          [store response_snippet for next turn]

PostToolUse is intentionally NOT registered: feedback signals require the next
user message to be present; before_prediction() handles this at UserPromptSubmit
time (1-turn delay design).

In Mode 1, each hook invocation is a short-lived subprocess; FileBackend
(JSONL WAL) bridges state between UserPromptSubmit and Stop.

In Mode 3, the MCP server is the long-lived daemon; hook subprocesses are
thin IPC clients that reach the daemon over HTTP and exit in <100ms.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from .base import PlatformInstaller

_CLAUDE_DIR = Path.home() / ".claude"
_HOOKS_DIR = _CLAUDE_DIR / "hooks"
_SETTINGS = _CLAUDE_DIR / "settings.json"


class ClaudeCodeInstaller(PlatformInstaller):

    def detect(self) -> bool:
        return shutil.which("claude") is not None or _CLAUDE_DIR.exists()

    def install_package(self, editable_path: Path = None) -> bool:
        python = self._python()
        if not python:
            return False
        pkg = str(editable_path) if editable_path else "yicenet"
        flag = ["-e"] if editable_path else []
        r = subprocess.run(
            [python, "-m", "pip", "install", "--quiet", *flag, pkg],
            capture_output=True,
        )
        return r.returncode == 0

    # ── Mode 1: subprocess hooks ──────────────────────────────────────────────

    def register_hooks(self) -> None:
        """Mode 1: register subprocess hook (cold-start per message)."""
        python = self._python()
        if not python:
            raise RuntimeError("No suitable Python found for Claude Code hooks")

        _HOOKS_DIR.mkdir(parents=True, exist_ok=True)
        hook_script = self._write_hook_script()
        self._patch_hook_settings(python, hook_script)

    def unregister(self) -> None:
        """Remove hooks and MCP server registration added by any register_*()."""
        hook_script = _HOOKS_DIR / "yicenet_claude_hook.py"
        if hook_script.exists():
            hook_script.unlink()
        self._remove_hook_settings()
        self._remove_mcp_settings()

    # ── Mode 2: pure MCP ─────────────────────────────────────────────────────

    def register_mcp(self) -> None:
        """Mode 2: register MCP server (explicit tool calls; no auto-injection)."""
        serve = self._yicenet_serve()
        if not serve:
            raise RuntimeError(
                "yicenet-serve not found. Is yicenet installed in this venv?"
            )
        self._patch_mcp_settings(serve)

    def unregister_mcp(self) -> None:
        """Remove MCP server registration."""
        self._remove_mcp_settings()

    # ── Mode 3: hybrid (MCP daemon + thin IPC hooks) ─────────────────────────

    def register_hybrid(self) -> None:
        """Mode 3: register MCP daemon AND hooks; runner auto-detects IPC vs subprocess."""
        serve = self._yicenet_serve()
        if not serve:
            raise RuntimeError(
                "yicenet-serve not found. Is yicenet installed in this venv?"
            )
        python = self._python()
        if not python:
            raise RuntimeError("No suitable Python found for Claude Code hooks")

        # 1. Register MCP server as daemon
        self._patch_mcp_settings(serve)

        # 2. Register hooks — no mode env var; runner auto-detects via port file
        _HOOKS_DIR.mkdir(parents=True, exist_ok=True)
        hook_script = self._write_hook_script()
        self._patch_hook_settings(python, hook_script)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _python(self) -> str | None:
        return sys.executable

    def _yicenet_serve(self) -> str | None:
        """Locate the yicenet-serve executable in this venv."""
        serve = shutil.which("yicenet-serve")
        if serve:
            return serve
        python = self._python()
        if python:
            scripts = Path(python).parent
            for name in ("yicenet-serve.exe", "yicenet-serve"):
                candidate = scripts / name
                if candidate.exists():
                    return str(candidate)
        return None

    def _write_hook_script(self) -> Path:
        hook_script = _HOOKS_DIR / "yicenet_claude_hook.py"
        hook_script.write_text(self._hook_script_content(), encoding="utf-8")
        return hook_script

    def _hook_script_content(self) -> str:
        runner = Path(__file__).parent.parent / "tools" / "_claude_runner.py"
        return runner.read_text(encoding="utf-8")

    def _load_settings(self) -> dict:
        if _SETTINGS.exists():
            try:
                return json.loads(_SETTINGS.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_settings(self, settings: dict) -> None:
        _SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        _SETTINGS.write_text(
            json.dumps(settings, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _patch_hook_settings(self, python: str, hook_script: Path) -> None:
        settings = self._load_settings()
        hooks = settings.setdefault("hooks", {})

        def _ensure_hook(event: str, event_val: str) -> None:
            # Pass event as argv so it works even if Claude Code ignores env.
            cmd = f'"{python}" "{hook_script}" {event_val}'
            entries = hooks.setdefault(event, [])
            new_entry = {"hooks": [{"type": "command", "command": cmd}]}
            for e in entries:
                for h in e.get("hooks", []):
                    if "yicenet_claude_hook" in h.get("command", ""):
                        h["command"] = cmd
                        h.pop("env", None)  # strip legacy env
                        self._save_settings(settings)
                        return
            entries.append(new_entry)

        _ensure_hook("UserPromptSubmit", "pre")
        _ensure_hook("Stop", "stop")
        self._save_settings(settings)

    def _patch_mcp_settings(self, serve_path: str) -> None:
        settings = self._load_settings()
        mcp_servers = settings.setdefault("mcpServers", {})
        if "yicenet" not in mcp_servers:
            mcp_servers["yicenet"] = {"command": serve_path, "env": {}}
        self._save_settings(settings)

    def _remove_hook_settings(self) -> None:
        if not _SETTINGS.exists():
            return
        try:
            settings = self._load_settings()
            hooks = settings.get("hooks", {})
            for event in list(hooks.keys()):
                hooks[event] = [
                    e for e in hooks[event]
                    if not any(
                        "yicenet_claude_hook" in h.get("command", "")
                        for h in e.get("hooks", [])
                    )
                ]
                if not hooks[event]:
                    del hooks[event]
            if not hooks:
                settings.pop("hooks", None)
            self._save_settings(settings)
        except Exception:
            pass

    def _remove_mcp_settings(self) -> None:
        if not _SETTINGS.exists():
            return
        try:
            settings = self._load_settings()
            mcp_servers = settings.get("mcpServers", {})
            mcp_servers.pop("yicenet", None)
            if not mcp_servers:
                settings.pop("mcpServers", None)
            self._save_settings(settings)
        except Exception:
            pass
