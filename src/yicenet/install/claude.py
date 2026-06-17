"""ClaudeCodeInstaller — hook-based Claude Code integration.

Registered hooks (settings.json):
  UserPromptSubmit → before_prediction() [extract prior feedback + prescribe]
  Stop             → on_turn_complete()  [store response_snippet for next turn]

PostToolUse is intentionally NOT registered: feedback signals require the
next user message to be present; before_prediction() handles this at
UserPromptSubmit time (1-turn delay design).

The hook script runs in whichever Python has yicenet installed — typically
the venv this installer is invoked from (sys.executable). Each invocation
is a short-lived subprocess; the FileBackend (JSONL WAL) bridges state
between UserPromptSubmit and Stop.
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

    def register_hooks(self) -> None:
        python = self._python()
        if not python:
            raise RuntimeError("No suitable Python found for Claude Code hooks")

        _HOOKS_DIR.mkdir(parents=True, exist_ok=True)

        # Write the hook runner script
        hook_script = _HOOKS_DIR / "yicenet_claude_hook.py"
        hook_script.write_text(self._hook_script_content(), encoding="utf-8")

        # Patch settings.json
        self._patch_settings(python, hook_script)

    def unregister(self) -> None:
        hook_script = _HOOKS_DIR / "yicenet_claude_hook.py"
        if hook_script.exists():
            hook_script.unlink()
        self._remove_hook_settings()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _python(self) -> str | None:
        return sys.executable

    def _hook_script_content(self) -> str:
        runner = Path(__file__).parent.parent / "tools" / "_claude_runner.py"
        return runner.read_text(encoding="utf-8")

    def _patch_settings(self, python: str, hook_script: Path) -> None:
        settings = {}
        if _SETTINGS.exists():
            try:
                settings = json.loads(_SETTINGS.read_text(encoding="utf-8"))
            except Exception:
                pass

        cmd = f'"{python}" "{hook_script}"'
        hooks = settings.setdefault("hooks", {})

        def _ensure_hook(event: str, env_val: str) -> None:
            entries = hooks.setdefault(event, [])
            new_entry = {
                "hooks": [{
                    "type": "command",
                    "command": cmd,
                    "env": {"YICENET_HOOK_EVENT": env_val},
                }]
            }
            # Avoid duplicates
            for e in entries:
                for h in e.get("hooks", []):
                    if "yicenet_claude_hook" in h.get("command", ""):
                        return
            entries.append(new_entry)

        _ensure_hook("UserPromptSubmit", "pre")
        _ensure_hook("Stop", "stop")
        # PostToolUse removed: feedback signals require the next user message
        # to be present; before_prediction() in UserPromptSubmit handles this.

        _SETTINGS.write_text(
            json.dumps(settings, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _remove_hook_settings(self) -> None:
        if not _SETTINGS.exists():
            return
        try:
            settings = json.loads(_SETTINGS.read_text(encoding="utf-8"))
            hooks = settings.get("hooks", {})
            for event in ["UserPromptSubmit", "PostToolUse", "Stop"]:
                if event in hooks:
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
                del settings["hooks"]
            _SETTINGS.write_text(
                json.dumps(settings, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
