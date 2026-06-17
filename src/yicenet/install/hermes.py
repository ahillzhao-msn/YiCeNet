"""HermesInstaller — registers YiCeNet as a Hermes plugin."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .base import PlatformInstaller

_HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
_PLUGIN_DIR = _HERMES_HOME / "plugins" / "yicenet-hooks"


class HermesInstaller(PlatformInstaller):

    def detect(self) -> bool:
        return shutil.which("hermes") is not None

    def install_package(self, editable_path: Path = None) -> bool:
        python = self._hermes_python()
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
        _PLUGIN_DIR.mkdir(parents=True, exist_ok=True)
        (_PLUGIN_DIR / "plugin.yaml").write_text(
            "name: yicenet-hooks\n"
            "version: '1'\n"
            "hooks: [pre_llm_call, post_tool_call, post_llm_call]\n",
            encoding="utf-8",
        )
        self._write_init_py()

    def unregister(self) -> None:
        if _PLUGIN_DIR.exists():
            shutil.rmtree(_PLUGIN_DIR)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _hermes_python(self) -> str | None:
        _local_app = Path(os.environ.get("LOCALAPPDATA", ""))
        for candidate in [
            _HERMES_HOME / "hermes-agent" / "venv" / "Scripts" / "python.exe",
            _HERMES_HOME / "hermes-agent" / "venv" / "bin" / "python3",
            _HERMES_HOME / ".venv" / "bin" / "python3",
            _local_app / "hermes" / "hermes-agent" / "venv" / "Scripts" / "python.exe",
            _local_app / "hermes" / "hermes-agent" / "venv" / "bin" / "python3",
        ]:
            if candidate.exists():
                return str(candidate)
        return None

    def _write_init_py(self) -> None:
        stub = Path(__file__).parent.parent / "tools" / "_hermes_stub.py"
        (_PLUGIN_DIR / "__init__.py").write_text(
            stub.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
