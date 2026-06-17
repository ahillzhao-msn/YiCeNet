"""HermesInstaller — registers YiCeNet as a Hermes plugin."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .base import PlatformInstaller


def _hermes_home() -> Path:
    """Derive Hermes home from sys.executable venv path.

    Expects: <HERMES_HOME>/hermes-agent/venv/{Scripts|bin}/python
    Falls back to HERMES_HOME env var if the structure doesn't match.
    """
    exe = Path(sys.executable).resolve()
    # exe.parents: [Scripts|bin, venv, hermes-agent, HERMES_HOME, ...]
    if len(exe.parents) >= 4 and exe.parents[2].name == "hermes-agent":
        return exe.parents[3]
    if os.environ.get("HERMES_HOME"):
        return Path(os.environ["HERMES_HOME"])
    return Path.home() / ".hermes"


class HermesInstaller(PlatformInstaller):

    def detect(self) -> bool:
        return shutil.which("hermes") is not None

    def install_package(self, editable_path: Path = None) -> bool:
        pkg = str(editable_path) if editable_path else "yicenet"
        flag = ["-e"] if editable_path else []
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", *flag, pkg],
            capture_output=True,
        )
        return r.returncode == 0

    def register_hooks(self) -> None:
        plugin_dir = _hermes_home() / "plugins" / "yicenet-hooks"
        plugin_dir.mkdir(parents=True, exist_ok=True)
        (plugin_dir / "plugin.yaml").write_text(
            "name: yicenet-hooks\n"
            "version: '1'\n"
            "hooks: [pre_llm_call, post_tool_call, post_llm_call]\n",
            encoding="utf-8",
        )
        self._write_init_py(plugin_dir)

    def unregister(self) -> None:
        plugin_dir = _hermes_home() / "plugins" / "yicenet-hooks"
        if plugin_dir.exists():
            shutil.rmtree(plugin_dir)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _write_init_py(self, plugin_dir: Path) -> None:
        stub = Path(__file__).parent.parent / "tools" / "_hermes_stub.py"
        (plugin_dir / "__init__.py").write_text(
            stub.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
