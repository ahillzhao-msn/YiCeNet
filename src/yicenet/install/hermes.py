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
        for candidate in [
            _HERMES_HOME / "hermes-agent" / "venv" / "Scripts" / "python.exe",
            _HERMES_HOME / "hermes-agent" / "venv" / "bin" / "python3",
            _HERMES_HOME / ".venv" / "bin" / "python3",
        ]:
            if candidate.exists():
                return str(candidate)
        return None

    def _write_init_py(self) -> None:
        (_PLUGIN_DIR / "__init__.py").write_text(
            '"""YiCeNet Hermes plugin — lifecycle hooks."""\n'
            "import os\n"
            "from pathlib import Path\n\n"
            "_cfg_path = Path.home() / '.yicenet' / 'config.yaml'\n"
            "if _cfg_path.exists():\n"
            "    try:\n"
            "        import yaml\n"
            "        _rt = yaml.safe_load(_cfg_path.read_text()) or {}\n"
            "        _rt = _rt.get('runtime', {})\n"
            "        if _rt.get('transformers_offline', True):\n"
            "            os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')\n"
            "        if _rt.get('hf_hub_offline', True):\n"
            "            os.environ.setdefault('HF_HUB_OFFLINE', '1')\n"
            "        if _rt.get('tqdm_disable', True):\n"
            "            os.environ.setdefault('TQDM_DISABLE', '1')\n"
            "    except Exception:\n"
            "        pass\n\n"
            "def pre_llm_call(context):\n"
            "    from yicenet.tools.hermes_hook import pre_llm_call as _fn\n"
            "    return _fn(context)\n\n"
            "def post_tool_call(context):\n"
            "    from yicenet.tools.hermes_hook import post_tool_call as _fn\n"
            "    return _fn(context)\n\n"
            "def post_llm_call(context):\n"
            "    from yicenet.tools.hermes_hook import post_llm_call as _fn\n"
            "    return _fn(context)\n",
            encoding="utf-8",
        )
