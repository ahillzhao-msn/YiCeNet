"""Tests for ClaudeCodeInstaller and HermesInstaller."""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from yicenet.install.claude import ClaudeCodeInstaller
from yicenet.install.hermes import HermesInstaller, _hermes_home


# ── ClaudeCodeInstaller ───────────────────────────────────────────────────────

class TestClaudeCodeInstaller:

    def test_python_returns_sys_executable(self):
        installer = ClaudeCodeInstaller()
        assert installer._python() == sys.executable

    def test_register_hooks_writes_script_and_settings(self, tmp_path):
        claude_dir = tmp_path / ".claude"
        hooks_dir = claude_dir / "hooks"
        settings_file = claude_dir / "settings.json"

        with patch("yicenet.install.claude._CLAUDE_DIR", claude_dir), \
             patch("yicenet.install.claude._HOOKS_DIR", hooks_dir), \
             patch("yicenet.install.claude._SETTINGS", settings_file):
            ClaudeCodeInstaller().register_hooks()

        assert (hooks_dir / "yicenet_claude_hook.py").exists()
        import json
        settings = json.loads(settings_file.read_text())
        hooks = settings.get("hooks", {})
        assert "UserPromptSubmit" in hooks
        assert "Stop" in hooks

    def test_register_hooks_no_duplicates(self, tmp_path):
        claude_dir = tmp_path / ".claude"
        hooks_dir = claude_dir / "hooks"
        settings_file = claude_dir / "settings.json"

        with patch("yicenet.install.claude._CLAUDE_DIR", claude_dir), \
             patch("yicenet.install.claude._HOOKS_DIR", hooks_dir), \
             patch("yicenet.install.claude._SETTINGS", settings_file):
            installer = ClaudeCodeInstaller()
            installer.register_hooks()
            installer.register_hooks()  # second call should not duplicate

        import json
        settings = json.loads(settings_file.read_text())
        assert len(settings["hooks"]["UserPromptSubmit"]) == 1
        assert len(settings["hooks"]["Stop"]) == 1

    def test_unregister_removes_hook_script(self, tmp_path):
        claude_dir = tmp_path / ".claude"
        hooks_dir = claude_dir / "hooks"
        settings_file = claude_dir / "settings.json"

        with patch("yicenet.install.claude._CLAUDE_DIR", claude_dir), \
             patch("yicenet.install.claude._HOOKS_DIR", hooks_dir), \
             patch("yicenet.install.claude._SETTINGS", settings_file):
            installer = ClaudeCodeInstaller()
            installer.register_hooks()
            assert (hooks_dir / "yicenet_claude_hook.py").exists()
            installer.unregister()

        assert not (hooks_dir / "yicenet_claude_hook.py").exists()

    def test_detect_true_when_claude_dir_exists(self, tmp_path):
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        with patch("yicenet.install.claude._CLAUDE_DIR", claude_dir):
            assert ClaudeCodeInstaller().detect() is True

    def test_detect_false_when_nothing_present(self, tmp_path):
        with patch("yicenet.install.claude._CLAUDE_DIR", tmp_path / ".claude"), \
             patch("shutil.which", return_value=None):
            assert ClaudeCodeInstaller().detect() is False


# ── HermesInstaller / _hermes_home ───────────────────────────────────────────

class TestHermesHome:

    def test_derives_from_sys_executable_windows_structure(self, tmp_path):
        fake_hermes = tmp_path / "hermes"
        fake_exe = fake_hermes / "hermes-agent" / "venv" / "Scripts" / "python.exe"
        fake_exe.parent.mkdir(parents=True)
        fake_exe.touch()

        with patch("sys.executable", str(fake_exe)):
            result = _hermes_home()

        assert result == fake_hermes.resolve()

    def test_derives_from_sys_executable_linux_structure(self, tmp_path):
        fake_hermes = tmp_path / ".hermes"
        fake_exe = fake_hermes / "hermes-agent" / "venv" / "bin" / "python3"
        fake_exe.parent.mkdir(parents=True)
        fake_exe.touch()

        with patch("sys.executable", str(fake_exe)):
            result = _hermes_home()

        assert result == fake_hermes.resolve()

    def test_falls_back_to_hermes_home_env(self, tmp_path):
        fake_exe = tmp_path / "some" / "other" / "python.exe"
        fake_exe.parent.mkdir(parents=True)
        fake_exe.touch()
        env_home = str(tmp_path / "custom-hermes")

        with patch("sys.executable", str(fake_exe)), \
             patch.dict("os.environ", {"HERMES_HOME": env_home}):
            result = _hermes_home()

        assert result == Path(env_home)

    def test_falls_back_to_default_when_no_env(self, tmp_path):
        fake_exe = tmp_path / "some" / "other" / "python.exe"
        fake_exe.parent.mkdir(parents=True)
        fake_exe.touch()

        env = {k: v for k, v in __import__("os").environ.items()
               if k not in ("HERMES_HOME", "LOCALAPPDATA")}
        with patch("sys.executable", str(fake_exe)), \
             patch.dict("os.environ", env, clear=True):
            result = _hermes_home()

        assert result == Path.home() / ".hermes"


class TestHermesInstaller:

    def test_register_hooks_writes_to_derived_hermes_home(self, tmp_path):
        fake_hermes = tmp_path / "hermes"
        fake_exe = fake_hermes / "hermes-agent" / "venv" / "Scripts" / "python.exe"
        fake_exe.parent.mkdir(parents=True)
        fake_exe.touch()

        stub = Path(__file__).parent.parent / "src" / "yicenet" / "tools" / "_hermes_stub.py"

        with patch("sys.executable", str(fake_exe)):
            HermesInstaller().register_hooks()

        plugin_dir = fake_hermes / "plugins" / "yicenet-hooks"
        assert (plugin_dir / "plugin.yaml").exists()
        assert (plugin_dir / "__init__.py").exists()

    def test_unregister_removes_plugin_dir(self, tmp_path):
        fake_hermes = tmp_path / "hermes"
        fake_exe = fake_hermes / "hermes-agent" / "venv" / "Scripts" / "python.exe"
        fake_exe.parent.mkdir(parents=True)
        fake_exe.touch()

        with patch("sys.executable", str(fake_exe)):
            installer = HermesInstaller()
            installer.register_hooks()
            plugin_dir = fake_hermes / "plugins" / "yicenet-hooks"
            assert plugin_dir.exists()
            installer.unregister()
            assert not plugin_dir.exists()
