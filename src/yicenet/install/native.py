"""Native hook client (`yicenet-hook`) — shared by all platform installers.

One binary serves every agent CLI: `yicenet-hook <platform> <event>` forwards the
raw hook payload to the daemon, which owns all platform semantics
(yicenet.daemon.platforms).  Built from native/yicenet-hook by
scripts/build-hook.ps1 / build-hook.sh into ~/.yicenet/bin.

The client spawns the daemon itself when it is not running, so it needs to know
which interpreter has yicenet installed: ~/.yicenet/daemon-python.
"""
from __future__ import annotations

import sys
from pathlib import Path

BIN_DIR = Path.home() / ".yicenet" / "bin"
DAEMON_PYTHON_FILE = Path.home() / ".yicenet" / "daemon-python"
_EXE = "yicenet-hook.exe" if sys.platform == "win32" else "yicenet-hook"


def hook_binary() -> "Path | None":
    """The installed native client, or None (installers then fall back to Python hooks)."""
    exe = BIN_DIR / _EXE
    return exe if exe.is_file() else None


def daemon_python() -> str:
    """This venv's own interpreter (pythonw.exe on Windows when present).

    Not launcher._daemon_python(): that routes past the venv's trampoline to the
    base interpreter, which only finds yicenet when __PYVENV_LAUNCHER__ is set —
    the Python launcher sets it, the native client does not.
    """
    exe = Path(sys.executable)
    if sys.platform == "win32":
        pythonw = exe.with_name("pythonw.exe")
        if pythonw.is_file():
            return str(pythonw)
    return str(exe)


def write_daemon_python() -> Path:
    """Record the interpreter the native client uses to spawn the daemon (this venv's)."""
    DAEMON_PYTHON_FILE.parent.mkdir(parents=True, exist_ok=True)
    DAEMON_PYTHON_FILE.write_text(daemon_python() + "\n", encoding="utf-8")
    return DAEMON_PYTHON_FILE


def is_native_command(command: str) -> bool:
    return "yicenet-hook" in command


# ── Prebuilt binaries from GitHub Releases ───────────────────────────────────

RELEASE_BASE = "https://github.com/ahillzhao-msn/YiCeNet/releases/latest/download"


def release_asset_name() -> "str | None":
    """Release asset for this OS/arch (see .github/workflows/build-release.yml), or None."""
    import platform

    machine = platform.machine().lower()
    arch = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(machine)
    if sys.platform == "win32" and arch == "x64":
        return "yicenet-hook-windows-x64.exe"
    if sys.platform.startswith("linux") and arch:
        return f"yicenet-hook-linux-{arch}"
    if sys.platform == "darwin" and arch == "arm64":
        return "yicenet-hook-macos-arm64"
    return None


def install_hook_binary(base_url: str = RELEASE_BASE) -> "Path | None":
    """Download the prebuilt client into BIN_DIR. Returns its path, or None (Python hooks stay)."""
    import os
    import urllib.request

    asset = release_asset_name()
    if asset is None:
        return None
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    dst = BIN_DIR / _EXE
    tmp = dst.with_name(dst.name + ".download")
    try:
        with urllib.request.urlopen(f"{base_url}/{asset}", timeout=60) as resp:
            tmp.write_bytes(resp.read())
    except Exception:
        tmp.unlink(missing_ok=True)
        return None
    if sys.platform != "win32":
        tmp.chmod(0o755)
    try:
        os.replace(tmp, dst)
    except PermissionError:  # Windows: a running hook holds the exe; move it aside
        old = dst.with_name(dst.name + ".old")
        old.unlink(missing_ok=True)
        dst.rename(old)
        os.replace(tmp, dst)
    return dst
