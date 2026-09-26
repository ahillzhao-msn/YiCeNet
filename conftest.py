"""pytest configuration: ensure src/ is on sys.path for tests."""
import sys
from pathlib import Path

src = Path(__file__).parent / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))


import pytest


@pytest.fixture(autouse=True)
def _isolate_native_hook(tmp_path_factory, monkeypatch):
    """Installers write ~/.yicenet/daemon-python and look for ~/.yicenet/bin; keep tests off the real home."""
    from yicenet.install import native
    root = tmp_path_factory.mktemp("yicenet-native")
    monkeypatch.setattr(native, "BIN_DIR", root / "bin")
    monkeypatch.setattr(native, "DAEMON_PYTHON_FILE", root / "daemon-python")
