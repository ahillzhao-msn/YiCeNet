"""PlatformInstaller ABC — contract for all platform adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class PlatformInstaller(ABC):
    """One concrete subclass per target platform (Hermes, Claude Code, …)."""

    @abstractmethod
    def detect(self) -> bool:
        """Return True if the target platform is present on this machine."""

    @abstractmethod
    def install_package(self, editable_path: Path = None) -> bool:
        """pip install yicenet into the platform's venv.

        Args:
            editable_path: if set, pip install -e <path>; else pip install yicenet.
        Returns True on success.
        """

    @abstractmethod
    def register_hooks(self) -> None:
        """Write lifecycle hook configuration for this platform."""

    @abstractmethod
    def unregister(self) -> None:
        """Remove all hook configuration written by register_hooks()."""

    def install(self, editable_path: Path = None) -> bool:
        """Full install: package + hooks. Returns False if platform not found."""
        if not self.detect():
            return False
        ok = self.install_package(editable_path)
        if ok:
            self.register_hooks()
        return ok
