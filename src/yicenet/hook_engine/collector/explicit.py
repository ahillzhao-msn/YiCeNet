"""ExplicitContextCollector — pre-built signals for MCP and tests."""

from __future__ import annotations

from typing import Optional

from .interface import ContextCollector
from .types import SignalVector


_SIGNAL_KEYS = list(SignalVector.__annotations__)


class ExplicitContextCollector(ContextCollector):
    """All signals passed via constructor; sniff methods are no-ops.

    Used by MCP tools (signals arrive as explicit arguments) and test
    harnesses (construct test vectors directly).
    """

    def __init__(self, signals: Optional[dict] = None) -> None:
        self._data = signals or {}

    def _noop(self, *args, **kwargs) -> None:
        pass

    sniff_user = sniff_api = sniff_tool = sniff_response = _noop  # type: ignore[assignment]
    sniff_hexagram = sniff_timing = _noop  # type: ignore[assignment]

    def build_vector(self, prev_metadata: Optional[dict] = None) -> dict:
        return {k: self._data.get(k, 0.0) for k in _SIGNAL_KEYS}
