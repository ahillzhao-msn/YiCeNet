"""ContextCollector ABC — platform-independent signal accumulation interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class ContextCollector(ABC):
    """Per-turn environment signal accumulator.

    Three implementations match three platform process models:
      DaemonContextCollector  — in-memory (Hermes, CC daemon)
      SubprocessContextCollector — file-based (CC subprocess)
      ExplicitContextCollector — pre-built (MCP, tests)
    """

    @abstractmethod
    def sniff_user(self, text: str, is_first_turn: bool = False) -> None:
        ...

    @abstractmethod
    def sniff_api(self, prompt_tokens: int, completion_tokens: int,
                  duration_ms: float = 0.0) -> None:
        ...

    @abstractmethod
    def sniff_tool(self, name: str, exit_code: int,
                   duration_ms: float, result_size_bytes: int = 0) -> None:
        ...

    @abstractmethod
    def sniff_response(self, text: str) -> None:
        ...

    @abstractmethod
    def sniff_hexagram(self, q_max: float, q_gap: float,
                       entropy: float) -> None:
        ...

    @abstractmethod
    def sniff_timing(self, user_interval_sec: Optional[float] = None,
                     prev_metadata: Optional[dict] = None) -> None:
        ...

    @abstractmethod
    def build_vector(self, prev_metadata: Optional[dict] = None) -> dict:
        ...
