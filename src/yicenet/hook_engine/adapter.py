"""
hook_engine.adapter — PlatformAdapter interface.

Each platform (Claude Code, Hermes, future) provides one concrete adapter.
All platform-specific knowledge lives here; the orchestrator is unaware of it.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class PlatformAdapter(Protocol):
    """Translate platform-specific hook payloads into domain primitives.

    Implementors:
      ClaudeCodeAdapter — tools/claude_hook.py
      HermesAdapter     — tools/hermes_hook.py
    """

    @property
    def platform_id(self) -> str:
        """Stable identifier injected into trajectory producer field.
        e.g. 'claude-code', 'hermes'
        """
        ...

    @property
    def process_model(self) -> str:
        """'subprocess' — short-lived process per hook event (Claude Code).
        'daemon'     — long-lived service process (Hermes, Loom).

        Determines whether MemoryBank needs a FileBackend for cross-event
        persistence.  subprocess always needs it; daemon uses config opt-in.
        """
        ...

    def session_id(self, payload: dict) -> str:
        """Derive a stable, short session identifier from the hook payload."""
        ...

    def turn_id(self, payload: dict) -> int:
        """Monotonically increasing turn counter within the session."""
        ...

    def prompt(self, payload: dict) -> str:
        """Current user message text (available at turn-start hooks)."""
        ...

    def assistant_response(self, payload: dict) -> str:
        """Full assistant response text (available at turn-end hooks).

        For Claude Code: read from transcript file.
        For Hermes: read from context['response'] or context['assistant_response'].
        """
        ...

    def platform_signals(self, payload: dict) -> Optional[dict]:
        """Pre-computed feedback signals from the platform, if available.

        When not None, the orchestrator uses these directly and skips
        text-based extraction.  Claude Code always returns None.
        Hermes may return signals from conversation_history analysis.
        """
        ...
