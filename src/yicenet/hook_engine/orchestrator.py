"""
hook_engine.orchestrator — Platform-independent turn lifecycle coordinator.

Reads and writes MemoryBank; delegates platform differences to the adapter.
The orchestrator knows nothing about Claude Code, Hermes, or file formats.

Turn lifecycle
--------------
before_prediction(payload)
    └─ consume last turn's pending feedback → submit_trajectory()

[platform does its own prediction → engine.store_turn() writes to MemoryBank]

on_turn_complete(payload)
    └─ read response from platform → update_turn_metadata() → flush (subprocess only)
"""
from __future__ import annotations

from .adapter import PlatformAdapter
from .extractor import extract_feedback, signals_from_platform, build_trajectory


class HookOrchestrator:
    """Coordinates cross-turn feedback using MemoryBank + PlatformAdapter."""

    def __init__(self, adapter: PlatformAdapter) -> None:
        self._adapter = adapter

    # ── public hook points ────────────────────────────────────────────────────

    def before_prediction(self, payload: dict) -> None:
        """Extract and submit feedback from the previous turn.

        Call at the very start of each turn, BEFORE platform prediction.
        Submits a real trajectory for Turn N-1 based on Turn N's prompt.
        No-op on the first turn (no previous turn in MemoryBank).
        """
        session_id = self._adapter.session_id(payload)
        bank = self._bank()
        bank.init_session(session_id)   # loads from FileBackend if subprocess

        last = bank.get_last_turn(session_id)
        if last is None or not last.metadata.get("response_snippet"):
            return  # no previous turn with response data yet

        # Prefer platform-provided signals (Hermes); fall back to text extraction
        raw = last.metadata.get("platform_signals")
        if raw:
            signals = signals_from_platform(raw)
        else:
            signals = extract_feedback(self._adapter.prompt(payload), last)

        try:
            from yicenet.flywheel import submit_trajectory
            submit_trajectory(
                build_trajectory(signals, last, session_id, self._adapter.platform_id)
            )
        except Exception:
            pass

    def on_turn_complete(self, payload: dict) -> None:
        """Record this turn's response for the next turn's feedback extraction.

        Call at turn end (Stop / post_llm_call), AFTER the assistant has replied.
        Updates the last TurnRecord's metadata with response_snippet and
        response_char_count, then flushes in-memory state (subprocess only).
        """
        session_id = self._adapter.session_id(payload)
        bank = self._bank()
        bank.init_session(session_id)

        last = bank.get_last_turn(session_id)
        if last is None:
            return

        response = self._adapter.assistant_response(payload)
        metadata: dict = {
            "response_snippet":    response[:300],
            "response_char_count": len(response),
        }
        raw_signals = self._adapter.platform_signals(payload)
        if raw_signals:
            metadata["platform_signals"] = raw_signals

        bank.update_turn_metadata(session_id, last.turn_id, metadata)

        # Subprocess hooks die after this call — flush memory (file is kept).
        # Daemon processes keep the session live for attention; no flush needed.
        if self._adapter.process_model == "subprocess":
            bank.flush_session(session_id)

    # ── internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _bank():
        from yicenet.memory_bank import get_memory_bank
        return get_memory_bank()
