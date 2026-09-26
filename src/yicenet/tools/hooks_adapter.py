"""HooksAdapter — ABC for all hook-lifecycle platform adapters.

Sits between PlatformAdapter (Protocol) and the concrete adapters
(ClaudeCodeAdapter, HermesAdapter).  Holds prediction + hook entry-point
logic that would otherwise be duplicated across every hook-based platform.

Dependency rule: may import from hook_engine/, engine_provider, providers,
memory_bank.  Never imports from any concrete platform adapter.

Hierarchy
---------
PlatformAdapter (Protocol)           ← structural contract for HookOrchestrator
HooksAdapter(ABC)                    ← this file
├── ClaudeCodeAdapter                ← only: session_id, assistant_response
└── HermesAdapter                    ← only: session_id, turn_id, assistant_response, platform_signals

MCPAdapter (standalone)              ← directly implements Protocol, no hook lifecycle
"""
from __future__ import annotations

import datetime
import functools
import json
import re
import sys
from abc import ABC, abstractmethod
from typing import Optional



def read_stdin_utf8() -> str:
    """Hook payload from stdin, decoded as UTF-8 (what Claude Code sends).

    Text-mode stdin on Windows uses the locale codepage and turns CJK prompts into mojibake
    with lone surrogates; read the bytes when a binary buffer exists (a real pipe), otherwise
    (e.g. an in-memory stream) fall back to the text interface.
    """
    buf = getattr(sys.stdin, "buffer", None)
    raw = buf.read().decode("utf-8", errors="replace") if buf is not None else sys.stdin.read()
    return raw.strip()

_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]")
QUESTION_MAX_CHARS = 512


class HooksAdapter(ABC):
    """Shared base for all hook-lifecycle platform adapters.

    Implements the PlatformAdapter protocol structurally (duck-typed).
    Concrete subclasses supply platform-specific field extraction;
    the shared predict / pre_message_send / stop lifecycle lives here.
    """

    # ── Context Collector (每轮一个实例) ──────────────────────────────────────

    def create_collector(self, payload: dict):
        """Factory for platform-appropriate collector.

        Default: DaemonContextCollector (works for Hermes, daemon-mode CC).
        Override in subclasses for different process models.
        """
        from yicenet.hook_engine.collector.daemon import DaemonContextCollector
        return DaemonContextCollector()

    def new_turn(self, payload: dict) -> None:
        """Start a new turn: creates a fresh ContextCollector.

        Call at the very start of pre_llm_call / predict_for_turn_payload.
        The collector accumulates signals from all hook callbacks during
        the turn, then produces a normalized vector at build_vector().
        """
        self._ctx = self.create_collector(payload)
        self._ctx.sniff_user(
            self.prompt(payload),
            is_first_turn=(self.turn_id(payload) == 0),
        )

    @property
    def ctx(self):
        """Current turn's context collector, if new_turn() was called."""
        return getattr(self, "_ctx", None)

    # ── Abstract: subclasses must implement ───────────────────────────────────

    @property
    @abstractmethod
    def platform_id(self) -> str:
        """Stable identifier injected into trajectory producer field."""

    @property
    @abstractmethod
    def process_model(self) -> str:
        """'subprocess' | 'daemon' — controls MemoryBank flush behaviour."""

    @abstractmethod
    def platform_session_id(self, payload: dict) -> str:
        """The platform's own session identifier for this payload ("" if none)."""

    @abstractmethod
    def assistant_response(self, payload: dict) -> str:
        """Full assistant response text available at turn-end hooks."""

    # ── Concrete defaults: subclasses may override ────────────────────────────

    def session_id(self, payload: dict) -> str:
        """YiCeNet's session key: "<platform_id>.<platform session id>".

        Unique across platforms sharing one daemon, stable across daemon
        restarts, and safe as a file name (the MemoryBank file is <key>.jsonl).
        Under an external session manager (LOOM) the caller's id is used as is.
        """
        raw = self.platform_session_id(payload)
        if not raw:
            return ""
        from yicenet.memory_bank import external_session_manager
        if external_session_manager():
            return str(raw)
        return f"{self.platform_id}.{_SAFE_ID.sub('_', str(raw))[:128]}"

    def turn_id(self, payload: dict) -> int:
        """Monotonically increasing turn counter (0-indexed).

        An explicit payload turn_id wins, then the platform's message history;
        otherwise YiCeNet counts turns itself from the session's persisted
        memory (Claude Code / Kimi payloads carry no history).
        """
        tid = payload.get("turn_id")
        if tid is not None:
            return int(tid)
        history = payload.get("conversation_history") or payload.get("messages")
        if history:
            return max(0, len(history) - 1)
        return self._next_turn_from_memory(self.session_id(payload))

    @staticmethod
    def _next_turn_from_memory(session_id: str) -> int:
        if not session_id:
            return 0
        from yicenet.memory_bank import get_memory_bank
        bank = get_memory_bank()
        bank.init_session(session_id)  # restores from disk after a daemon restart
        last = bank.get_last_turn(session_id)
        return last.turn_id + 1 if last is not None else 0

    def prompt(self, payload: dict) -> str:
        """Current user message text (available at turn-start hooks)."""
        return payload.get("prompt", "")

    def platform_signals(self, payload: dict) -> Optional[dict]:
        """Pre-computed feedback signals from the platform, if available.

        Default: None (most hook platforms do not pre-compute signals).
        Override in platforms that have rich conversation context (e.g. Hermes).
        """
        return None

    # ── Shared hook lifecycle ─────────────────────────────────────────────────

    def predict_for_turn_payload(self, payload: "dict | None") -> "dict | None":
        """Core prediction cycle — platform-agnostic.

        Runs before_prediction feedback, calls the YiCeNet engine, renders the
        hexagram label to stderr, and returns the JSON-serialisable result dict.
        Returns None on any failure so callers can decide the fallback.

        Delivery is intentionally NOT handled here:
          pre_message_send() prints the result to stdout (subprocess injection).
          daemon hook_server.py returns the result over HTTP (IPC delivery).
        """
        # Pin the turn number once: counting from memory would move on as soon
        # as this turn is stored.
        payload = dict(payload or {})
        payload.setdefault("turn_id", self.turn_id(payload))
        self.new_turn(payload)
        session_id = self.session_id(payload)
        turn_id = self.turn_id(payload)
        prompt = self.prompt(payload)

        try:
            self._orch.before_prediction(payload)
        except Exception:
            pass

        try:
            from yicenet.config import get_platform_config
            from yicenet.display import get_display
            from yicenet.engine_provider import EngineProvider

            engine = EngineProvider.get_engine()
            env = {
                "hour_of_day": datetime.datetime.now().hour,
                "session_turn": turn_id,
            }
            # What is known before answering: the previous turn's context.
            wm_context = {"context_pre": self._previous_context(session_id, turn_id),
                          "turn_id": turn_id}
            result = engine.predict(
                prompt or "general task",
                temperature=0.1,
                environment=env,
                session_id=session_id,
                turn_id=turn_id,
                return_prescription=True,
                # The question is what the 本卦 is cast from and what the flywheel
                # re-encodes: the chain follows the customer, not the diviner.
                turn_summary=(prompt or "")[:QUESTION_MAX_CHARS],
                wm_context=wm_context,
            )
            self._record_decision(session_id, turn_id, result, wm_context)

            # Feed hexagram Q-values into the context collector
            candidates = result.get("candidates", [])
            q_values = [c.get("q_value", 0.0) for c in candidates if "q_value" in c]
            if q_values:
                q_max = max(q_values)
                q_top = sorted(q_values, reverse=True)
                q_gap = (q_top[0] - q_top[1]) if len(q_top) > 1 else 0.0
            else:
                q_max = result.get("env_confidence", 0.0)
                q_gap = 0.0

            import math
            candidates_list = result.get("candidates", [])
            if candidates_list:
                q_vals = [c.get("q_value", 0.0) for c in candidates_list]
                total = sum(max(v, 1e-10) for v in q_vals)
                probs = [max(v, 1e-10) / total for v in q_vals]
                entropy = -sum(p * math.log(p) for p in probs)
            else:
                entropy = 0.0

            if self.ctx:
                self.ctx.sniff_hexagram(q_max, q_gap, entropy)

            prescription = result.get("context_prescription", {})

            platform_cfg = get_platform_config(self.platform_id)
            display = get_display(platform_cfg.get("display"))
            chain = None
            if session_id and display.needs_chain:
                from yicenet.memory_bank import get_memory_bank
                chain = get_memory_bank().get_hexagram_history(session_id)
            label = display.render(result, chain=chain)

            sys.stderr.write(label + "\n")
            sys.stderr.flush()

            return {
                "yicenet": {
                    "session_id":     session_id,
                    "turn_id":        turn_id,
                    "label":          label,
                    "hexagram":       result.get("selected_hexagram_name", ""),
                    "action":         result.get("action_name", ""),
                    "env_confidence": result.get("env_confidence", 0.0),
                    "context_status": result.get("context_status", "thin"),
                    "prescription":   prescription,
                }
            }
        except Exception:
            return None

    def pre_message_send(self, payload: "dict | None" = None) -> None:
        """UserPromptSubmit entry point — subprocess stdout injection.

        Reads the hook payload from stdin if not supplied, runs
        predict_for_turn_payload, and prints the result JSON to stdout
        so Claude Code injects it into the context window.
        """
        if payload is None:
            payload = self._read_payload()
        result = self.predict_for_turn_payload(payload)
        if result is None:
            sys.exit(0)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    def stop(self, payload: "dict | None" = None) -> None:
        """Stop / post-LLM entry point.

        Records the assistant response for the next turn's feedback extraction.
        For subprocess adapters, also flushes the in-memory session buffer
        (controlled by process_model in HookOrchestrator.on_turn_complete).
        """
        if payload is None:
            payload = self._read_payload()
        try:
            self._orch.on_turn_complete(payload)
        except Exception:
            pass

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _previous_context(session_id: str, turn_id: int) -> dict:
        """The previous turn's context_vector ({} when there is none)."""
        if not session_id or turn_id <= 0:
            return {}
        from yicenet.memory_bank import get_memory_bank
        bank = get_memory_bank()
        bank.init_session(session_id)
        prev = bank.get_turn(session_id, turn_id - 1)
        return dict((prev.metadata or {}).get("context_vector") or {}) if prev else {}

    @staticmethod
    def _record_decision(session_id: str, turn_id: int, result: dict, wm_context: dict) -> None:
        """Keep this turn's decision (本卦, candidates, 之卦, propensity, context_pre)
        on its TurnRecord; the next turn turns it into a trajectory."""
        decision = result.get("decision")
        if not session_id or not decision:
            return
        from yicenet.memory_bank import get_memory_bank
        get_memory_bank().update_turn_metadata(
            session_id, turn_id, {**decision, "context_pre": wm_context["context_pre"]})

    @functools.cached_property
    def _orch(self):
        """Lazy HookOrchestrator bound to this adapter instance."""
        from yicenet.hook_engine import HookOrchestrator
        return HookOrchestrator(self)

    @staticmethod
    def _read_payload() -> dict:
        try:
            raw = read_stdin_utf8()
            return json.loads(raw) if raw else {}
        except Exception:
            return {}
