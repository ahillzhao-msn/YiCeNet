"""Session memory: persistence, retention policy, external session manager (LOOM)."""
import os
import time
from unittest.mock import patch

import numpy as np
import pytest

import yicenet.memory_bank as mb
from yicenet.memory_bank import FileBackend, MemoryBank


def _vec(i: int) -> np.ndarray:
    v = np.zeros(8, dtype=np.float32)
    v[i % 8] = 1.0
    return v


@pytest.fixture
def fresh_singleton(monkeypatch):
    monkeypatch.setattr(mb, "_default_bank", None)
    yield
    monkeypatch.setattr(mb, "_default_bank", None)


class _Adapter:
    platform_id = "claude-code"
    process_model = "daemon"


class TestPersistence:

    def test_session_survives_restart_with_vectors(self, tmp_path):
        bank = MemoryBank(backend=FileBackend(tmp_path, store_vectors=True))
        for t in range(3):
            bank.store_turn("claude-code.s1", t, _vec(t), hexagram_id=t)

        reborn = MemoryBank(backend=FileBackend(tmp_path, store_vectors=True))
        reborn.init_session("claude-code.s1")
        assert reborn.get_last_turn("claude-code.s1").turn_id == 2
        keys, meta = reborn.get_session_keys("claude-code.s1")
        assert keys.shape == (3, 8)
        assert [m["turn_id"] for m in meta] == [0, 1, 2]

    def test_keys_and_metadata_stay_row_aligned_without_vectors(self, tmp_path):
        bank = MemoryBank(backend=FileBackend(tmp_path, store_vectors=False))
        bank.store_turn("s", 0, _vec(0), 1)
        reborn = MemoryBank(backend=FileBackend(tmp_path, store_vectors=False))
        reborn.init_session("s")
        reborn.store_turn("s", 1, _vec(1), 2)
        keys, meta = reborn.get_session_keys("s")
        assert keys.shape[0] == len(meta) == 1
        assert meta[0]["turn_id"] == 1


class TestTurnLookup:

    def test_get_turn(self):
        bank = MemoryBank()
        for t in range(3):
            bank.store_turn("s", t, _vec(t), hexagram_id=10 + t)
        assert bank.get_turn("s", 1).hexagram_id == 11
        assert bank.get_turn("s", 9) is None
        assert bank.get_turn("nope", 0) is None

    def test_stop_hook_on_second_turn_links_previous_context(self):
        """on_turn_complete looks up turn N-1 once turns are really counted."""
        from yicenet.hook_engine import HookOrchestrator
        from yicenet.tools.claude_hook import ClaudeCodeAdapter

        bank = MemoryBank()
        adapter = ClaudeCodeAdapter(process_model="daemon")
        sid = adapter.session_id({"session_id": "s2"})
        bank.store_turn(sid, 0, _vec(0), 1, metadata={"context_vector": {"tok_user_satisfaction": 0.5}})
        bank.store_turn(sid, 1, _vec(1), 2)
        adapter.new_turn({"session_id": "s2", "prompt": "x", "turn_id": 1})
        with patch("yicenet.memory_bank.get_memory_bank", return_value=bank), \
             patch.object(adapter, "assistant_response", return_value="done"):
            HookOrchestrator(adapter).on_turn_complete({"session_id": "s2"})
        meta = bank.get_turn(sid, 1).metadata
        assert meta["response_snippet"] == "done"
        assert "context_vector" in meta


class TestRetention:

    def test_idle_sessions_deleted_live_ones_trimmed(self, tmp_path):
        fb = FileBackend(tmp_path, store_vectors=True)
        for t in range(10):
            fb.append("live", mb.TurnRecord(t, _vec(t), t))
            fb.append("old", mb.TurnRecord(t, _vec(t), t))
        old = tmp_path / "old.jsonl"
        past = time.time() - 72 * 3600
        os.utime(old, (past, past))

        deleted = fb.cleanup_stale(max_age_hours=48, max_turns=4)

        assert deleted == 1 and not old.exists()
        assert [r.turn_id for r in fb.load("live")] == [6, 7, 8, 9]

    def test_turn_numbers_continue_after_trim(self, tmp_path):
        bank = MemoryBank(max_turns_per_session=3, backend=FileBackend(tmp_path))
        for t in range(5):
            bank.store_turn("s", t, _vec(t), t)
        assert bank.get_last_turn("s").turn_id == 4
        assert bank.get_turn_count("s") == 3

    def test_enforce_retention_evicts_idle_from_memory(self, tmp_path):
        bank = MemoryBank(backend=FileBackend(tmp_path))
        bank.store_turn("a", 0, _vec(0), 0)
        bank.store_turn("b", 0, _vec(0), 0)
        bank._last_access["a"] -= 72 * 3600
        assert bank.enforce_retention(max_age_hours=48) == 1
        assert bank.get_active_sessions() == ["b"]


class TestSessionManagerSwitch:

    def test_default_yicenet_persists(self, fresh_singleton, tmp_path):
        with patch.object(mb, "memory_config", return_value=dict(mb.MEMORY_DEFAULTS)), \
             patch("yicenet.config.yicenet_data_dir", return_value=tmp_path):
            mb.configure_memory_bank_for(_Adapter())
        bank = mb.get_memory_bank()
        assert isinstance(bank._backend, FileBackend)
        assert bank._max_turns == mb.MEMORY_DEFAULTS["max_turns"]

    def test_external_manager_persists_nothing(self, fresh_singleton):
        cfg = {**mb.MEMORY_DEFAULTS, "session_manager": "external"}
        with patch.object(mb, "memory_config", return_value=cfg):
            mb.configure_memory_bank_for(_Adapter())
            assert mb.external_session_manager()
        assert mb.get_memory_bank()._backend is None

    def test_user_config_overrides_defaults(self):
        with patch("yicenet.config.load_user_config",
                   return_value={"memory": {"session_manager": "external", "max_turns": 50}}):
            cfg = mb.memory_config()
        assert cfg["session_manager"] == "external"
        assert cfg["max_turns"] == 50
        assert cfg["session_ttl_hours"] == mb.MEMORY_DEFAULTS["session_ttl_hours"]
