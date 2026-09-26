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


class TestFlywheelCarriesTheQuestion:
    """本卦 = the customer's question: trajectories must carry it, not a session-id placeholder."""

    def test_trajectory_is_previous_question_plus_reaction(self, tmp_path):
        from yicenet.hook_engine import HookOrchestrator
        from yicenet.tools.claude_hook import ClaudeCodeAdapter

        bank = MemoryBank()
        adapter = ClaudeCodeAdapter(process_model="daemon")
        sid = adapter.session_id({"session_id": "q1"})
        bank.store_turn(sid, 0, _vec(0), 5, summary="帮我重构登录模块",
                        metadata={"response_snippet": "已重构"})
        submitted = []
        with patch("yicenet.memory_bank.get_memory_bank", return_value=bank), \
             patch("yicenet.flywheel.submit_trajectory", side_effect=submitted.append):
            HookOrchestrator(adapter).before_prediction({"session_id": "q1", "prompt": "不对，重新来"})

        (t,) = submitted
        assert t["user_text"] == "帮我重构登录模块"
        assert t["next_user_text"] == "不对，重新来"
        assert t["trajectory"]["hexagram_evolution"] == [5]
        assert t["trajectory"]["corrected"] is True

    def test_submit_keeps_question_and_drops_the_rest(self, tmp_path):
        import json
        from yicenet import flywheel
        with patch.object(flywheel, "yicenet_data_dir", return_value=tmp_path):
            flywheel.submit_trajectory({"producer": "claude-code", "conversation_id": "c",
                                        "user_text": "问题", "next_user_text": "反应",
                                        "trajectory": {"completed": True}})
            for text in (None, "", "   ", "[loom] c2",
                         "[IMPORTANT: Background process proc_1 completed]",
                         "[CONTEXT COMPACTION — REFERENCE ONLY] ..."):
                flywheel.submit_trajectory({"producer": "loom", "conversation_id": "c2",
                                            "user_text": text, "trajectory": {}})
        rows = [json.loads(l) for l in (tmp_path / "flywheel_buffer.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [(r["user_text"], r["next_user_text"]) for r in rows] == [("问题", "反应")]

    def test_is_question(self):
        from yicenet.flywheel import is_question
        assert is_question("帮我重构登录模块")
        assert is_question("[WIP] 继续做第三步")  # brackets alone are fine
        for bad in (None, 42, "", " \n", "[claude-code] a51ffae7c48e", "[loom-hooks] sid=20260811_135",
                    "[IMPORTANT: Background process x completed]", "[CONTEXT COMPACTION — x]",
                    "<system-reminder>x</system-reminder>", "<command-name>/clear</command-name>",
                    "[Request interrupted by user]"):
            assert not is_question(bad), bad

    def test_extract_question_strips_wrappers(self):
        """Real Hermes turns: notices wrapped around the customer's words."""
        from yicenet.flywheel import extract_question
        assert extract_question(
            "[Note: model was just switched from deepseek-v4-flash to qwen3.5-9b via LM Studio. "
            "Adjust your self-identification accordingly.]\n\nhi") == "hi"
        assert extract_question(
            "[The user attached an image but it couldn't be analyzed. You can try examining it "
            "with vision_analyze using image_url: C:\\Users\\x\\clip_1.png]\n\n这是原始比赛记录，能找到哪里错了么？"
            "\n\n[Your active task list was preserved across context compression]\n"
            "- [ ] 7. 评估并实现 mine_events.py 挖掘管道（P1 全量） (pending)") == "这是原始比赛记录，能找到哪里错了么？"
        assert extract_question(
            "[Your active task list was preserved across context compression]\n"
            "- [>] 4. 引擎 GTP 测试补全 (in_progress)\n- [ ] 5. APK 构建 (pending)") == ""
        summary = "[CONTEXT COMPACTION — REFERENCE ONLY] summary…\n\n--- END OF CONTEXT SUMMARY — respond to the message below ---"
        assert extract_question(summary) == ""
        assert extract_question(summary + "\n继续吧") == "继续吧"
        assert extract_question("[IMPORTANT: Background process p completed (exit code 0).\nOutput: [x]\n完成]") == ""

    def test_scan_skips_injected_messages(self, tmp_path):
        import json
        from types import SimpleNamespace
        from yicenet import flywheel

        def sample(text):
            return SimpleNamespace(user_text=text, source="hermes", conversation_id="h1", timestamp=1.0,
                                   token_cost=0, satisfaction=0.0, continued=True, corrected=False,
                                   completed=True, praised=False, abandoned=False, response_length=0,
                                   source_msg_id="m", embedding=None)

        src = SimpleNamespace(source_id="hermes", scan_since=lambda since: [
            sample("既然是自研引擎，我们肯定是mobile优先"),
            sample("[IMPORTANT: Background process proc_24c703f526a5 completed]")])
        with patch.object(flywheel, "yicenet_data_dir", return_value=tmp_path):
            got = flywheel.scan_all_sources({"last_run": 0}, sources=[src])
        assert [s["user_text"] for s in got] == ["既然是自研引擎，我们肯定是mobile优先"]
        written = (tmp_path / "flywheel_buffer.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(written) == 1 and json.loads(written[0])["producer"] == "hermes"
