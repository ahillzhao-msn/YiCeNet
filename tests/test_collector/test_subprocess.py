"""Tests for SubprocessContextCollector — file-based cross-process accumulator."""

from __future__ import annotations

import json

import pytest

from yicenet.hook_engine.collector.subprocess import SubprocessContextCollector
from yicenet.hook_engine.collector.types import SignalVector


EXPECTED_KEYS = set(SignalVector.__annotations__)


@pytest.fixture
def ctx_dir(tmp_path, monkeypatch):
    """Redirect the side-channel base directory to a temp path."""
    monkeypatch.setattr(SubprocessContextCollector, "_BASE", tmp_path)
    return tmp_path


class TestFileIO:

    def test_sniff_creates_jsonl_file(self, ctx_dir):
        c = SubprocessContextCollector("sess1", 0)
        c.sniff_user("hello", is_first_turn=True)
        path = ctx_dir / "sess1.0.jsonl"
        assert path.exists()
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["type"] == "user"
        assert event["text"] == "hello"
        assert event["is_first"] is True

    def test_multiple_events_append(self, ctx_dir):
        c = SubprocessContextCollector("sess1", 0)
        c.sniff_user("hi")
        c.sniff_tool("bash", 0, 100.0, 50)
        c.sniff_api(500, 200, 1000.0)
        path = ctx_dir / "sess1.0.jsonl"
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

    def test_cleanup_removes_file(self, ctx_dir):
        c = SubprocessContextCollector("sess1", 0)
        c.sniff_user("hi")
        path = ctx_dir / "sess1.0.jsonl"
        assert path.exists()
        c.cleanup()
        assert not path.exists()

    def test_cleanup_noop_when_no_file(self, ctx_dir):
        c = SubprocessContextCollector("sess1", 99)
        c.cleanup()  # must not raise


class TestCrossProcessReplay:

    def test_separate_instances_share_file(self, ctx_dir):
        """Simulate two subprocess invocations writing to the same turn file."""
        c1 = SubprocessContextCollector("sess1", 0)
        c1.sniff_user("write code for me", is_first_turn=True)
        c1.sniff_hexagram(0.8, 0.1, 1.5)

        c2 = SubprocessContextCollector("sess1", 0)
        c2.sniff_tool("bash", 0, 500.0, 100)
        c2.sniff_tool("read", 0, 200.0, 50)

        c3 = SubprocessContextCollector("sess1", 0)
        c3.sniff_response("```python\nprint('done')\n```")
        vec = c3.build_vector()

        assert set(vec.keys()) == EXPECTED_KEYS
        assert vec["tok_is_first_turn"] == 1.0
        assert vec["tok_tool_count"] == pytest.approx(0.2)
        assert vec["tok_has_code"] == 1.0
        assert vec["tok_hex_conf"] == pytest.approx(0.8)

    def test_build_vector_returns_27_keys(self, ctx_dir):
        c = SubprocessContextCollector("s", 0)
        c.sniff_user("test")
        vec = c.build_vector()
        assert set(vec.keys()) == EXPECTED_KEYS

    def test_build_vector_with_prev_metadata(self, ctx_dir):
        c = SubprocessContextCollector("s", 1)
        c.sniff_user("next turn")
        c.sniff_hexagram(0.9, 0.05, 1.0)
        prev = {"tok_hex_conf": 0.5}
        vec = c.build_vector(prev_metadata=prev)
        assert vec["tok_drift_trend"] == pytest.approx(0.4)


class TestSniffAll:

    def test_sniff_api(self, ctx_dir):
        c = SubprocessContextCollector("s", 0)
        c.sniff_api(1024, 512, 2000.0)
        vec = c.build_vector()
        assert vec["tok_prompt_tokens"] == pytest.approx(0.25)
        assert vec["tok_completion_tokens"] == pytest.approx(0.125)

    def test_sniff_timing(self, ctx_dir):
        c = SubprocessContextCollector("s", 0)
        c.sniff_timing(user_interval_sec=30.0)
        vec = c.build_vector()
        assert vec["tok_user_speed"] == pytest.approx(0.5)

    def test_sniff_hexagram(self, ctx_dir):
        c = SubprocessContextCollector("s", 0)
        c.sniff_hexagram(0.7, 0.2, 3.0)
        vec = c.build_vector()
        assert vec["tok_hex_conf"] == pytest.approx(0.7)
        assert vec["tok_hex_q_gap"] == pytest.approx(0.2)
        assert vec["tok_hex_entropy"] == pytest.approx(0.75)
