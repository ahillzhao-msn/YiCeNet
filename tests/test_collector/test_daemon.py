"""Tests for DaemonContextCollector — full 27-dim in-process accumulator."""

from __future__ import annotations

import pytest

from yicenet.hook_engine.collector.daemon import DaemonContextCollector
from yicenet.hook_engine.collector.types import SignalVector


EXPECTED_KEYS = set(SignalVector.__annotations__)


class TestDaemonBasic:

    def test_empty_build_returns_27_keys(self):
        ctx = DaemonContextCollector()
        vec = ctx.build_vector()
        assert set(vec.keys()) == EXPECTED_KEYS

    def test_all_values_are_float(self):
        ctx = DaemonContextCollector()
        vec = ctx.build_vector()
        for k, v in vec.items():
            assert isinstance(v, float), f"{k} is {type(v)}"


class TestSniffUser:

    def test_input_length_normalized(self):
        ctx = DaemonContextCollector()
        ctx.sniff_user("a" * 256)
        vec = ctx.build_vector()
        assert vec["tok_user_input_len"] == pytest.approx(0.5)

    def test_input_length_clamped_at_1(self):
        ctx = DaemonContextCollector()
        ctx.sniff_user("x" * 1024)
        vec = ctx.build_vector()
        assert vec["tok_user_input_len"] == 1.0

    def test_first_turn_flag(self):
        ctx = DaemonContextCollector()
        ctx.sniff_user("hello", is_first_turn=True)
        vec = ctx.build_vector()
        assert vec["tok_is_first_turn"] == 1.0

    def test_not_first_turn(self):
        ctx = DaemonContextCollector()
        ctx.sniff_user("hello")
        vec = ctx.build_vector()
        assert vec["tok_is_first_turn"] == 0.0


class TestSniffApi:

    def test_token_counts_normalized(self):
        ctx = DaemonContextCollector()
        ctx.sniff_api(2048, 1024, 5000.0)
        vec = ctx.build_vector()
        assert vec["tok_prompt_tokens"] == pytest.approx(0.5)
        assert vec["tok_completion_tokens"] == pytest.approx(0.25)
        assert vec["tok_api_duration"] == pytest.approx(0.5)


class TestSniffTool:

    def test_single_successful_tool(self):
        ctx = DaemonContextCollector()
        ctx.sniff_tool("bash", 0, 1000.0, 500)
        vec = ctx.build_vector()
        assert vec["tok_tool_count"] == pytest.approx(0.1)
        assert vec["tok_tool_success_rate"] == 1.0
        assert vec["tok_tool_retry_count"] == 0.0
        assert vec["tok_tool_diversity"] == pytest.approx(0.125)

    def test_mixed_tools(self):
        ctx = DaemonContextCollector()
        ctx.sniff_tool("bash", 0, 500.0, 100)
        ctx.sniff_tool("read", 0, 200.0, 200)
        ctx.sniff_tool("bash", 1, 300.0, 50)
        vec = ctx.build_vector()
        assert vec["tok_tool_count"] == pytest.approx(0.3)
        assert vec["tok_tool_success_rate"] == pytest.approx(2.0 / 3)
        assert vec["tok_tool_diversity"] == pytest.approx(2.0 / 8)

    def test_retry_counting(self):
        ctx = DaemonContextCollector()
        ctx.sniff_tool("bash", 0, 100.0)
        ctx.sniff_tool("bash", 1, 100.0)
        ctx.sniff_tool("bash", 1, 100.0)
        vec = ctx.build_vector()
        assert vec["tok_tool_retry_count"] == pytest.approx(2.0 / 5)


class TestSniffResponse:

    def test_response_length(self):
        ctx = DaemonContextCollector()
        ctx.sniff_response("x" * 2000)
        vec = ctx.build_vector()
        assert vec["tok_response_len"] == pytest.approx(0.5)

    def test_code_detection(self):
        ctx = DaemonContextCollector()
        ctx.sniff_response("Here:\n```python\nprint('hi')\n```\nDone.")
        vec = ctx.build_vector()
        assert vec["tok_has_code"] == 1.0
        assert vec["tok_code_block_count"] == pytest.approx(0.1)

    def test_no_code(self):
        ctx = DaemonContextCollector()
        ctx.sniff_response("Just text, no code blocks.")
        vec = ctx.build_vector()
        assert vec["tok_has_code"] == 0.0
        assert vec["tok_code_block_count"] == 0.0


class TestSniffHexagram:

    def test_hexagram_values(self):
        ctx = DaemonContextCollector()
        ctx.sniff_hexagram(q_max=0.8, q_gap=0.15, entropy=2.0)
        vec = ctx.build_vector()
        assert vec["tok_hex_conf"] == pytest.approx(0.8)
        assert vec["tok_hex_q_gap"] == pytest.approx(0.15)
        assert vec["tok_hex_entropy"] == pytest.approx(0.5)


class TestCrossTurn:

    def test_drift_trend_with_prev_metadata(self):
        ctx = DaemonContextCollector()
        ctx.sniff_hexagram(q_max=0.9, q_gap=0.1, entropy=1.0)
        prev = {"tok_hex_conf": 0.6, "tok_user_satisfaction": 0.5}
        vec = ctx.build_vector(prev_metadata=prev)
        assert vec["tok_drift_trend"] == pytest.approx(0.3)

    def test_prev_correction_and_praise(self):
        ctx = DaemonContextCollector()
        prev = {"tok_is_correction": 1.0, "tok_is_praise": 0.0}
        vec = ctx.build_vector(prev_metadata=prev)
        assert vec["tok_is_prev_correction"] == 1.0
        assert vec["tok_is_prev_praise"] == 0.0

    def test_mood_trend_uses_prev_satisfaction(self):
        ctx = DaemonContextCollector()
        ctx.sniff_user("good work")
        ctx.sniff_response("```python\npass\n```")
        prev = {"tok_user_satisfaction": 0.0}
        vec = ctx.build_vector(prev_metadata=prev)
        assert vec["tok_mood_trend"] != 0.0

    def test_user_speed_ratio(self):
        ctx = DaemonContextCollector()
        ctx.sniff_timing(user_interval_sec=15.0)
        prev = {"tok_user_speed": 0.5}
        vec = ctx.build_vector(prev_metadata=prev)
        assert vec["tok_user_speed"] == pytest.approx(0.25)
        assert vec["tok_user_speed_ratio"] == pytest.approx(0.5)

    def test_sniff_timing_prev_metadata_used_when_no_build_arg(self):
        ctx = DaemonContextCollector()
        ctx.sniff_hexagram(q_max=0.7, q_gap=0.05, entropy=1.0)
        ctx.sniff_timing(prev_metadata={"tok_hex_conf": 0.3})
        vec = ctx.build_vector()
        assert vec["tok_drift_trend"] == pytest.approx(0.4)

    def test_build_vector_prev_metadata_overrides_sniff_timing(self):
        ctx = DaemonContextCollector()
        ctx.sniff_hexagram(q_max=0.7, q_gap=0.05, entropy=1.0)
        ctx.sniff_timing(prev_metadata={"tok_hex_conf": 0.3})
        vec = ctx.build_vector(prev_metadata={"tok_hex_conf": 0.5})
        assert vec["tok_drift_trend"] == pytest.approx(0.2)


class TestSatisfaction:

    def test_satisfaction_in_range(self):
        ctx = DaemonContextCollector()
        ctx.sniff_user("help me")
        ctx.sniff_tool("bash", 0, 100.0)
        ctx.sniff_response("```python\npass\n```")
        ctx.sniff_hexagram(0.9, 0.1, 1.0)
        vec = ctx.build_vector()
        assert -1.0 <= vec["tok_user_satisfaction"] <= 1.0

    def test_abandon_on_empty_input(self):
        ctx = DaemonContextCollector()
        vec = ctx.build_vector()
        assert vec["tok_is_abandon"] == 1.0


class TestRepr:

    def test_repr_format(self):
        ctx = DaemonContextCollector()
        ctx.sniff_tool("bash", 0, 100.0)
        ctx.sniff_api(100, 200)
        r = repr(ctx)
        assert "DaemonContextCollector" in r
        assert "tools=1" in r
