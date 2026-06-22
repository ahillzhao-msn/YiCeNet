"""Tests for ExplicitContextCollector — pre-built signal vectors."""

from __future__ import annotations

import pytest

from yicenet.hook_engine.collector.explicit import ExplicitContextCollector
from yicenet.hook_engine.collector.types import SignalVector


EXPECTED_KEYS = set(SignalVector.__annotations__)


class TestExplicitBasic:

    def test_empty_returns_27_zero_keys(self):
        c = ExplicitContextCollector()
        vec = c.build_vector()
        assert set(vec.keys()) == EXPECTED_KEYS
        assert all(v == 0.0 for v in vec.values())

    def test_signals_pass_through(self):
        signals = {"tok_hex_conf": 0.9, "tok_is_first_turn": 1.0}
        c = ExplicitContextCollector(signals)
        vec = c.build_vector()
        assert vec["tok_hex_conf"] == 0.9
        assert vec["tok_is_first_turn"] == 1.0
        assert vec["tok_user_input_len"] == 0.0

    def test_unknown_keys_ignored(self):
        c = ExplicitContextCollector({"not_a_signal": 42.0})
        vec = c.build_vector()
        assert "not_a_signal" not in vec
        assert set(vec.keys()) == EXPECTED_KEYS


class TestSniffNoops:

    def test_sniff_methods_are_noops(self):
        c = ExplicitContextCollector({"tok_hex_conf": 0.5})
        c.sniff_user("hello", is_first_turn=True)
        c.sniff_api(100, 200, 300.0)
        c.sniff_tool("bash", 0, 100.0, 50)
        c.sniff_response("```code```")
        c.sniff_hexagram(1.0, 0.5, 2.0)
        c.sniff_timing(user_interval_sec=10.0)
        vec = c.build_vector()
        assert vec["tok_hex_conf"] == 0.5
        assert vec["tok_is_first_turn"] == 0.0

    def test_prev_metadata_ignored(self):
        c = ExplicitContextCollector({"tok_drift_trend": 0.3})
        vec = c.build_vector(prev_metadata={"tok_hex_conf": 0.9})
        assert vec["tok_drift_trend"] == 0.3


class TestAsABC:

    def test_is_context_collector_subclass(self):
        from yicenet.hook_engine.collector.interface import ContextCollector
        assert issubclass(ExplicitContextCollector, ContextCollector)

    def test_isinstance_check(self):
        from yicenet.hook_engine.collector.interface import ContextCollector
        c = ExplicitContextCollector()
        assert isinstance(c, ContextCollector)
