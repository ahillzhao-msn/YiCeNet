"""Tests for ContextCollector ABC contract."""

from __future__ import annotations

import pytest


class TestContextCollectorABC:

    def test_cannot_instantiate_abc(self):
        from yicenet.hook_engine.collector.interface import ContextCollector
        with pytest.raises(TypeError):
            ContextCollector()

    def test_incomplete_subclass_raises(self):
        from yicenet.hook_engine.collector.interface import ContextCollector

        class Partial(ContextCollector):
            def sniff_user(self, text, is_first_turn=False): ...
            def sniff_api(self, pt, ct, d=0.0): ...

        with pytest.raises(TypeError):
            Partial()

    def test_complete_subclass_instantiates(self):
        from yicenet.hook_engine.collector.interface import ContextCollector

        class Complete(ContextCollector):
            def sniff_user(self, text, is_first_turn=False): pass
            def sniff_api(self, pt, ct, d=0.0): pass
            def sniff_tool(self, n, ec, d, rs=0): pass
            def sniff_response(self, text): pass
            def sniff_hexagram(self, qm, qg, e): pass
            def sniff_timing(self, ui=None, pm=None): pass
            def build_vector(self, pm=None): return {}

        c = Complete()
        assert c.build_vector() == {}

    def test_all_three_implementations_are_subclasses(self):
        from yicenet.hook_engine.collector.interface import ContextCollector
        from yicenet.hook_engine.collector.daemon import DaemonContextCollector
        from yicenet.hook_engine.collector.subprocess import SubprocessContextCollector
        from yicenet.hook_engine.collector.explicit import ExplicitContextCollector

        assert issubclass(DaemonContextCollector, ContextCollector)
        assert issubclass(SubprocessContextCollector, ContextCollector)
        assert issubclass(ExplicitContextCollector, ContextCollector)

    def test_signal_vector_has_27_fields(self):
        from yicenet.hook_engine.collector.types import SignalVector
        assert len(SignalVector.__annotations__) == 27

    def test_all_signal_vector_keys_start_with_tok(self):
        from yicenet.hook_engine.collector.types import SignalVector
        for key in SignalVector.__annotations__:
            assert key.startswith("tok_"), f"{key} does not start with tok_"
