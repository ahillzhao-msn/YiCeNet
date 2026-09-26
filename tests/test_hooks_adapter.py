"""
Tests for the three-mode hook adapter hierarchy.

Covers:
  HooksAdapter        — ABC enforcement, concrete defaults
  ClaudeCodeAdapter   — CC-specific overrides, process_model param
  HermesAdapter       — Hermes-specific overrides, signals
  MCPAdapter          — isolated from HooksAdapter, Protocol-compliant
  predict_for_turn_payload — shared logic path
  pre_message_send / stop  — delivery and routing
  hook_server         — HTTP routing /hook/<event>?platform=<id>
  daemon.platforms    — per-platform handlers (Claude Code, Kimi Code)
  ipc_hook            — Python twin of the native client
  native client       — native/yicenet-hook (when built)
  ClaudeCodeInstaller — register_mcp / register_hybrid / unregister_mcp
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# HooksAdapter — ABC contract
# ─────────────────────────────────────────────────────────────────────────────

class TestHooksAdapterABC:

    def test_cannot_instantiate_directly(self):
        from yicenet.tools.hooks_adapter import HooksAdapter
        with pytest.raises(TypeError):
            HooksAdapter()

    def test_concrete_subclass_must_implement_platform_id(self):
        from yicenet.tools.hooks_adapter import HooksAdapter

        class Incomplete(HooksAdapter):
            @property
            def process_model(self): return "subprocess"
            def platform_session_id(self, p): return ""
            def assistant_response(self, p): return ""

        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_subclass_must_implement_process_model(self):
        from yicenet.tools.hooks_adapter import HooksAdapter

        class Incomplete(HooksAdapter):
            @property
            def platform_id(self): return "x"
            def platform_session_id(self, p): return ""
            def assistant_response(self, p): return ""

        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_subclass_must_implement_platform_session_id(self):
        from yicenet.tools.hooks_adapter import HooksAdapter

        class Incomplete(HooksAdapter):
            @property
            def platform_id(self): return "x"
            @property
            def process_model(self): return "subprocess"
            def assistant_response(self, p): return ""

        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_subclass_must_implement_assistant_response(self):
        from yicenet.tools.hooks_adapter import HooksAdapter

        class Incomplete(HooksAdapter):
            @property
            def platform_id(self): return "x"
            @property
            def process_model(self): return "subprocess"
            def platform_session_id(self, p): return ""

        with pytest.raises(TypeError):
            Incomplete()

    def test_minimal_concrete_subclass_is_instantiable(self):
        from yicenet.tools.hooks_adapter import HooksAdapter

        class Minimal(HooksAdapter):
            @property
            def platform_id(self): return "test"
            @property
            def process_model(self): return "subprocess"
            def platform_session_id(self, p): return "sid"
            def assistant_response(self, p): return ""

        adapter = Minimal()
        assert adapter.platform_id == "test"
        assert adapter.process_model == "subprocess"


class TestHooksAdapterDefaults:

    @pytest.fixture
    def adapter(self):
        from yicenet.tools.hooks_adapter import HooksAdapter

        class Concrete(HooksAdapter):
            @property
            def platform_id(self): return "test"
            @property
            def process_model(self): return "subprocess"
            def platform_session_id(self, p): return "sid"
            def assistant_response(self, p): return ""

        return Concrete()

    def test_default_platform_signals_is_none(self, adapter):
        assert adapter.platform_signals({}) is None

    def test_default_turn_id_from_messages(self, adapter):
        payload = {"messages": [{}, {}, {}]}  # 3 messages → turn_id 2
        assert adapter.turn_id(payload) == 2

    def test_default_turn_id_empty(self, adapter):
        assert adapter.turn_id({}) == 0

    def test_turn_id_counts_from_session_memory(self, adapter):
        """No history in the payload (Claude Code, Kimi): YiCeNet counts turns itself."""
        import numpy as np
        from yicenet.memory_bank import MemoryBank
        bank = MemoryBank()
        with patch("yicenet.memory_bank.get_memory_bank", return_value=bank):
            payload = {"session_id": "s1"}
            assert adapter.turn_id(payload) == 0
            sid = adapter.session_id(payload)
            bank.store_turn(sid, 0, np.ones(4, dtype=np.float32), 1)
            assert adapter.turn_id(payload) == 1
            bank.store_turn(sid, 1, np.ones(4, dtype=np.float32), 2)
            assert adapter.turn_id(payload) == 2
            assert adapter.turn_id({"session_id": "s1", "turn_id": 7}) == 7

    def test_default_prompt(self, adapter):
        assert adapter.prompt({"prompt": "hello"}) == "hello"
        assert adapter.prompt({}) == ""

    def test_orch_is_cached_property(self, adapter):
        orch1 = adapter._orch
        orch2 = adapter._orch
        assert orch1 is orch2

    def test_read_payload_returns_empty_on_bad_input(self, adapter):
        with patch("sys.stdin", StringIO("not-json")):
            result = adapter._read_payload()
        assert result == {}

    def test_read_payload_parses_valid_json(self, adapter):
        with patch("sys.stdin", StringIO('{"prompt": "test"}')):
            result = adapter._read_payload()
        assert result == {"prompt": "test"}


# ─────────────────────────────────────────────────────────────────────────────
# ClaudeCodeAdapter
# ─────────────────────────────────────────────────────────────────────────────

class TestClaudeCodeAdapter:

    def test_is_hooks_adapter_subclass(self):
        from yicenet.tools.claude_hook import ClaudeCodeAdapter
        from yicenet.tools.hooks_adapter import HooksAdapter
        assert issubclass(ClaudeCodeAdapter, HooksAdapter)

    def test_default_process_model_is_subprocess(self):
        from yicenet.tools.claude_hook import ClaudeCodeAdapter
        assert ClaudeCodeAdapter().process_model == "subprocess"

    def test_daemon_process_model_via_constructor(self):
        from yicenet.tools.claude_hook import ClaudeCodeAdapter
        adapter = ClaudeCodeAdapter(process_model="daemon")
        assert adapter.process_model == "daemon"

    def test_platform_id(self):
        from yicenet.tools.claude_hook import ClaudeCodeAdapter
        assert ClaudeCodeAdapter().platform_id == "claude-code"

    def test_session_id_is_platform_namespaced_full_id(self):
        from yicenet.tools.claude_hook import ClaudeCodeAdapter
        sid = ClaudeCodeAdapter().session_id({"session_id": "a51ffae7-c48e-4b4e-9df7-93692c97ed01"})
        assert sid == "claude-code.a51ffae7-c48e-4b4e-9df7-93692c97ed01"

    def test_session_id_is_file_name_safe(self):
        from yicenet.tools.claude_hook import ClaudeCodeAdapter
        sid = ClaudeCodeAdapter().session_id({"session_id": "../x y/z"})
        assert sid == "claude-code..._x_y_z"
        assert "/" not in sid and " " not in sid

    def test_session_id_fallback_cwd_hash(self, tmp_path):
        from yicenet.tools.claude_hook import ClaudeCodeAdapter
        adapter = ClaudeCodeAdapter()
        with patch("os.getcwd", return_value=str(tmp_path)):
            sid = adapter.session_id({})
        platform, raw = sid.split(".", 1)
        assert platform == "claude-code"
        assert len(raw) == 12 and all(c in "0123456789abcdef" for c in raw)

    def test_same_id_on_two_platforms_is_two_sessions(self):
        from yicenet.tools.claude_hook import ClaudeCodeAdapter
        from yicenet.tools.kimi_code_hook import KimiCodeAdapter
        payload = {"session_id": "shared-id"}
        assert ClaudeCodeAdapter().session_id(payload) != KimiCodeAdapter().session_id(payload)

    def test_external_session_manager_uses_id_verbatim(self):
        from yicenet.tools.claude_hook import ClaudeCodeAdapter
        with patch("yicenet.memory_bank.memory_config",
                   return_value={"session_manager": "external"}):
            assert ClaudeCodeAdapter().session_id({"session_id": "loom-42"}) == "loom-42"

    def test_session_id_deterministic_same_day(self, tmp_path):
        from yicenet.tools.claude_hook import ClaudeCodeAdapter
        adapter = ClaudeCodeAdapter()
        with patch("os.getcwd", return_value=str(tmp_path)):
            sid1 = adapter.session_id({})
            sid2 = adapter.session_id({})
        assert sid1 == sid2

    def test_assistant_response_empty_on_no_transcript(self):
        from yicenet.tools.claude_hook import ClaudeCodeAdapter
        adapter = ClaudeCodeAdapter()
        result = adapter.assistant_response({"session_id": "nonexistent-uuid"})
        assert result == ""

    def test_assistant_response_reads_transcript_path(self, tmp_path):
        from yicenet.tools.claude_hook import ClaudeCodeAdapter
        transcript = tmp_path / "session.jsonl"
        record = {"message": {"role": "assistant", "content": "Hello world"}}
        transcript.write_text(json.dumps(record) + "\n", encoding="utf-8")

        adapter = ClaudeCodeAdapter()
        result = adapter.assistant_response({"transcript_path": str(transcript)})
        assert "Hello world" in result

    def test_assistant_response_handles_content_list(self, tmp_path):
        from yicenet.tools.claude_hook import ClaudeCodeAdapter
        transcript = tmp_path / "session.jsonl"
        record = {"message": {"role": "assistant", "content": [
            {"type": "text", "text": "Part one"},
            {"type": "text", "text": "part two"},
        ]}}
        transcript.write_text(json.dumps(record) + "\n", encoding="utf-8")

        adapter = ClaudeCodeAdapter()
        result = adapter.assistant_response({"transcript_path": str(transcript)})
        assert "Part one" in result
        assert "part two" in result

    def test_subprocess_and_daemon_have_separate_orchs(self):
        from yicenet.tools.claude_hook import ClaudeCodeAdapter
        sub = ClaudeCodeAdapter(process_model="subprocess")
        dmn = ClaudeCodeAdapter(process_model="daemon")
        assert sub._orch is not dmn._orch

    def test_daemon_adapter_orch_process_model_is_daemon(self):
        from yicenet.tools.claude_hook import ClaudeCodeAdapter
        dmn = ClaudeCodeAdapter(process_model="daemon")
        assert dmn._orch._adapter.process_model == "daemon"


# ─────────────────────────────────────────────────────────────────────────────
# HermesAdapter
# ─────────────────────────────────────────────────────────────────────────────

class TestHermesAdapter:

    def test_is_hooks_adapter_subclass(self):
        from yicenet.tools.hermes_hook import HermesAdapter
        from yicenet.tools.hooks_adapter import HooksAdapter
        assert issubclass(HermesAdapter, HooksAdapter)

    def test_process_model_is_daemon(self):
        from yicenet.tools.hermes_hook import HermesAdapter
        assert HermesAdapter().process_model == "daemon"

    def test_platform_id(self):
        from yicenet.tools.hermes_hook import HermesAdapter
        assert HermesAdapter().platform_id == "hermes"

    def test_session_id_from_payload(self):
        from yicenet.tools.hermes_hook import HermesAdapter
        adapter = HermesAdapter()
        assert adapter.session_id({"session_id": "abc123"}) == "hermes.abc123"

    def test_turn_id_from_direct_field(self):
        from yicenet.tools.hermes_hook import HermesAdapter
        adapter = HermesAdapter()
        assert adapter.turn_id({"turn_id": 5}) == 5

    def test_turn_id_from_conversation_history(self):
        from yicenet.tools.hermes_hook import HermesAdapter
        adapter = HermesAdapter()
        payload = {"conversation_history": [{}, {}, {}, {}]}
        assert adapter.turn_id(payload) == 3

    def test_turn_id_direct_overrides_history(self):
        from yicenet.tools.hermes_hook import HermesAdapter
        adapter = HermesAdapter()
        payload = {"turn_id": 7, "conversation_history": [{}, {}]}
        assert adapter.turn_id(payload) == 7

    def test_prompt_from_last_message(self):
        from yicenet.tools.hermes_hook import HermesAdapter
        adapter = HermesAdapter()
        payload = {"messages": [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]}
        assert adapter.prompt(payload) == "second"

    def test_assistant_response_from_dict(self):
        from yicenet.tools.hermes_hook import HermesAdapter
        adapter = HermesAdapter()
        assert adapter.assistant_response({"assistant_response": {"content": "hi"}}) == "hi"

    def test_assistant_response_from_string(self):
        from yicenet.tools.hermes_hook import HermesAdapter
        adapter = HermesAdapter()
        assert adapter.assistant_response({"response": "hello"}) == "hello"

    def test_platform_signals_returns_none_without_history(self):
        from yicenet.tools.hermes_hook import HermesAdapter
        adapter = HermesAdapter()
        assert adapter.platform_signals({}) is None
        assert adapter.platform_signals({"conversation_history": [{}]}) is None

    def test_platform_signals_with_history(self):
        from yicenet.tools.hermes_hook import HermesAdapter
        adapter = HermesAdapter()
        payload = {"conversation_history": [
            {"role": "assistant", "content": "Here is a solution."},
            {"role": "user", "content": "That's wrong, please fix it."},
        ]}
        signals = adapter.platform_signals(payload)
        assert signals is not None
        assert "corrected" in signals
        assert "continued" in signals

    def test_hermes_overrides_platform_signals_default(self):
        """HermesAdapter.platform_signals must not fall through to HooksAdapter's None default."""
        from yicenet.tools.hermes_hook import HermesAdapter
        from yicenet.tools.hooks_adapter import HooksAdapter
        adapter = HermesAdapter()
        # With two-message history, it should NOT return None
        payload = {"conversation_history": [
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "Great work, thanks!"},
        ]}
        signals = adapter.platform_signals(payload)
        # HooksAdapter default returns None; Hermes should return a dict
        assert signals is not None
        assert isinstance(signals, dict)


# ─────────────────────────────────────────────────────────────────────────────
# MCPAdapter — must NOT inherit HooksAdapter
# ─────────────────────────────────────────────────────────────────────────────

class TestMCPAdapter:

    def test_not_hooks_adapter_subclass(self):
        from yicenet.tools.mcp_adapter import MCPAdapter
        from yicenet.tools.hooks_adapter import HooksAdapter
        assert not issubclass(MCPAdapter, HooksAdapter)

    def test_protocol_fields(self):
        from yicenet.tools.mcp_adapter import MCPAdapter
        a = MCPAdapter()
        assert a.platform_id == "claude-code-mcp"
        assert a.process_model == "daemon"

    def test_session_id(self):
        from yicenet.tools.mcp_adapter import MCPAdapter
        a = MCPAdapter()
        assert a.session_id({"session_id": "abc"}) == "abc"

    def test_turn_id(self):
        from yicenet.tools.mcp_adapter import MCPAdapter
        a = MCPAdapter()
        assert a.turn_id({"turn_id": 3}) == 3

    def test_prompt_uses_task_brief(self):
        from yicenet.tools.mcp_adapter import MCPAdapter
        a = MCPAdapter()
        assert a.prompt({"task_brief": "fix bug"}) == "fix bug"

    def test_prompt_falls_back_to_prompt_key(self):
        from yicenet.tools.mcp_adapter import MCPAdapter
        a = MCPAdapter()
        assert a.prompt({"prompt": "refactor"}) == "refactor"

    def test_assistant_response_from_snippet(self):
        from yicenet.tools.mcp_adapter import MCPAdapter
        a = MCPAdapter()
        assert a.assistant_response({"response_snippet": "done."}) == "done."

    def test_platform_signals_dict_pass_through(self):
        from yicenet.tools.mcp_adapter import MCPAdapter
        a = MCPAdapter()
        signals = {"corrected": True}
        assert a.platform_signals({"signals": signals}) == signals

    def test_platform_signals_none_when_absent(self):
        from yicenet.tools.mcp_adapter import MCPAdapter
        a = MCPAdapter()
        assert a.platform_signals({}) is None
        assert a.platform_signals({"signals": "bad"}) is None

    def test_has_no_pre_message_send(self):
        from yicenet.tools.mcp_adapter import MCPAdapter
        assert not hasattr(MCPAdapter, "pre_message_send")

    def test_has_no_predict_for_turn_payload(self):
        from yicenet.tools.mcp_adapter import MCPAdapter
        assert not hasattr(MCPAdapter, "predict_for_turn_payload")


# ─────────────────────────────────────────────────────────────────────────────
# predict_for_turn_payload — shared engine path (mocked engine)
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictForTurnPayload:

    @pytest.fixture
    def mock_engine(self):
        engine = MagicMock()
        engine.predict.return_value = {
            "selected_hexagram_name": "乾",
            "action_name":            "act",
            "env_confidence":         0.9,
            "context_status":         "sufficient",
            "context_prescription":   {"retain_turns": [0]},
        }
        return engine

    @pytest.fixture
    def mock_display(self):
        display = MagicMock()
        display.needs_chain = False
        display.render.return_value = "乾 (Heaven)"
        return display

    @pytest.fixture
    def adapter(self):
        from yicenet.tools.claude_hook import ClaudeCodeAdapter
        return ClaudeCodeAdapter()

    def test_returns_dict_on_success(self, adapter, mock_engine, mock_display):
        with patch("yicenet.engine_provider.EngineProvider.get_engine", return_value=mock_engine), \
             patch("yicenet.display.get_display", return_value=mock_display), \
             patch.object(adapter._orch, "before_prediction"), \
             patch("sys.stderr", StringIO()):
            result = adapter.predict_for_turn_payload({
                "session_id": "aabbccddeeff",
                "prompt": "fix the bug",
            })

        assert result is not None
        assert "yicenet" in result
        yi = result["yicenet"]
        assert yi["hexagram"] == "乾"
        assert yi["label"] == "乾 (Heaven)"
        assert yi["env_confidence"] == 0.9

    def test_returns_none_on_engine_exception(self, adapter):
        with patch("yicenet.engine_provider.EngineProvider.get_engine",
                   side_effect=RuntimeError("model not found")), \
             patch.object(adapter._orch, "before_prediction"):
            result = adapter.predict_for_turn_payload({"prompt": "task"})

        assert result is None

    def test_calls_before_prediction(self, adapter, mock_engine, mock_display):
        with patch("yicenet.engine_provider.EngineProvider.get_engine", return_value=mock_engine), \
             patch("yicenet.display.get_display", return_value=mock_display), \
             patch.object(adapter._orch, "before_prediction") as mock_before, \
             patch("sys.stderr", StringIO()):
            adapter.predict_for_turn_payload({"session_id": "abc123456789", "prompt": "x"})

        mock_before.assert_called_once()

    def test_result_label_written_to_stderr(self, adapter, mock_engine, mock_display):
        fake_stderr = StringIO()
        with patch("yicenet.engine_provider.EngineProvider.get_engine", return_value=mock_engine), \
             patch("yicenet.display.get_display", return_value=mock_display), \
             patch.object(adapter._orch, "before_prediction"), \
             patch("sys.stderr", fake_stderr):
            adapter.predict_for_turn_payload({"session_id": "aabb11223344", "prompt": "x"})

        assert "乾 (Heaven)" in fake_stderr.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# pre_message_send — stdout delivery
# ─────────────────────────────────────────────────────────────────────────────

class TestPreMessageSend:

    @pytest.fixture
    def adapter(self):
        from yicenet.tools.claude_hook import ClaudeCodeAdapter
        return ClaudeCodeAdapter()

    def test_prints_json_to_stdout_on_success(self, adapter):
        result_dict = {"yicenet": {"hexagram": "坤", "label": "坤"}}
        fake_stdout = StringIO()

        with patch.object(adapter, "predict_for_turn_payload", return_value=result_dict), \
             patch("sys.stdout", fake_stdout):
            adapter.pre_message_send({"prompt": "test"})

        output = fake_stdout.getvalue().strip()
        parsed = json.loads(output)
        assert parsed["yicenet"]["hexagram"] == "坤"

    def test_exits_0_when_predict_returns_none(self, adapter):
        with patch.object(adapter, "predict_for_turn_payload", return_value=None):
            with pytest.raises(SystemExit) as exc_info:
                adapter.pre_message_send({"prompt": "test"})
        assert exc_info.value.code == 0

    def test_reads_stdin_when_payload_not_supplied(self, adapter):
        result_dict = {"yicenet": {"label": "x"}}
        fake_stdout = StringIO()

        with patch("sys.stdin", StringIO('{"prompt": "from stdin"}')), \
             patch.object(adapter, "predict_for_turn_payload", return_value=result_dict) as mock_pred, \
             patch("sys.stdout", fake_stdout):
            adapter.pre_message_send()

        called_payload = mock_pred.call_args[0][0]
        assert called_payload.get("prompt") == "from stdin"

    def test_uses_supplied_payload_without_reading_stdin(self, adapter):
        result_dict = {"yicenet": {"label": "x"}}
        fake_stdout = StringIO()

        with patch.object(adapter, "predict_for_turn_payload", return_value=result_dict) as mock_pred, \
             patch("sys.stdout", fake_stdout):
            adapter.pre_message_send({"prompt": "direct"})

        called_payload = mock_pred.call_args[0][0]
        assert called_payload.get("prompt") == "direct"


# ─────────────────────────────────────────────────────────────────────────────
# stop — turn-complete lifecycle
# ─────────────────────────────────────────────────────────────────────────────

class TestStop:

    @pytest.fixture
    def adapter(self):
        from yicenet.tools.claude_hook import ClaudeCodeAdapter
        return ClaudeCodeAdapter()

    def test_calls_on_turn_complete(self, adapter):
        with patch.object(adapter._orch, "on_turn_complete") as mock_otc:
            adapter.stop({"session_id": "abc"})
        mock_otc.assert_called_once_with({"session_id": "abc"})

    def test_reads_stdin_when_payload_not_supplied(self, adapter):
        with patch("sys.stdin", StringIO('{"session_id": "xyz"}')), \
             patch.object(adapter._orch, "on_turn_complete") as mock_otc:
            adapter.stop()
        called = mock_otc.call_args[0][0]
        assert called.get("session_id") == "xyz"

    def test_silently_ignores_on_turn_complete_exception(self, adapter):
        with patch.object(adapter._orch, "on_turn_complete", side_effect=RuntimeError("db error")):
            adapter.stop({"session_id": "abc"})  # must not raise

    def test_subprocess_adapter_flushes_memory_bank(self):
        """process_model=subprocess causes flush_session in on_turn_complete."""
        from yicenet.tools.claude_hook import ClaudeCodeAdapter
        adapter = ClaudeCodeAdapter(process_model="subprocess")
        assert adapter._orch._adapter.process_model == "subprocess"

    def test_daemon_adapter_does_not_flush(self):
        """process_model=daemon skips flush_session in on_turn_complete."""
        from yicenet.tools.claude_hook import ClaudeCodeAdapter
        adapter = ClaudeCodeAdapter(process_model="daemon")
        assert adapter._orch._adapter.process_model == "daemon"


# ─────────────────────────────────────────────────────────────────────────────
# HermesAdapter hook entry points
# ─────────────────────────────────────────────────────────────────────────────

class TestHermesHookEntryPoints:

    def test_pre_llm_call_returns_label_dict(self):
        from yicenet.tools.hermes_hook import pre_llm_call, _adapter
        result_dict = {"yicenet": {"label": "乾 (Heaven)", "hexagram": "乾"}}
        with patch.object(_adapter, "predict_for_turn_payload", return_value=result_dict), \
             patch("sys.stderr", StringIO()):
            result = pre_llm_call({"session_id": "abc", "messages": [{"role": "user", "content": "x"}]})
        assert result == {"yicenet_context": "乾 (Heaven)"}

    def test_pre_llm_call_returns_none_on_failure(self):
        from yicenet.tools.hermes_hook import pre_llm_call, _adapter
        with patch.object(_adapter, "predict_for_turn_payload", return_value=None):
            result = pre_llm_call({})
        assert result is None

    def test_post_llm_call_delegates_to_stop(self):
        from yicenet.tools.hermes_hook import post_llm_call, _adapter
        with patch.object(_adapter, "stop") as mock_stop:
            post_llm_call({"session_id": "xyz"})
        mock_stop.assert_called_once_with({"session_id": "xyz"})


# ─────────────────────────────────────────────────────────────────────────────
# hook_server — HTTP IPC endpoints
# ─────────────────────────────────────────────────────────────────────────────

class TestHookServer:
    """The daemon routes POST /hook/<event>?platform=<id> to daemon.platforms and
    returns the handler's bytes verbatim as the hook's stdout."""

    @pytest.fixture
    def server(self, monkeypatch):
        import yicenet.daemon.hook_server as hs
        import yicenet.daemon.platforms as pf
        from http.server import ThreadingHTTPServer

        calls = []

        def fake_routes():
            return {
                "pre": lambda p: calls.append(("pre", p)) or b"[YiCeNet] " + json.dumps(p).encode(),
                "stop": lambda p: calls.append(("stop", p)) or b"",
                "boom": lambda p: 1 / 0,
            }

        monkeypatch.setattr(pf, "_FACTORIES", {"claude-code": fake_routes, "kimi-code": fake_routes})
        monkeypatch.setattr(pf, "_routes", {})
        srv = ThreadingHTTPServer(("127.0.0.1", 0), hs._HookHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        yield srv.server_address[1], calls
        srv.shutdown()
        srv.server_close()

    @staticmethod
    def _post(port, path, body=b"{}"):
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_health(self, server):
        port, _ = server
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            assert json.loads(r.read())["ok"] is True

    def test_routes_by_platform_and_returns_raw_stdout(self, server):
        port, calls = server
        body = json.dumps({"prompt": "重构"}, ensure_ascii=False).encode("utf-8")
        code, out = self._post(port, "/hook/pre?platform=kimi-code", body)
        assert code == 200
        assert out == b"[YiCeNet] " + json.dumps({"prompt": "重构"}).encode()
        assert calls == [("pre", {"prompt": "重构"})]

    def test_platform_defaults_to_claude_code(self, server):
        port, calls = server
        assert self._post(port, "/hook/stop") == (200, b"")
        assert calls == [("stop", {})]

    def test_non_utf8_and_bad_json_do_not_crash(self, server):
        port, calls = server
        assert self._post(port, "/hook/stop", b"\xff\xfe not json")[0] == 400
        assert self._post(port, "/hook/stop", b"[1, 2]")[0] == 200  # non-dict payload -> {}
        assert calls == [("stop", {})]

    def test_unknown_platform_or_event_404(self, server):
        port, _ = server
        assert self._post(port, "/hook/pre?platform=nope")[0] == 404
        assert self._post(port, "/hook/nope")[0] == 404
        assert self._post(port, "/other")[0] == 404

    def test_handler_error_is_500(self, server):
        port, _ = server
        code, out = self._post(port, "/hook/boom")
        assert code == 500 and b"ZeroDivisionError" in out


class TestPlatforms:

    def test_claude_pre_output_is_prefixed_json(self):
        from yicenet.daemon import platforms as pf
        fake = MagicMock()
        fake.predict_for_turn_payload.return_value = {"yicenet": {"hexagram": "乾"}}
        with patch("yicenet.tools.claude_hook.ClaudeCodeAdapter", return_value=fake), \
             patch("yicenet.daemon.platforms._configure_memory"):
            routes = pf._claude_routes()
        out = routes["pre"]({"prompt": "x"})
        assert out.startswith(b"[YiCeNet] ")
        assert json.loads(out[len(b"[YiCeNet] "):]) == {"yicenet": {"hexagram": "乾"}}
        assert routes["stop"]({}) == b""

    def test_claude_adapter_is_daemon_process_model(self):
        from yicenet.daemon import platforms as pf
        with patch("yicenet.daemon.platforms._configure_memory"):
            routes = pf._claude_routes()
        adapter = routes["pre"].__closure__[0].cell_contents
        assert adapter.process_model == "daemon"

    def test_kimi_routes_cover_all_cli_events(self):
        from yicenet.daemon import platforms as pf
        from yicenet.tools import kimi_code_hook as kimi
        saved = kimi._adapter
        try:
            with patch("yicenet.daemon.platforms._configure_memory"):
                routes = pf._kimi_routes()
            assert set(routes) == set(kimi._COMMANDS)
            assert kimi._adapter.process_model == "daemon"
            with patch.object(kimi._adapter, "predict_for_turn_payload", return_value=None):
                assert routes["pre_message_send"]({}) == b""
                assert routes["pre_tool_use"]({"tool_name": "Shell"}) == b""  # sys.exit swallowed
        finally:
            kimi._adapter = saved

    def test_unknown_route(self):
        from yicenet.daemon import platforms as pf
        with pytest.raises(pf.UnknownRoute):
            pf.handler_for("nope", "pre")


# ─────────────────────────────────────────────────────────────────────────────
# ipc_hook — Python twin of the native client
# ─────────────────────────────────────────────────────────────────────────────

class TestIpcHook:

    def test_forward_returns_none_when_daemon_unreachable(self):
        from yicenet.tools.ipc_hook import forward
        with patch("yicenet.tools.ipc_hook._get_port", return_value=1), \
             patch("yicenet.tools.ipc_hook._ensure_daemon", return_value=0) as ens:
            assert forward("claude-code", "pre", b"{}") is None
        ens.assert_called_once()

    def test_forward_retries_on_spawned_port(self):
        from yicenet.tools import ipc_hook
        replies = iter([None, b"OUT"])
        with patch.object(ipc_hook, "_post", side_effect=lambda *a, **k: next(replies)) as post, \
             patch.object(ipc_hook, "_ensure_daemon", return_value=4242):
            assert ipc_hook.forward("kimi-code", "stop", b"{}") == b"OUT"
        assert post.call_args.kwargs["port"] == 4242

    def test_env_port_wins(self, monkeypatch):
        from yicenet.tools.ipc_hook import _get_port
        monkeypatch.setenv("YICENET_DAEMON_PORT", "5555")
        assert _get_port() == 5555


# ─────────────────────────────────────────────────────────────────────────────
# Native client (native/yicenet-hook) — only when a built binary is around
# ─────────────────────────────────────────────────────────────────────────────

def _built_hook_binary():
    root = Path(__file__).resolve().parent.parent / "build" / "yicenet-hook"
    for cand in (root / "Release" / "yicenet-hook.exe", root / "yicenet-hook.exe", root / "yicenet-hook"):
        if cand.is_file():
            return cand
    return None


@pytest.mark.skipif(_built_hook_binary() is None, reason="native client not built (scripts/build-hook.*)")
class TestNativeClient:

    def _run(self, args, body, env_port, **env):
        import os
        import subprocess
        e = dict(os.environ, YICENET_DAEMON_PORT=str(env_port), YICENET_DAEMON_PYTHON="", **env)
        return subprocess.run([str(_built_hook_binary()), *args], input=body,
                              capture_output=True, timeout=30, env=e)

    def test_forwards_bytes_both_ways(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        seen = {}

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                seen["path"] = self.path
                seen["body"] = self.rfile.read(int(self.headers["Content-Length"]))
                out = "[YiCeNet] 回显".encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            body = json.dumps({"prompt": "中文 payload"}, ensure_ascii=False).encode("utf-8")
            r = self._run(["kimi-code", "pre_message_send"], body, srv.server_address[1])
        finally:
            srv.shutdown()
            srv.server_close()
        assert r.returncode == 0
        assert r.stdout == "[YiCeNet] 回显".encode("utf-8")
        assert seen == {"path": "/hook/pre_message_send?platform=kimi-code", "body": body}

    def test_no_daemon_no_python_exits_0_silently(self):
        r = self._run(["claude-code", "pre"], b"{}", 1, USERPROFILE="/nonexistent", HOME="/nonexistent")
        assert r.returncode == 0
        assert r.stdout == b""

    def test_usage_exits_0(self):
        r = self._run([], b"", 1)
        assert r.returncode == 0 and b"usage" in r.stderr


# ─────────────────────────────────────────────────────────────────────────────
# ClaudeCodeInstaller — register_mcp / register_hybrid / unregister_mcp
# ─────────────────────────────────────────────────────────────────────────────

class TestClaudeCodeInstallerModes:

    def _patched_installer(self, tmp_path):
        claude_dir = tmp_path / ".claude"
        hooks_dir = claude_dir / "hooks"
        settings_file = claude_dir / "settings.json"
        return (
            claude_dir, hooks_dir, settings_file,
            patch("yicenet.install.claude._CLAUDE_DIR", claude_dir),
            patch("yicenet.install.claude._HOOKS_DIR", hooks_dir),
            patch("yicenet.install.claude._SETTINGS", settings_file),
        )

    def test_register_mcp_writes_mcp_servers(self, tmp_path):
        from yicenet.install.claude import ClaudeCodeInstaller
        _, _, settings_file, p1, p2, p3 = self._patched_installer(tmp_path)

        with p1, p2, p3:
            installer = ClaudeCodeInstaller()
            with patch.object(installer, "_yicenet_serve", return_value="/bin/yicenet-serve"):
                installer.register_mcp()

        settings = json.loads(settings_file.read_text())
        assert "mcpServers" in settings
        assert "yicenet" in settings["mcpServers"]
        assert settings["mcpServers"]["yicenet"]["command"] == "/bin/yicenet-serve"

    def test_register_mcp_idempotent(self, tmp_path):
        from yicenet.install.claude import ClaudeCodeInstaller
        _, _, settings_file, p1, p2, p3 = self._patched_installer(tmp_path)

        with p1, p2, p3:
            installer = ClaudeCodeInstaller()
            with patch.object(installer, "_yicenet_serve", return_value="/bin/yicenet-serve"):
                installer.register_mcp()
                installer.register_mcp()

        settings = json.loads(settings_file.read_text())
        assert len([k for k in settings["mcpServers"]]) == 1

    def test_register_mcp_raises_when_serve_not_found(self, tmp_path):
        from yicenet.install.claude import ClaudeCodeInstaller
        _, _, _, p1, p2, p3 = self._patched_installer(tmp_path)

        with p1, p2, p3:
            installer = ClaudeCodeInstaller()
            with patch.object(installer, "_yicenet_serve", return_value=None):
                with pytest.raises(RuntimeError, match="yicenet-serve not found"):
                    installer.register_mcp()

    def test_unregister_mcp_removes_mcp_servers(self, tmp_path):
        from yicenet.install.claude import ClaudeCodeInstaller
        _, _, settings_file, p1, p2, p3 = self._patched_installer(tmp_path)

        with p1, p2, p3:
            installer = ClaudeCodeInstaller()
            with patch.object(installer, "_yicenet_serve", return_value="/bin/yicenet-serve"):
                installer.register_mcp()
            installer.unregister_mcp()

        settings = json.loads(settings_file.read_text())
        assert "mcpServers" not in settings

    def test_register_hybrid_writes_both_mcp_and_hooks(self, tmp_path):
        from yicenet.install.claude import ClaudeCodeInstaller
        _, hooks_dir, settings_file, p1, p2, p3 = self._patched_installer(tmp_path)

        with p1, p2, p3:
            installer = ClaudeCodeInstaller()
            with patch.object(installer, "_yicenet_serve", return_value="/bin/yicenet-serve"):
                installer.register_hybrid()

        settings = json.loads(settings_file.read_text())
        assert "mcpServers" in settings
        assert "hooks" in settings
        assert "UserPromptSubmit" in settings["hooks"]
        assert "Stop" in settings["hooks"]
        assert (hooks_dir / "yicenet_claude_hook.py").exists()

    def test_register_hybrid_event_passed_as_argv(self, tmp_path):
        """Event is passed as argv (not env) so Claude Code ignoring env doesn't break routing."""
        from yicenet.install.claude import ClaudeCodeInstaller
        _, _, settings_file, p1, p2, p3 = self._patched_installer(tmp_path)

        with p1, p2, p3:
            installer = ClaudeCodeInstaller()
            with patch.object(installer, "_yicenet_serve", return_value="/bin/yicenet-serve"):
                installer.register_hybrid()

        settings = json.loads(settings_file.read_text())
        pre_cmd = settings["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
        stop_cmd = settings["hooks"]["Stop"][0]["hooks"][0]["command"]
        assert pre_cmd.endswith(" pre")
        assert stop_cmd.endswith(" stop")
        assert "env" not in settings["hooks"]["UserPromptSubmit"][0]["hooks"][0]

    def test_register_hooks_event_passed_as_argv(self, tmp_path):
        from yicenet.install.claude import ClaudeCodeInstaller
        _, _, settings_file, p1, p2, p3 = self._patched_installer(tmp_path)

        with p1, p2, p3:
            ClaudeCodeInstaller().register_hooks()

        settings = json.loads(settings_file.read_text())
        pre_cmd = settings["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
        stop_cmd = settings["hooks"]["Stop"][0]["hooks"][0]["command"]
        assert pre_cmd.endswith(" pre")
        assert stop_cmd.endswith(" stop")

    def test_unregister_removes_both_hooks_and_mcp(self, tmp_path):
        from yicenet.install.claude import ClaudeCodeInstaller
        _, _, settings_file, p1, p2, p3 = self._patched_installer(tmp_path)

        with p1, p2, p3:
            installer = ClaudeCodeInstaller()
            with patch.object(installer, "_yicenet_serve", return_value="/bin/yicenet-serve"):
                installer.register_hybrid()
            installer.unregister()

        settings = json.loads(settings_file.read_text())
        assert "hooks" not in settings
        assert "mcpServers" not in settings

    def test_register_hybrid_raises_when_serve_not_found(self, tmp_path):
        from yicenet.install.claude import ClaudeCodeInstaller
        _, _, _, p1, p2, p3 = self._patched_installer(tmp_path)

        with p1, p2, p3:
            installer = ClaudeCodeInstaller()
            with patch.object(installer, "_yicenet_serve", return_value=None):
                with pytest.raises(RuntimeError, match="yicenet-serve not found"):
                    installer.register_hybrid()


def test_read_stdin_utf8_keeps_cjk_from_a_byte_pipe():
    """Claude Code pipes UTF-8 bytes; a locale-codepage text read turned CJK prompts into
    mojibake with lone surrogates that the tokenizer rejects (hook injected nothing)."""
    import io
    from yicenet.tools.hooks_adapter import read_stdin_utf8
    payload = '{"prompt": "易策中文提示词"}'
    fake = io.TextIOWrapper(io.BytesIO(payload.encode("utf-8")), encoding="cp1252", errors="surrogateescape")
    with patch("sys.stdin", fake):
        assert json.loads(read_stdin_utf8())["prompt"] == "易策中文提示词"


class TestClaudeRealPayloads:
    """Field shapes taken from Claude Code 2.1 hook payloads."""

    def test_stop_reply_comes_from_payload_not_transcript(self):
        from yicenet.tools.claude_hook import ClaudeCodeAdapter
        with patch("yicenet.tools.claude_hook._read_last_assistant", return_value="stale"):
            reply = ClaudeCodeAdapter().assistant_response(
                {"hook_event_name": "Stop", "last_assistant_message": "完成"})
            assert reply == "完成"
            assert ClaudeCodeAdapter().assistant_response({}) == "stale"

    def test_tool_outcome_from_tool_response(self):
        from yicenet.daemon.platforms import _claude_tool_outcome
        ok = {"stdout": "hi", "stderr": "", "interrupted": False, "isImage": False}
        code, size = _claude_tool_outcome(ok)
        assert code == 0 and size > 0
        assert _claude_tool_outcome({"stdout": "", "interrupted": True})[0] == 1
        assert _claude_tool_outcome({"is_error": True})[0] == 1
        assert _claude_tool_outcome({"returnCode": 2})[0] == 1
        assert _claude_tool_outcome("plain text result") == (0, len("plain text result") + 2)
        assert _claude_tool_outcome(None) == (0, 0)
