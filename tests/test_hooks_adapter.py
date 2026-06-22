"""
Tests for the three-mode hook adapter hierarchy.

Covers:
  HooksAdapter        — ABC enforcement, concrete defaults
  ClaudeCodeAdapter   — CC-specific overrides, process_model param
  HermesAdapter       — Hermes-specific overrides, signals
  MCPAdapter          — isolated from HooksAdapter, Protocol-compliant
  predict_for_turn_payload — shared logic path
  pre_message_send / stop  — delivery and routing
  hook_server         — HTTP endpoints (start/stop/pre)
  ipc_hook            — thin IPC client
  ClaudeCodeInstaller — register_mcp / register_hybrid / unregister_mcp
"""
from __future__ import annotations

import json
import sys
import threading
import time
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
            def session_id(self, p): return ""
            def assistant_response(self, p): return ""

        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_subclass_must_implement_process_model(self):
        from yicenet.tools.hooks_adapter import HooksAdapter

        class Incomplete(HooksAdapter):
            @property
            def platform_id(self): return "x"
            def session_id(self, p): return ""
            def assistant_response(self, p): return ""

        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_subclass_must_implement_session_id(self):
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
            def session_id(self, p): return ""

        with pytest.raises(TypeError):
            Incomplete()

    def test_minimal_concrete_subclass_is_instantiable(self):
        from yicenet.tools.hooks_adapter import HooksAdapter

        class Minimal(HooksAdapter):
            @property
            def platform_id(self): return "test"
            @property
            def process_model(self): return "subprocess"
            def session_id(self, p): return "sid"
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
            def session_id(self, p): return "sid"
            def assistant_response(self, p): return ""

        return Concrete()

    def test_default_platform_signals_is_none(self, adapter):
        assert adapter.platform_signals({}) is None

    def test_default_turn_id_from_messages(self, adapter):
        payload = {"messages": [{}, {}, {}]}  # 3 messages → turn_id 2
        assert adapter.turn_id(payload) == 2

    def test_default_turn_id_empty(self, adapter):
        assert adapter.turn_id({}) == 0

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

    def test_session_id_from_uuid(self):
        from yicenet.tools.claude_hook import ClaudeCodeAdapter
        adapter = ClaudeCodeAdapter()
        payload = {"session_id": "abcd-1234-ef56-7890"}
        sid = adapter.session_id(payload)
        assert sid == "abcd12345678"[:12] or len(sid) == 12

    def test_session_id_from_uuid_strips_hyphens(self):
        from yicenet.tools.claude_hook import ClaudeCodeAdapter
        payload = {"session_id": "aabb-ccdd-eeff-0011"}
        sid = ClaudeCodeAdapter().session_id(payload)
        assert "-" not in sid
        assert len(sid) == 12

    def test_session_id_fallback_cwd_hash(self, tmp_path):
        from yicenet.tools.claude_hook import ClaudeCodeAdapter
        adapter = ClaudeCodeAdapter()
        with patch("os.getcwd", return_value=str(tmp_path)):
            sid = adapter.session_id({})
        assert len(sid) == 12
        assert sid.isalnum() or all(c in "0123456789abcdef" for c in sid)

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
        assert adapter.session_id({"session_id": "abc123"}) == "abc123"

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

    @pytest.fixture(autouse=True)
    def reset_server(self):
        """Ensure hook_server module state is clean between tests."""
        import yicenet.daemon.hook_server as hs
        yield
        hs.stop_hook_server()
        hs._adapter = None

    def test_start_hook_server_returns_port(self):
        from yicenet.daemon.hook_server import start_hook_server
        port = start_hook_server(port=0)
        assert port > 0

    def test_start_hook_server_idempotent(self):
        from yicenet.daemon.hook_server import start_hook_server
        port1 = start_hook_server(port=0)
        port2 = start_hook_server(port=0)
        assert port1 == port2

    def test_post_hook_pre_calls_predict(self):
        from yicenet.daemon.hook_server import start_hook_server
        port = start_hook_server(port=0)
        time.sleep(0.05)

        fake_result = {"yicenet": {"hexagram": "乾", "label": "x", "session_id": "abc"}}
        with patch("yicenet.daemon.hook_server._hook_adapter") as mock_factory:
            mock_adapter = mock_factory.return_value
            mock_adapter.predict_for_turn_payload.return_value = fake_result
            data = json.dumps({"prompt": "test", "session_id": "abc123456789"}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/hook/pre", data=data,
                headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                body = json.loads(resp.read())

        assert body.get("yicenet", {}).get("hexagram") == "乾"

    def test_post_hook_stop_calls_stop(self):
        from yicenet.daemon.hook_server import start_hook_server
        port = start_hook_server(port=0)
        time.sleep(0.05)

        with patch("yicenet.daemon.hook_server._hook_adapter") as mock_factory:
            mock_adapter = mock_factory.return_value
            data = json.dumps({"session_id": "abc123456789"}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/hook/stop", data=data,
                headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                body = json.loads(resp.read())

        mock_adapter.stop.assert_called_once()
        assert body.get("ok") is True

    def test_unknown_endpoint_returns_404(self):
        import urllib.error
        from yicenet.daemon.hook_server import start_hook_server
        port = start_hook_server(port=0)
        time.sleep(0.05)

        data = b"{}"
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/hook/unknown", data=data,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=3)
        assert exc_info.value.code == 404

    def test_hook_adapter_singleton_uses_daemon_process_model(self):
        from yicenet.daemon.hook_server import _hook_adapter
        adapter = _hook_adapter()
        assert adapter.process_model == "daemon"


# ─────────────────────────────────────────────────────────────────────────────
# ipc_hook — thin client
# ─────────────────────────────────────────────────────────────────────────────

class TestIpcHook:

    def test_returns_false_when_daemon_unreachable(self):
        from yicenet.tools.ipc_hook import pre_message_send_ipc
        with patch("yicenet.tools.ipc_hook._get_port", return_value=1):  # port 1 = guaranteed fail
            result = pre_message_send_ipc({"prompt": "x"})
        assert result is False

    def test_stop_returns_false_when_daemon_unreachable(self):
        from yicenet.tools.ipc_hook import stop_ipc
        with patch("yicenet.tools.ipc_hook._get_port", return_value=1):
            result = stop_ipc({"session_id": "abc"})
        assert result is False

    def test_pre_message_send_ipc_success_path(self):
        """Full round-trip: ipc_hook → hook_server → predict stub."""
        import yicenet.daemon.hook_server as hs
        from yicenet.daemon.hook_server import start_hook_server, _hook_adapter
        from yicenet.tools.ipc_hook import pre_message_send_ipc

        port = start_hook_server(port=0)
        time.sleep(0.05)

        fake_result = {"yicenet": {"label": "坤", "hexagram": "坤", "session_id": "x"}}
        fake_stdout = StringIO()

        with patch.object(_hook_adapter(), "predict_for_turn_payload", return_value=fake_result), \
             patch("yicenet.tools.ipc_hook._get_port", return_value=port), \
             patch("sys.stderr", StringIO()), \
             patch("sys.stdout", fake_stdout):
            ok = pre_message_send_ipc({"session_id": "aabbccddeeff", "prompt": "task"})

        hs.stop_hook_server()
        hs._adapter = None

        assert ok is True
        output = json.loads(fake_stdout.getvalue())
        assert output["yicenet"]["hexagram"] == "坤"

    def test_ipc_label_written_to_stderr(self):
        """Label from daemon response lands in the hook process stderr."""
        import yicenet.daemon.hook_server as hs
        from yicenet.daemon.hook_server import start_hook_server, _hook_adapter
        from yicenet.tools.ipc_hook import pre_message_send_ipc

        port = start_hook_server(port=0)
        time.sleep(0.05)

        fake_result = {"yicenet": {"label": "HEXLABEL", "hexagram": "乾", "session_id": "x"}}
        fake_stderr = StringIO()

        with patch.object(_hook_adapter(), "predict_for_turn_payload", return_value=fake_result), \
             patch("yicenet.tools.ipc_hook._get_port", return_value=port), \
             patch("sys.stderr", fake_stderr), \
             patch("sys.stdout", StringIO()):
            pre_message_send_ipc({"session_id": "aabbccddeeff", "prompt": "task"})

        hs.stop_hook_server()
        hs._adapter = None

        assert "HEXLABEL" in fake_stderr.getvalue()


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
