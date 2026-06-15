"""Tests for datasource adapters and env_context utilities.

Relies on pytest fixtures for temporary files to avoid polluting real
Hermes/Claude Code data stores.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import math
from pathlib import Path

import pytest
import torch

from yicenet.env_context import (
    build_env_vec, compute_env_confidence,
    ENV_DIM,
)
from yicenet.datasource import DataSource, Sample
from yicenet.datasource.buffer import FlywheelBufferSource
from yicenet.datasource.hermes import HermesDataSource


# ── env_context tests ────────────────────────────────────────────────────────


class TestBuildEnvVec:

    def test_empty_context_returns_none(self):
        """Empty or None context should return None (no-op in projector)."""
        assert build_env_vec(None) is None
        assert build_env_vec({}) is None

    def test_partial_context_uses_defaults(self):
        """Missing keys default to 0 (or 0.5 for satisfaction)."""
        v = build_env_vec({"hour_of_day": 12})
        assert v is not None
        assert v.shape == (ENV_DIM,)
        assert isinstance(v, torch.Tensor)
        assert v.dtype == torch.float32
        # slot 2 (session_turn) defaults to 0
        assert v[2].item() == 0.0

    def test_time_phase_encoding(self):
        """Sine/cosine hour encoding should be periodic."""
        v_midnight = build_env_vec({"hour_of_day": 0})
        v_noon = build_env_vec({"hour_of_day": 12})
        v_midnight2 = build_env_vec({"hour_of_day": 24})
        assert v_midnight is not None and v_noon is not None and v_midnight2 is not None
        # sin(0) ≈ 0, sin(π) ≈ 0
        assert abs(v_midnight[0].item()) < 0.05
        assert abs(v_noon[0].item()) < 0.05
        # cos(0) ≈ 1, cos(π) ≈ -1
        assert abs(v_midnight[1].item() - 1.0) < 0.05
        assert abs(v_noon[1].item() - (-1.0)) < 0.05
        # 24h wraps to same as 0h
        assert abs(v_midnight[0].item() - v_midnight2[0].item()) < 0.01
        assert abs(v_midnight[1].item() - v_midnight2[1].item()) < 0.01

    def test_session_turn_clamping(self):
        """Session turn should clamp at 50 (slot[4] in 16-dim layout)."""
        v1 = build_env_vec({"session_turn": 100})
        v2 = build_env_vec({"session_turn": 50})
        assert v1 is not None and v2 is not None
        assert v1[4].item() == 1.0, f"expected 1.0, got {v1[4].item()}"
        assert v2[4].item() == 1.0

    def test_last_hexagram_id_default(self):
        """Unknown hexagram ID defaults to 0.0 (slot[6] in 16-dim)."""
        v = build_env_vec({"session_turn": 5})
        assert v is not None
        assert v[6].item() == 0.0  # last_hexagram_id

    def test_last_tool_success_overrides_satisfaction(self):
        """last_tool_success=True should set slot[11]=1.0 in 16-dim layout."""
        # Build env with last_tool_success=True
        v_ok = build_env_vec({"last_tool_success": True})
        v_fail = build_env_vec({"last_tool_success": False})
        v_none = build_env_vec({})
        assert v_ok is not None and v_fail is not None
        # slot 11 = last_tool_success_bin
        assert v_ok[11].item() == 1.0, f"expected 1.0, got {v_ok[11].item()}"
        assert v_fail[11].item() == 0.0, f"expected 0.0, got {v_fail[11].item()}"

    def test_full_context_all_slots_defined(self):
        """All 16 slots should produce valid values when fully specified."""
        ctx = {
            "hour_of_day": 14,
            "day_of_week": 2,
            "session_turn": 7,
            "last_hexagram_id": 35,
            "correction_rate": 0.2,
            "satisfaction_ema": 0.8,
            "attention_entropy": 2.0,
            "last_tool_success": True,
        }
        v = build_env_vec(ctx)
        assert v is not None
        assert v.shape == (ENV_DIM,)
        # Time phase slots [0-3] are bipolar [-1, 1]; remaining [4-15] are in [0, 1]
        for i in range(4):
            assert -1.0 <= v[i].item() <= 1.0, f"slot[{i}] out of range: {v[i].item()}"
        for i in range(4, ENV_DIM):
            assert 0.0 <= v[i].item() <= 1.0, f"slot[{i}] out of range: {v[i].item()}"
        # Time and session fields should be non-zero with full context
        assert v[0].item() != 0.0 or v[1].item() != 0.0  # sin/cos hour


class TestComputeEnvConfidence:

    def test_empty_probe_returns_partial(self):
        """No probe data → partial confidence with default values."""
        conf, status, hint = compute_env_confidence(None, None)
        assert 0.0 <= conf <= 1.0
        assert status == "partial"

    def test_high_confidence_from_low_entropy_large_gap(self):
        """Low logit entropy + large Q-gap → 'sufficient'."""
        # probe_list[2] = logit_entropy (low = focused), probe_list[6] = q_gap (large = decisive)
        probes = [0.5, 0.3, 0.5, 0.2, 0.3, 0.4, 3.0, 0.1, 0.8]
        q_vals = [0.9, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.0]
        conf, status, hint = compute_env_confidence(probes, q_vals)
        assert status in ("sufficient", "partial"), f"expected sufficient/partial, got {status}"
        assert hint == "", f"expected empty hint, got {hint}"

    def test_low_confidence_from_high_entropy_small_gap(self):
        """High entropy + tiny Q-gap → 'thin' with signal suggestion."""
        probes = [0.5, 0.3, 3.5, 0.2, 0.3, 0.4, 0.05, 4.0, 0.8]
        q_vals = [0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08]
        conf, status, hint = compute_env_confidence(probes, q_vals)
        assert status == "thin", f"expected thin, got {status}"
        assert "session_turn" in hint, f"hint should suggest signals, got: {hint}"


# ── FlywheelBufferSource tests ──────────────────────────────────────────────


class TestFlywheelBufferSource:

    def test_no_file_returns_empty(self):
        """Source should return empty list when buffer file doesn't exist."""
        src = FlywheelBufferSource(buffer_path=Path("/nonexistent/buffer.jsonl"))
        samples = src.scan_since(0.0)
        assert samples == []

    def test_reads_single_trajectory(self, tmp_path):
        """A single valid JSONL line should produce one Sample."""
        buf = tmp_path / "flywheel_buffer.jsonl"
        record = {
            "producer": "test",
            "conversation_id": "sess1",
            "user_text": "hello",
            "timestamp": 1000.0,
            "satisfaction": 0.8,
            "token_cost": 123,
            "continued": True,
            "corrected": False,
            "completed": False,
            "praised": False,
            "abandoned": False,
            "token_efficiency": 0.5,
        }
        buf.write_text(json.dumps(record) + "\n", encoding="utf-8")

        src = FlywheelBufferSource(buffer_path=buf)
        samples = src.scan_since(0.0)
        assert len(samples) == 1
        assert samples[0].conversation_id == "test:sess1"
        assert samples[0].source == "buffer"
        assert samples[0].satisfaction == 0.8
        assert samples[0].continued is True

    def test_incremental_scan_respects_offset(self, tmp_path):
        """After reading once, the same entries should not be re-read."""
        buf = tmp_path / "flywheel_buffer.jsonl"
        line1 = json.dumps({"producer": "t", "conversation_id": "s1",
                            "user_text": "a", "timestamp": 100.0,
                            "continued": True, "corrected": False,
                            "completed": False, "praised": False,
                            "abandoned": False}) + "\n"
        line2 = json.dumps({"producer": "t", "conversation_id": "s2",
                            "user_text": "b", "timestamp": 200.0,
                            "continued": True, "corrected": False,
                            "completed": False, "praised": False,
                            "abandoned": False}) + "\n"
        buf.write_text(line1 + line2, encoding="utf-8")

        src = FlywheelBufferSource(buffer_path=buf)
        first = src.scan_since(0.0)
        assert len(first) == 2

        # Second scan should return 0 (offset has advanced)
        second = src.scan_since(0.0)
        assert len(second) == 0

        # After appending a new line, only that line should be read
        with buf.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"producer": "t", "conversation_id": "s3",
                                "user_text": "c", "timestamp": 300.0,
                                "continued": True, "corrected": False,
                                "completed": False, "praised": False,
                                "abandoned": False}) + "\n")
        third = src.scan_since(0.0)
        assert len(third) == 1
        assert third[0].conversation_id == "t:s3"

    def test_reset_offset_after_truncation(self, tmp_path):
        """When the file shrinks, offset should reset to 0."""
        buf = tmp_path / "flywheel_buffer.jsonl"
        line1 = json.dumps({"producer": "t", "conversation_id": "s1",
                            "user_text": "a very long text here to ensure file sizes differ",
                            "timestamp": 100.0,
                            "continued": True, "corrected": False,
                            "completed": False, "praised": False,
                            "abandoned": False}) + "\n"
        buf.write_text(line1, encoding="utf-8")

        src = FlywheelBufferSource(buffer_path=buf)
        src.scan_since(0.0)  # advances offset

        # Truncate and write a shorter file
        buf.write_text(json.dumps({"producer": "t", "conversation_id": "s2",
                                   "user_text": "short",
                                   "timestamp": 200.0,
                                   "continued": True, "corrected": False,
                                   "completed": False, "praised": False,
                                   "abandoned": False}) + "\n",
                       encoding="utf-8")

        samples = src.scan_since(0.0)
        assert len(samples) == 1
        assert samples[0].conversation_id == "t:s2"


# ── HermesDataSource tests (mock DB) ────────────────────────────────────────


class TestHermesDataSource:

    def test_unavailable_db_returns_empty(self):
        """No Hermes state.db → is_available() returns False."""
        src = HermesDataSource(db_path=Path("/nonexistent/state.db"))
        assert not src.is_available()
        assert src.scan_since(0.0) == []

    def test_source_id(self):
        """source_id should be 'hermes'."""
        src = HermesDataSource(db_path=Path("/nonexistent/state.db"))
        assert src.source_id == "hermes"


# ── Sample dataclass tests ──────────────────────────────────────────────────


class TestSampleDataclass:

    def test_default_fields(self):
        """Sample should have sensible defaults."""
        s = Sample(conversation_id="test:1", source="test",
                   user_text="hello", assistant_text="hi")
        assert s.conversation_id == "test:1"
        assert s.source == "test"
        assert s.continued is False  # default
        assert s.corrected is False
        assert s.embedding == []

    def test_conversation_id_prefix_pattern(self):
        """conversation_id should include source prefix."""
        s = Sample(conversation_id="hermes:abc123", source="hermes",
                   user_text="hello", assistant_text="hi")
        assert s.conversation_id == "hermes:abc123"
        assert s.conversation_id.startswith("hermes:")
