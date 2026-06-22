"""DaemonContextCollector — in-process signal accumulator (full 27-dim)."""

from __future__ import annotations

import time
from typing import Optional

from .interface import ContextCollector


class DaemonContextCollector(ContextCollector):
    """In-process accumulator for daemon-mode platforms (Hermes, CC daemon).

    Created at turn start, garbage-collected at turn end.
    All signal computation logic lives here; SubprocessContextCollector
    replays file events into a temporary instance of this class.
    """

    def __init__(self) -> None:
        self._user_text: str = ""
        self._is_first_turn: bool = False

        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._api_duration_ms: float = 0.0

        self._tools: list[dict] = []
        self._response_text: str = ""
        self._response_has_code: bool = False
        self._code_block_count: int = 0

        self._hexagram_q_max: float = 0.0
        self._hexagram_q_gap: float = 0.0
        self._hexagram_entropy: float = 0.0

        self._turn_timestamp: float = 0.0

        self._user_interval_sec: Optional[float] = None
        self._prev_metadata: Optional[dict] = None

        self._satisfaction: Optional[float] = None
        self._is_correction_cached: Optional[bool] = None
        self._is_praise_cached: Optional[bool] = None

    # -- sniff methods --------------------------------------------------------

    def sniff_user(self, text: str, is_first_turn: bool = False) -> None:
        self._user_text = text
        self._is_first_turn = is_first_turn
        if not self._turn_timestamp:
            self._turn_timestamp = time.time()

    def sniff_api(self, prompt_tokens: int, completion_tokens: int,
                  duration_ms: float = 0.0) -> None:
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._api_duration_ms = duration_ms

    def sniff_tool(self, name: str, exit_code: int,
                   duration_ms: float, result_size_bytes: int = 0) -> None:
        self._tools.append({
            "name": name,
            "exit_code": exit_code,
            "duration_ms": duration_ms,
            "result_size": result_size_bytes,
        })

    def sniff_response(self, text: str) -> None:
        self._response_text = text
        self._response_has_code = "```" in text or "~~~" in text
        cnt = text.count("```") // 2
        if cnt == 0:
            cnt = text.count("~~~") // 2
        self._code_block_count = cnt

    def sniff_hexagram(self, q_max: float, q_gap: float,
                       entropy: float) -> None:
        self._hexagram_q_max = q_max
        self._hexagram_q_gap = q_gap
        self._hexagram_entropy = entropy

    def sniff_timing(self, user_interval_sec: Optional[float] = None,
                     prev_metadata: Optional[dict] = None) -> None:
        self._user_interval_sec = user_interval_sec
        self._prev_metadata = prev_metadata

    # -- build ----------------------------------------------------------------

    def build_vector(self, prev_metadata: Optional[dict] = None) -> dict:
        prev = prev_metadata or self._prev_metadata or {}

        n_tools = len(self._tools)
        success_count = sum(1 for t in self._tools if t["exit_code"] == 0)
        retry_count = self._count_retries()
        tool_duration_total = sum(t["duration_ms"] for t in self._tools)
        tool_output_total = sum(t["result_size"] for t in self._tools)
        unique_tools = len(set(t["name"] for t in self._tools))

        current_sat = self._compute_satisfaction(n_tools, success_count)

        prev_sat = prev.get("tok_user_satisfaction", 0.0)
        mood_trend = current_sat - prev_sat

        prev_speed = prev.get("tok_user_speed", 0.3)
        if self._user_interval_sec is not None:
            current_speed = min(self._user_interval_sec / 60.0, 1.0)
            speed_ratio = current_speed / max(prev_speed, 0.05)
        else:
            current_speed = 0.3
            speed_ratio = 1.0

        prev_q_max = prev.get("tok_hex_conf", 0.0)
        drift_trend = self._hexagram_q_max - prev_q_max

        is_prev_correction = float(prev.get("tok_is_correction", 0.0))
        is_prev_praise = float(prev.get("tok_is_praise", 0.0))

        user_input_len = min(len(self._user_text) / 512, 1.0)

        return {
            "tok_user_input_len":       user_input_len,
            "tok_is_first_turn":        float(self._is_first_turn),

            "tok_prompt_tokens":        min(self._prompt_tokens / 4096, 1.0),
            "tok_completion_tokens":    min(self._completion_tokens / 4096, 1.0),
            "tok_api_duration":         min(self._api_duration_ms / 10000, 1.0),

            "tok_tool_count":           min(n_tools / 10, 1.0),
            "tok_tool_success_rate":    success_count / max(n_tools, 1),
            "tok_tool_retry_count":     min(retry_count / 5, 1.0),
            "tok_tool_duration":        min(tool_duration_total / 30000, 1.0),
            "tok_tool_output_size":     min(tool_output_total / 1_000_000, 1.0),
            "tok_tool_diversity":       min(unique_tools / 8, 1.0),

            "tok_response_len":         min(len(self._response_text) / 4000, 1.0),
            "tok_has_code":             float(self._response_has_code),
            "tok_code_block_count":     min(self._code_block_count / 10, 1.0),

            "tok_hex_conf":             float(self._hexagram_q_max),
            "tok_hex_q_gap":            float(self._hexagram_q_gap),
            "tok_hex_entropy":          min(max(self._hexagram_entropy / 4.0, 0.0), 1.0),

            "tok_user_speed":           current_speed,
            "tok_user_speed_ratio":     min(speed_ratio, 2.0),
            "tok_mood_trend":           max(-1.0, min(1.0, mood_trend)),

            "tok_drift_trend":          max(-1.0, min(1.0, drift_trend)),
            "tok_is_prev_correction":   is_prev_correction,
            "tok_is_prev_praise":       is_prev_praise,

            "tok_user_satisfaction":    current_sat,
            "tok_is_correction":        float(self._is_correction()),
            "tok_is_praise":            float(self._is_praise()),
            "tok_is_abandon":           float(self._is_abandon()),
        }

    # -- internal -------------------------------------------------------------

    def _count_retries(self) -> int:
        if not self._tools:
            return 0
        retries = 0
        for i in range(1, len(self._tools)):
            prev_t = self._tools[i - 1]
            curr_t = self._tools[i]
            if curr_t["name"] == prev_t["name"] and curr_t["exit_code"] != 0:
                retries += 1
        return retries

    def _compute_satisfaction(self, n_tools: int, success_count: int) -> float:
        has_code = self._response_has_code
        success_rate = success_count / max(n_tools, 1)
        user_engaged = min(len(self._user_text) / 200, 1.0) if self._user_text else 0.0
        interval_urgent = 1.0 - min((self._user_interval_sec or 30) / 30, 1.0)

        score = 0.0
        score += 0.20 * float(has_code)
        score += 0.30 * success_rate
        score += 0.20 * user_engaged
        score += 0.10 * interval_urgent
        score += 0.20 * float(self._hexagram_q_max)

        if self._is_correction():
            score -= 0.5
        return max(-1.0, min(1.0, score))

    def _is_correction(self) -> bool:
        if self._is_correction_cached is not None:
            return self._is_correction_cached
        if not self._user_text:
            self._is_correction_cached = False
            return False
        from yicenet.external_metrics import _check_patterns, _CORRECTION_PATTERNS
        self._is_correction_cached = _check_patterns(self._user_text, _CORRECTION_PATTERNS)
        return self._is_correction_cached

    def _is_praise(self) -> bool:
        if self._is_praise_cached is not None:
            return self._is_praise_cached
        if not self._user_text:
            self._is_praise_cached = False
            return False
        from yicenet.external_metrics import _check_patterns, _PRAISE_PATTERNS
        self._is_praise_cached = _check_patterns(self._user_text, _PRAISE_PATTERNS)
        return self._is_praise_cached

    def _is_abandon(self) -> bool:
        if not self._user_text:
            return True
        from yicenet.external_metrics import _check_patterns, _ABANDON_PATTERNS
        return _check_patterns(self._user_text, _ABANDON_PATTERNS)

    def __repr__(self) -> str:
        n_tools = len(self._tools)
        return (f"<DaemonContextCollector tools={n_tools} "
                f"api=({self._prompt_tokens}+{self._completion_tokens}) "
                f"has_code={self._response_has_code} "
                f"hex_q={self._hexagram_q_max:.2f}>")
