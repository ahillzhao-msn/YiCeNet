"""SignalVector — normalized R^27 turn-level environment vector."""

from __future__ import annotations

from typing import TypedDict


class SignalVector(TypedDict):
    # User input (2)
    tok_user_input_len: float
    tok_is_first_turn: float

    # API consumption (3)
    tok_prompt_tokens: float
    tok_completion_tokens: float
    tok_api_duration: float

    # Tool execution (6)
    tok_tool_count: float
    tok_tool_success_rate: float
    tok_tool_retry_count: float
    tok_tool_duration: float
    tok_tool_output_size: float
    tok_tool_diversity: float

    # Response properties (3)
    tok_response_len: float
    tok_has_code: float
    tok_code_block_count: float

    # Hexagram (3)
    tok_hex_conf: float
    tok_hex_q_gap: float
    tok_hex_entropy: float

    # Cross-turn timing (3)
    tok_user_speed: float
    tok_user_speed_ratio: float
    tok_mood_trend: float

    # Prior turn state (3)
    tok_drift_trend: float
    tok_is_prev_correction: float
    tok_is_prev_praise: float

    # Current-turn feedback (4)
    tok_user_satisfaction: float
    tok_is_correction: float
    tok_is_praise: float
    tok_is_abandon: float
