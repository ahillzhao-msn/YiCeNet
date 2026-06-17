"""
hook_engine — platform-independent hook lifecycle management.

Public surface:
    PlatformAdapter   — Protocol every platform adapter must implement
    HookOrchestrator  — Coordinates MemoryBank reads/writes + feedback submission
    FeedbackSignals   — Immutable result of feedback extraction
    extract_feedback  — Pure function: infer signals from next prompt + last turn
    signals_from_platform — Lift platform-provided signal dict to FeedbackSignals
    build_trajectory  — Assemble submit_trajectory() payload

Dependency rule: hook_engine → external_metrics, memory_bank, flywheel.
Never imports from tools/ or any platform-specific module.
"""
from .adapter import PlatformAdapter
from .extractor import FeedbackSignals, extract_feedback, signals_from_platform, build_trajectory
from .orchestrator import HookOrchestrator

__all__ = [
    "PlatformAdapter",
    "HookOrchestrator",
    "FeedbackSignals",
    "extract_feedback",
    "signals_from_platform",
    "build_trajectory",
]
