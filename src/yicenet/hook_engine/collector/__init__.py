"""collector — platform-independent context signal accumulation."""

from .interface import ContextCollector
from .types import SignalVector
from .daemon import DaemonContextCollector
from .subprocess import SubprocessContextCollector
from .explicit import ExplicitContextCollector

__all__ = [
    "ContextCollector",
    "SignalVector",
    "DaemonContextCollector",
    "SubprocessContextCollector",
    "ExplicitContextCollector",
]
