"""
YiCeNet (æ˜“ç­–ç½‘ç»œ) â€” I Ching inspired lightweight orchestration engine for Hermes.
~5.6M parameters (5,671,859), ~22 MB FP32, <3 ms inference.
"""

__version__ = "17.0.0"

# Public API
from .yicenet_engine import YiCeNetEngine, get_engine, predict
from .engine_provider import EngineProvider
from .model import YiCeNet, count_parameters
from .config import YiCeNetConfig, yicenet_home, yicenet_data_dir, yicenet_checkpoint_dir
from .display import (
    format_prediction, hexagram_symbol, hexagram_judgment,
    get_display, HEXAGRAM_NAMES, HEXAGRAM_SYMBOLS,
    TerminalDisplay, JsonDisplay, SilentDisplay,
)
from .memory_bank import MemoryBank, get_memory_bank, TurnRecord
from .cross_attention import CrossAttention, ContextPrescription, Prescription
from .types import PredictionResult, EnvAnalysis, DisplayConfig
from .interfaces import IDisplay
from .tokenizer import download_tokenizer, tokenizer_available

__all__ = [
    "YiCeNetEngine", "get_engine", "predict",
    "EngineProvider",
    "YiCeNet", "count_parameters",
    "YiCeNetConfig", "yicenet_home", "yicenet_data_dir", "yicenet_checkpoint_dir",
    "format_prediction", "hexagram_symbol", "hexagram_judgment",
    "get_display", "HEXAGRAM_NAMES", "HEXAGRAM_SYMBOLS",
    "TerminalDisplay", "JsonDisplay", "SilentDisplay",
    "DisplayConfig", "IDisplay",
    "MemoryBank", "get_memory_bank", "TurnRecord",
    "CrossAttention", "ContextPrescription", "Prescription",
    "PredictionResult", "EnvAnalysis",
    "download_tokenizer", "tokenizer_available",
]

