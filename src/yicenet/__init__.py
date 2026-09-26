"""
YiCeNet (易策网络) — I Ching inspired lightweight orchestration engine for Hermes.
~5.6M parameters (5,671,859), ~22 MB FP32, <3 ms inference.
"""

__version__ = "17.0.0"

# Public API, imported lazily (see __getattr__): name -> submodule
_LAZY = {
    **dict.fromkeys(["YiCeNetEngine", "get_engine", "predict"], ".yicenet_engine"),
    "EngineProvider": ".engine_provider",
    **dict.fromkeys(["YiCeNet", "count_parameters"], ".model"),
    **dict.fromkeys(["YiCeNetConfig", "yicenet_home", "yicenet_data_dir", "yicenet_checkpoint_dir"], ".config"),
    **dict.fromkeys(["format_prediction", "hexagram_symbol", "hexagram_judgment", "get_display",
                     "HEXAGRAM_NAMES", "HEXAGRAM_SYMBOLS", "TerminalDisplay", "JsonDisplay",
                     "SilentDisplay"], ".display"),
    **dict.fromkeys(["MemoryBank", "get_memory_bank", "TurnRecord"], ".memory_bank"),
    **dict.fromkeys(["CrossAttention", "ContextPrescription", "Prescription"], ".cross_attention"),
    **dict.fromkeys(["PredictionResult", "EnvAnalysis", "DisplayConfig"], ".types"),
    "IDisplay": ".interfaces",
    **dict.fromkeys(["install_tokenizer", "tokenizer_available"], ".tokenizer"),
}

__all__ = list(_LAZY)


def __getattr__(name):
    """Import public names on first use (PEP 562): `import yicenet.tools.ipc_hook` from a hook
    process must not pull in torch and the model just because the package is imported."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
