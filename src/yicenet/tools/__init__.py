"""YiCeNet hook implementations for Hermes and Claude Code."""

# Imported lazily (see __getattr__) so the IPC hook path stays stdlib-only.
_LAZY = {
    "HooksAdapter": ".hooks_adapter",
    "ClaudeCodeAdapter": ".claude_hook",
    "HermesAdapter": ".hermes_hook",
    "MCPAdapter": ".mcp_adapter",
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
