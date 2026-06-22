"""YiCeNet Hermes plugin — lifecycle hooks.

This file is the source of truth for what gets installed at
~/.hermes/plugins/yicenet-hooks/__init__.py by HermesInstaller.
Edit here; re-run install to deploy.

Hermes calls hook callbacks as cb(**kwargs), not cb(context).
The stub converts **kw → dict before delegating to hermes_hook.
"""

from __future__ import annotations

from typing import Any

# Configure MemoryBank for daemon process model before any hook fires.
from yicenet.tools.hermes_hook import HermesAdapter
from yicenet.memory_bank import configure_memory_bank_for
configure_memory_bank_for(HermesAdapter())


def pre_llm_call(**kw: Any) -> dict | str | None:
    from yicenet.tools.hermes_hook import pre_llm_call as _fn
    return _fn(kw)


def post_tool_call(**kw: Any) -> None:
    from yicenet.tools.hermes_hook import post_tool_call as _fn
    _fn(kw)


def post_llm_call(**kw: Any) -> None:
    from yicenet.tools.hermes_hook import post_llm_call as _fn
    _fn(kw)


def post_api_request(**kw: Any) -> None:
    from yicenet.tools.hermes_hook import post_api_request as _fn
    _fn(kw)


# ── Plugin Registration ──────────────────────────────────────


def register(ctx: Any) -> None:
    """Register all YiCeNet lifecycle hooks."""
    ctx.register_hook("pre_llm_call", pre_llm_call)
    ctx.register_hook("post_tool_call", post_tool_call)
    ctx.register_hook("post_api_request", post_api_request)
    ctx.register_hook("post_llm_call", post_llm_call)
