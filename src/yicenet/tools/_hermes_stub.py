"""YiCeNet Hermes plugin — lifecycle hooks.

This file is the source of truth for what gets installed at
~/.hermes/plugins/yicenet-hooks/__init__.py by HermesInstaller.
Edit here; re-run install to deploy.
"""

# Configure MemoryBank for daemon process model before any hook fires.
from yicenet.tools.hermes_hook import HermesAdapter
from yicenet.memory_bank import configure_memory_bank_for
configure_memory_bank_for(HermesAdapter())


def pre_llm_call(context):
    from yicenet.tools.hermes_hook import pre_llm_call as _fn
    return _fn(context)

def post_tool_call(context):
    from yicenet.tools.hermes_hook import post_tool_call as _fn
    return _fn(context)

def post_llm_call(context):
    from yicenet.tools.hermes_hook import post_llm_call as _fn
    return _fn(context)

def post_api_request(context):
    from yicenet.tools.hermes_hook import post_api_request as _fn
    return _fn(context)
