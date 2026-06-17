"""YiCeNet Hermes plugin — lifecycle hooks.

This file is the source of truth for what gets installed at
~/.hermes/plugins/yicenet-hooks/__init__.py by HermesInstaller.
Edit here; re-run install to deploy.
"""

def pre_llm_call(context):
    from yicenet.tools.hermes_hook import pre_llm_call as _fn
    return _fn(context)

def post_tool_call(context):
    from yicenet.tools.hermes_hook import post_tool_call as _fn
    return _fn(context)

def post_llm_call(context):
    from yicenet.tools.hermes_hook import post_llm_call as _fn
    return _fn(context)
