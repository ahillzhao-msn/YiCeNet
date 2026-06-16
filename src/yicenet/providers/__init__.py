"""
ProviderRegistry — component composition for YiCeNetEngine.

Default registry wires the production components.
override() replaces selected components for testing without touching the rest.

Slots:
    tokenizer — ITokenizer   (text encoding)
    memory    — IMemoryBank  (session turn history)
    display   — IDisplay     (prediction rendering + hook injection)

Usage:
    # Production
    registry = ProviderRegistry.default()
    rendered = registry.display.render(result)

    # Testing (mock display, real tokenizer + memory)
    registry = ProviderRegistry.override(display=SilentDisplay())
"""
from __future__ import annotations

from dataclasses import dataclass

from yicenet.interfaces import ITokenizer, IMemoryBank, IDisplay


@dataclass
class ProviderRegistry:
    """Holds one concrete implementation per injectable interface."""

    tokenizer: ITokenizer
    memory: IMemoryBank
    display: IDisplay

    @classmethod
    def default(cls) -> "ProviderRegistry":
        """Production registry: Qwen tokenizer + MemoryBank + config-driven display."""
        from yicenet.tokenizer import QwenTokenizerAdapter
        from yicenet.memory_bank import MemoryBank
        from yicenet.display import get_display
        return cls(
            tokenizer=QwenTokenizerAdapter(),
            memory=MemoryBank(),
            display=get_display(),
        )

    @classmethod
    def override(cls, **kwargs) -> "ProviderRegistry":
        """Test registry: swap selected slots, keep production defaults for the rest."""
        base = cls.default()
        for key, value in kwargs.items():
            if not hasattr(base, key):
                raise ValueError(f"Unknown registry slot: {key!r}")
            object.__setattr__(base, key, value)
        return base
