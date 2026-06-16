"""
ProviderRegistry — component composition for YiCeNetEngine.

Default registry wires the production components.
Override() replaces selected components for testing without touching the rest.

Usage:
    # Production
    registry = ProviderRegistry.default()
    engine = YiCeNetEngine(model, registry.tokenizer, registry.memory)

    # Testing (mock tokenizer + memory, real model)
    registry = ProviderRegistry.override(
        tokenizer=MockTokenizer(),
        memory=MockMemoryBank(),
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from yicenet.interfaces import ITokenizer, IMemoryBank


@dataclass
class ProviderRegistry:
    """Holds one concrete implementation per injectable interface."""

    tokenizer: ITokenizer
    memory: IMemoryBank

    @classmethod
    def default(cls) -> "ProviderRegistry":
        """Production registry: Qwen tokenizer + in-process MemoryBank."""
        from yicenet.tokenizer import QwenTokenizerAdapter
        from yicenet.memory_bank import MemoryBank
        return cls(
            tokenizer=QwenTokenizerAdapter(),
            memory=MemoryBank(),
        )

    @classmethod
    def override(cls, **kwargs) -> "ProviderRegistry":
        """Test registry: swap selected components, keep defaults for the rest."""
        base = cls.default()
        for key, value in kwargs.items():
            if not hasattr(base, key):
                raise ValueError(f"Unknown registry slot: {key!r}")
            object.__setattr__(base, key, value)
        return base
