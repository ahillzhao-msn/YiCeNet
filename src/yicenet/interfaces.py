"""
YiCeNet 抽象接口層 — 定義 CPU/GPU 自適應的抽象協議。

設計原則：
  - 公開方法（無前綴）：外部 API，子類不可重寫（final）
  - 保護方法（_前綴）：子類可重寫，供不同設備實現差異化邏輯
  - 私有方法（__前綴）：內部實現細節，不應被外部或子類訪問

當前僅有一個設備實現（GPU/CPU 自適應 via torch Tensors），
但接口設計預留多實現擴展空間。
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .types import PredictionResult, EnvAnalysis


# ── Phase 2: Core component interfaces ────────────────────────────────────────


class ITokenizer(ABC):
    """Replaceable tokenizer. Default: Qwen2.5-0.5B BPE."""

    @abstractmethod
    def encode(self, text: str, max_len: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids, attention_mask) on CPU."""

    @abstractmethod
    def get_vocab_size(self) -> int: ...

    @abstractmethod
    def download(self, hf_token: str = "") -> bool:
        """Download model files to local cache. Returns True on success."""


class IEncoder(ABC):
    """Replaceable encoder. Default: TinyEncoder (4-layer Transformer, 256-dim)."""

    @abstractmethod
    def encode_context(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor,
        env_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return (1, D) context embedding."""


class IRouter(ABC):
    """Replaceable router. Default: Gumbel-Softmax (τ: 1.0→0.1)."""

    @abstractmethod
    def divine(
        self, h: torch.Tensor, tau: float = 1.0, hard: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (hexagram_idx (1,), probs (1, 64))."""


class IValueNetwork(ABC):
    """Replaceable value network. Default: 3-layer MLP (256→128→64→1)."""

    @abstractmethod
    def score(self, candidate_embeds: torch.Tensor) -> torch.Tensor:
        """Return Q-values tensor (1, 8, 1)."""


class IMemoryBank(ABC):
    """Injectable short-term memory. Not a global singleton."""

    @abstractmethod
    def init_session(self, session_id: str) -> None: ...

    @abstractmethod
    def store_turn(
        self,
        session_id: str,
        turn_id: int,
        encoder_output: np.ndarray,
        hexagram_id: int,
        summary: str = "",
        timestamp: float = 0.0,
        metadata: "dict | None" = None,
    ) -> None: ...

    @abstractmethod
    def update_turn_metadata(
        self, session_id: str, turn_id: int, metadata: dict
    ) -> None: ...

    @abstractmethod
    def get_last_turn(self, session_id: str) -> "object | None": ...

    @abstractmethod
    def get_session_keys(
        self, session_id: str
    ) -> tuple[np.ndarray, list[dict]]: ...

    @abstractmethod
    def get_hexagram_history(self, session_id: str) -> list[int]: ...

    @abstractmethod
    def get_turn_count(self, session_id: str) -> int: ...


class IEngine(ABC):
    """Inference engine façade (not a God Class)."""

    @abstractmethod
    def predict(
        self,
        task_brief: str,
        temperature: float = 0.1,
        deterministic: bool = False,
        environment: Optional[dict] = None,
    ) -> "PredictionResult": ...

    @abstractmethod
    def analyze(
        self,
        task_brief: str,
        environment: Optional[dict] = None,
    ) -> "EnvAnalysis": ...

    @abstractmethod
    def switch_model(self, checkpoint: str) -> bool: ...


class IDisplay(ABC):
    """Pluggable display renderer. Injected via ProviderRegistry.display.

    Separates *what* to render (callers supply result + optional chain)
    from *how* to render (implementation decides mode, encoding, format).

    Implementations:
        TerminalDisplay — human-readable compact/detailed for terminals/prompts
        JsonDisplay     — structured JSON for machine consumers (hook injection)
        SilentDisplay   — no-op for testing
    """

    @property
    def needs_chain(self) -> bool:
        """True if this renderer will use chain history passed to render()."""
        return False

    @abstractmethod
    def render(
        self,
        result: "PredictionResult",
        chain: "list[int] | None" = None,
    ) -> str:
        """Render a prediction result to string.

        chain: optional hexagram_ids (0-indexed) from MemoryBank.get_hexagram_history().
               Implementations may ignore it (JsonDisplay, SilentDisplay).
               TerminalDisplay uses it only when hexagram_chain=True in config.
        """

    @abstractmethod
    def render_chain(self, hexagram_ids: list[int]) -> str:
        """Render chain history standalone.

        hexagram_ids: 0-indexed list from MemoryBank.get_hexagram_history().
        Returns a string representation of the chain (e.g. "乾→屯→恒").
        """


# ── Existing interface (unchanged) ────────────────────────────────────────────


class ProbeExtractor(ABC):
    """
    探針提取器抽象接口。

    六探針從模型前向傳播的中間狀態中提取 ℝ⁹ 向量。
    不同設備實現可自定義提取策略（例如 GPU 用 CUDA stream 非同步），
    但返回格式一致。

    使用方式：
        extractor = ProbeExtractor.create()
        probe_tensor = extractor.extract(h, router_logits, ...)  # (9,)
    """

    @abstractmethod
    def _extract_impl(
        self,
        h: torch.Tensor,
        router_logits: torch.Tensor,
        router_probs: torch.Tensor,
        candidate_values: torch.Tensor,
        hexagram_idx: torch.Tensor,
        prev_hexagram_idx: Optional[torch.Tensor],
        action_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        子類實現：從模型中間狀態提取 ℝ⁹ 探針向量。

        Args:
            h: (1, D) encoder 輸出
            router_logits: (1, 64) router 原始 logits
            router_probs: (1, 64) router softmax 概率
            candidate_values: (1, 8, 1) 8 候選 Q 值
            hexagram_idx: (1,) 選中卦 ID
            prev_hexagram_idx: (1,) or None 上輪卦 ID
            action_logits: (1, num_actions) action decoder 輸出

        Returns:
            probe_vec: (9,) float32 tensor — 六探針值
        """
        ...

    # ── 公開方法（final，子類不可重寫）──

    def extract(
        self,
        h: torch.Tensor,
        router_logits: torch.Tensor,
        router_probs: torch.Tensor,
        candidate_values: torch.Tensor,
        hexagram_idx: torch.Tensor,
        prev_hexagram_idx: Optional[torch.Tensor],
        action_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        提取六探針並返回 ℝ⁹ float32 tensor。

        Returns:
            (9,) tensor: [h_norm, h_entropy, logit_entropy,
                          clan_upper, clan_lower, clan_opposite,
                          q_gap, jump_distance, action_confidence]
        """
        result = self._extract_impl(
            h, router_logits, router_probs,
            candidate_values, hexagram_idx,
            prev_hexagram_idx, action_logits,
        )
        # 裁剪到合理範圍
        result[0] = result[0].clamp(0.0, 10.0)   # h_norm
        result[1] = result[1].clamp(min=0.0)      # h_entropy
        result[2] = result[2].clamp(min=0.0)      # logit_entropy
        result[6] = result[6].clamp(min=0.0)      # q_gap
        return result

    # ── 工廠方法 ──

    @staticmethod
    def create() -> "ProbeExtractor":
        """創建默認探針提取器（GPU/CPU 自適應）。"""
        from .probes import _ProbeExtractorImpl
        return _ProbeExtractorImpl()
