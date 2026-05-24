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
from typing import Optional, Tuple

import torch


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
