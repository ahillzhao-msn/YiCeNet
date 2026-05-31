"""
World Model v2 — YiCeNet 雙頭冪律衰減世界模型。

架構變革：
  舊版：單頭(ℝ³²⁰→ℝ¹)，預測標量獎勵，所有卦象同一目標
  新版：雙頭共享底層，輸入六探針(ℝ⁹)+卦ID(ℝ⁶⁴)=ℝ⁷³
        頭A：結果卦象分布預測 (ℝ⁶⁴)，冪律 τ=30天
        頭B：外部向量預測 (ℝᴺ)，冪律 τ=3天

雙頭共享層讓短效波動中有用的信息反哺長效模式。
冪律遺忘曲線保證古老樣本永不歸零——只是越來越輕的耳語。
"""

import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def power_law_weight(
    timestamp: float,
    now: float,
    tau_days: float = 30.0,
    alpha: float = 1.5,
) -> float:
    """
    冪律遺忘曲線權重。

    w(t) = (1 + age_days / τ)^(-α)

    Args:
        timestamp: 樣本時間戳（Unix秒）
        now: 當前時間戳（Unix秒）
        tau_days: 特徵時間常數（天）
        alpha: 衰減指數

    Returns:
        float: 權重 [0, 1]
    """
    age_days = (now - timestamp) / 86400.0
    if age_days < 0:
        age_days = 0.0
    return (1.0 + age_days / tau_days) ** (-alpha)


def power_law_weight_batch(
    timestamps: torch.Tensor,
    now: float,
    tau_days: float = 30.0,
    alpha: float = 1.5,
) -> torch.Tensor:
    """批量冪律權重。"""
    age_days = (now - timestamps) / 86400.0
    age_days = age_days.clamp(min=0.0)
    return (1.0 + age_days / tau_days) ** (-alpha)


class WorldModelV2(nn.Module):
    """
    雙頭世界模型 v2。

    輸入：探針向量 ℝ⁹ + 卦象 one-hot ℝ⁶⁴ → ℝ⁷³
    頭A：結果卦象分布 ℝ⁶⁴（長效，τ_slow=30天）
    頭B：外部向量預測 ℝᴺ（短效，τ_fast=3天）

    參數量：~20K（共享層 73→128 + 頭A 128→64 + 頭B 128→N）
    """

    def __init__(
        self,
        probe_dim: int = 9,
        num_hexagrams: int = 64,
        shared_dim: int = 128,
        num_external_metrics: int = 3,
        slow_tau_days: float = 30.0,
        fast_tau_days: float = 3.0,
        alpha: float = 1.5,
        beta: float = 0.3,
    ):
        super().__init__()

        self.num_hexagrams = num_hexagrams
        self.num_external_metrics = num_external_metrics
        self.slow_tau_days = slow_tau_days
        self.fast_tau_days = fast_tau_days
        self.alpha = alpha
        self.beta = beta

        input_dim = probe_dim + num_hexagrams  # ℝ⁷³

        # ── 共享層 ──
        self.shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, shared_dim),
            nn.GELU(),
        )

        # ── 頭A：結果卦象分布 ──
        self.head_hexagram = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.GELU(),
            nn.Linear(shared_dim // 2, num_hexagrams),
        )

        # ── 頭B：外部向量 ──
        self.head_external = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.GELU(),
            nn.Linear(shared_dim // 2, num_external_metrics),
            nn.Sigmoid(),  # 约束输出到[0,1]，匹配 target_ext 范围
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.orthogonal_(p, gain=0.5)
            if "bias" in name:
                nn.init.zeros_(p)

    def forward(
        self,
        probes: torch.Tensor,
        hexagram_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            probes: (B, 9) 六探針向量
            hexagram_id: (B,) 選中卦象 ID

        Returns:
            hexagram_dist: (B, 64) 預測結果卦象分布（頭A）
            external_vec: (B, N) 預測外部向量（頭B）
        """
        B = probes.shape[0]

        # 卦象 one-hot
        hex_onehot = F.one_hot(hexagram_id, num_classes=self.num_hexagrams).float()  # (B, 64)

        # 拼接輸入
        x = torch.cat([probes, hex_onehot], dim=-1)  # (B, 73)

        # 共享層
        h = self.shared(x)  # (B, 128)

        # 頭A：結果卦象分布（softmax 輸出 = 概率分布）
        hex_logits = self.head_hexagram(h)  # (B, 64)
        hexagram_dist = F.softmax(hex_logits / 0.5, dim=-1)

        # 頭B：外部向量（sigmoid 輸出 = 0-1）
        external_vec = torch.sigmoid(self.head_external(h))  # (B, N)

        return hexagram_dist, external_vec

    def compute_weighted_loss(
        self,
        pred_hex_dist: torch.Tensor,
        target_hex_dist: torch.Tensor,
        pred_ext_vec: torch.Tensor,
        target_ext_vec: torch.Tensor,
        weights_slow: torch.Tensor,
        weights_fast: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        計算冪律加權的雙頭 loss。

        Args:
            pred_hex_dist: (B, 64) 預測結果卦象分布
            target_hex_dist: (B, 64) 實際結果卦象投影
            pred_ext_vec: (B, N) 預測外部向量
            target_ext_vec: (B, N) 實際外部向量
            weights_slow: (B,) 頭A冪律權重
            weights_fast: (B,) 頭B冪律權重

        Returns:
            loss_A: 頭A loss（標量）
            loss_B: 頭B loss（標量）
            total_loss: loss_A + β·loss_B
        """
        # 頭A：KL散度（預測分布 vs 目標分布）
        kl = (target_hex_dist * (target_hex_dist.clamp(min=1e-8).log() - pred_hex_dist.clamp(min=1e-8).log())).sum(dim=-1)
        loss_A = (weights_slow * kl).sum() / weights_slow.sum().clamp(min=1e-8)

        # 頭B：MSE（外部向量）
        se = (pred_ext_vec - target_ext_vec).pow(2).mean(dim=-1)
        loss_B = (weights_fast * se).sum() / weights_fast.sum().clamp(min=1e-8)

        total_loss = loss_A + self.beta * loss_B

        return loss_A, loss_B, total_loss

    @torch.no_grad()
    def predict_headA(
        self,
        probes: torch.Tensor,
        hexagram_id: torch.Tensor,
    ) -> torch.Tensor:
        """推理時：只走頭A（結果卦象分布）。返回 (B, 64)"""
        self.eval()
        hex_dist, _ = self.forward(probes, hexagram_id)
        return hex_dist

    @torch.no_grad()
    def predict_headB(
        self,
        probes: torch.Tensor,
        hexagram_id: torch.Tensor,
    ) -> torch.Tensor:
        """推理時：只走頭B（外部向量預測，參考信息）。返回 (B, N)"""
        self.eval()
        _, ext_vec = self.forward(probes, hexagram_id)
        return ext_vec

    def save(self, path: str):
        """Save checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "probe_dim": self.shared[1].in_features - self.num_hexagrams,
                "num_hexagrams": self.num_hexagrams,
                "shared_dim": self.shared[1].out_features,
                "num_external_metrics": self.num_external_metrics,
                "slow_tau_days": self.slow_tau_days,
                "fast_tau_days": self.fast_tau_days,
                "alpha": self.alpha,
                "beta": self.beta,
            },
        }, path)

    def compute_endogenous_weight(
        self,
        probes: torch.Tensor,
        hexagram_id: torch.Tensor,
        target_dist: torch.Tensor,
    ) -> torch.Tensor:
        """
        全內生權重：用 WM 預測與實際目標的 KL 驚訝度作為噪聲感知。
        
        原理：
        - WM 預測準確 (低KL) → 這條樣本符合學到的模式 → 乾淨信號 → 高權重
        - WM 預測失準 (高KL) → 這條偏離學到的模式 → 潛在噪聲 → 低權重
        
        Args:
            probes: (B, 9) 探針向量
            hexagram_id: (B,) 卦象 ID  
            target_dist: (B, 64) 目標卦象分布

        Returns:
            weight: (B,) 內生權重 [0, 1]
        """
        import torch.nn.functional as F
        
        if probes.dim() == 1:
            probes = probes.unsqueeze(0)
        if target_dist.dim() == 1:
            target_dist = target_dist.unsqueeze(0)
        
        with torch.no_grad():
            pred_dist, _ = self.forward(probes, hexagram_id)
            
            # KL 驚訝度
            kl = F.kl_div(
                pred_dist.clamp(min=1e-8).log(),
                target_dist.clamp(min=1e-8),
                reduction="none"
            ).sum(dim=-1)  # (B,)
            
            # 驚訝→權重轉換（sigmoid 曲線）
            # 低驚訝 (<0.02): weight ~1.0 → 強學習
            # 中驚訝 (0.02-0.1): weight ~0.5 → 部分學習
            # 高驚訝 (>0.1): weight ~0.1 → 近似忽略
            weight = 1.0 - torch.sigmoid((kl - 0.03) * 50)
        
        return weight

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "WorldModelV2":
        """從 checkpoint 載入。"""
        saved = torch.load(path, map_location=device, weights_only=False)
        cfg = saved["config"]
        model = cls(
            probe_dim=cfg.get("probe_dim", 9),
            num_hexagrams=cfg.get("num_hexagrams", 64),
            shared_dim=cfg.get("shared_dim", 128),
            num_external_metrics=cfg.get("num_external_metrics", 3),
            slow_tau_days=cfg.get("slow_tau_days", 30.0),
            fast_tau_days=cfg.get("fast_tau_days", 3.0),
            alpha=cfg.get("alpha", 1.5),
            beta=cfg.get("beta", 0.3),
        ).to(device)
        model.load_state_dict(saved["state_dict"])
        return model


# ── 前向兼容：舊版 WorldModel 的替身 ──
# 舊代碼調用 WorldModel.load() → 改成 WorldModelV2.load()
WorldModel = WorldModelV2
