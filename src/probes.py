"""
六探針系統 — YiCeNet v5 內部狀態傳感器。

Design:
  - ProbeExtractor 抽象接口定義在 interfaces.py
  - _ProbeExtractorImpl 是唯一實現（GPU/CPU 自適應）
  - 所有硬編碼常數委託給 constants.PrecomputedHexagramTables
  - 返回 ℝ⁹ float32 tensor，調用端自行 .tolist()
  - 私有方法用 __ 前綴標記內部實現細節

探針列表（ℝ⁹）：
  ① h密度       — encoder 輸出範數 + 激活分布熵     [2]
  ② Logits形狀   — router 64-logits 的熵              [1]
  ③ 卦象家族     — 上下卦 + 錯卦歸屬                   [3]
  ④ Q值差距     — best Q - 2nd best Q                 [1]
  ⑤ 跳躍度      — 本輪卦 vs 上輪卦的卦象空間距離     [1]
  ⑥ 動作置信    — action decoder 輸出的負熵           [1]
"""

from typing import Optional

import torch
import torch.nn.functional as F

from .constants import PrecomputedHexagramTables
from .interfaces import ProbeExtractor


class _ProbeExtractorImpl(ProbeExtractor):
    """
    默認探針提取器實現（GPU/CPU 自適應）。

    所有計算在 tensor 上完成，最後一次構建 ℝ⁹ 向量。
    GPU: 1 次 CUDA 同步（.tolist() 在調用端）
    CPU: 零同步，純計算
    """

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
        """實現探針提取。返回 (9,) 張量。

        內部策略（接口透明）：
        - GPU：完整 entropy/norm/sort 計算，6 次 CUDA 同步
        - CPU：峰值近似代替熵計算（避開 softmax+log 鏈），速度 2-3x
        """
        tables = PrecomputedHexagramTables.get_instance()
        is_cpu = h.device.type == "cpu"

        # ── ① h密度 (ℝ²)：GPU用完整熵，CPU用峰值近似 ──
        h_1d = h[0]
        hn = float(h_1d.norm().item())
        if is_cpu:
            # CPU 快速路徑：用 1 - max(softmax) 近似熵
            hp = F.softmax(h_1d / 1.0, dim=-1)
            he = float(1.0 - hp.max().item())
        else:
            hp = F.softmax(h_1d / 1.0, dim=-1).clamp(min=1e-8)
            he = float(-(hp * torch.log(hp)).sum().item())

        # ── ② Logits形狀熵 (ℝ¹)：同上 ──
        if is_cpu:
            rp = F.softmax(router_probs[0] / 1.0, dim=-1)
            le = float(1.0 - rp.max().item())
        else:
            rp = router_probs[0].clamp(min=1e-8)
            le = float(-(rp * torch.log(rp)).sum().item())

        # ── ③ 卦象家族 (ℝ³) — 路徑相同 ──
        hex_id_int = int(hexagram_idx[0].item())
        cu = float(tables.upper_trigrams[hex_id_int].item())
        cl = float(tables.lower_trigrams[hex_id_int].item())
        co = float(tables.opposite_upper[hex_id_int].item())

        # ── ④ Q值差距 (ℝ¹) — 路徑相同 ──
        vals_sorted, _ = candidate_values[0, :, 0].sort(descending=True)
        qg = float((vals_sorted[0] - vals_sorted[1]).item()) if vals_sorted.numel() >= 2 else 0.0

        # ── ⑤ 跳躍度 (ℝ¹) — 路徑相同 ──
        if prev_hexagram_idx is not None:
            pi = int(prev_hexagram_idx[0].item())
            jd = float(tables.hamming_matrix[pi, hex_id_int].item())
        else:
            jd = 0.0

        # ── ⑥ 動作置信度 (ℝ¹)：GPU用負熵，CPU用峰值近似 ──
        if is_cpu:
            ap = F.softmax(action_logits[0] / 1.0, dim=-1)
            ac = float(ap.max().item() - 1.0)  # 負的偏離度
        else:
            ap = F.softmax(action_logits[0] / 1.0, dim=-1).clamp(min=1e-8)
            ac = -float((ap * torch.log(ap)).sum().item())

        # ── 組裝 ℝ⁹ tensor ──
        return torch.tensor(
            [hn, he, le, cu, cl, co, qg, jd, ac],
            dtype=torch.float32,
        )


# ══════════════════════════════════════════════════════════════
# 向下兼容接口
# ══════════════════════════════════════════════════════════════

# 模組級單例提取器（惰性初始化）
_extractor: Optional[ProbeExtractor] = None


def _get_extractor() -> ProbeExtractor:
    global _extractor
    if _extractor is None:
        _extractor = ProbeExtractor.create()
    return _extractor


def extract_probes_tensor(
    h: torch.Tensor,
    router_logits: torch.Tensor,
    router_probs: torch.Tensor,
    candidate_values: torch.Tensor,
    hexagram_idx: torch.Tensor,
    prev_hexagram_idx: Optional[torch.Tensor],
    action_logits: torch.Tensor,
) -> torch.Tensor:
    """
    張量化探針提取 — 返回 ℝ⁹ float32 tensor。

    性能最優路徑（無 Python 層包裝，直接 tensor-in tensor-out）。
    調用端通過 .tolist() 獲得 Python 值。
    """
    return _get_extractor().extract(
        h, router_logits, router_probs,
        candidate_values, hexagram_idx,
        prev_hexagram_idx, action_logits,
    )


def extract_probes(
    h: torch.Tensor,
    router_logits: torch.Tensor,
    router_probs: torch.Tensor,
    candidate_values: torch.Tensor,
    hexagram_idx: int,
    prev_hexagram_idx: Optional[int],
    action_logits: torch.Tensor,
):
    """
    向後兼容包裝 — 接受 Python int，返回 ProbeVector NamedTuple。

    新代碼請使用 extract_probes_tensor() 直接返回 tensor。
    """
    device = h.device
    hex_t = torch.tensor([hexagram_idx], device=device)
    prev_t = torch.tensor([prev_hexagram_idx], device=device) if prev_hexagram_idx is not None else None
    probe_t = extract_probes_tensor(h, router_logits, router_probs, candidate_values, hex_t, prev_t, action_logits)

    # 轉為 NamedTuple 保持向後兼容
    vals = probe_t.tolist()
    from collections import namedtuple
    PV = namedtuple("ProbeVector",
        "h_norm h_entropy logit_entropy "
        "clan_upper clan_lower clan_opposite "
        "q_gap jump_distance action_confidence")
    return PV(*vals)


# 舊版函數保留引用，但不推薦使用
entropy_from_logits = lambda logits: -(F.softmax(logits / 1.0, dim=-1).clamp(min=1e-8) * torch.log(F.softmax(logits / 1.0, dim=-1).clamp(min=1e-8))).sum().item() if logits.dim() == 1 else -(F.softmax(logits[0] / 1.0, dim=-1).clamp(min=1e-8) * torch.log(F.softmax(logits[0] / 1.0, dim=-1).clamp(min=1e-8))).sum().item()
negative_entropy = lambda p: -(p.clamp(min=1e-8) * torch.log(p.clamp(min=1e-8))).sum().item()
clan_mapping = lambda idx: (
    float(PrecomputedHexagramTables.get_instance().upper_trigrams[idx].item()),
    float(PrecomputedHexagramTables.get_instance().lower_trigrams[idx].item()),
    float(PrecomputedHexagramTables.get_instance().opposite_upper[idx].item()),
) if 0 <= idx < 64 else (0.0, 0.0, 0.0)
