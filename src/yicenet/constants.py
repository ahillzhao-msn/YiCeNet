"""
YiCeNet 預計算常數 — 所有硬編碼卦象數據集中管理。

本模組負責：
  1. 從 YiCeNetConfig.hexagram_patterns 構建所有派生常數
  2. 惰性初始化預計算張量（首次使用時構建一次）
  3. 提供統一的公開接口供 probes / hexagram / model 使用

所有張量常數一旦構建即凍結（不參與梯度計算，不修改）。
"""

from typing import Optional
import torch


class PrecomputedHexagramTables:
    """
    卦象預計算表 — 所有派生常數集中管理。

    使用方式（惰性初始化，模組級單例）：
        tables = PrecomputedHexagramTables.get_instance()
        upper = tables.upper_trigrams  # (64,)

    Public 屬性（唯讀張量，永不修改）：
        - hex_bits:          (64, 6)    六爻位元張量
        - upper_trigrams:    (64,)      上卦 ID (0-7)
        - lower_trigrams:    (64,)      下卦 ID (0-7)
        - opposite_upper:    (64,)      錯卦的上卦 ID (0-7)
        - hamming_matrix:    (64, 64)   兩兩 Hamming 距離 [0, 1]
    """

    _instance: Optional["PrecomputedHexagramTables"] = None

    def __init__(self, hexagram_patterns: Optional[tuple] = None):
        """
        Args:
            hexagram_patterns: 64 個整數，每整數低6位=六爻模式。
                               若為 None 則從默認 King Wen 順序構建。
        """
        patterns = hexagram_patterns if hexagram_patterns is not None else self._default_patterns()
        assert len(patterns) == 64, f"需要 64 個卦象模式，收到 {len(patterns)}"

        # ── (64,) integer patterns ──
        self._raw: torch.Tensor = torch.tensor(list(patterns), dtype=torch.int64, requires_grad=False)
        self._frozen: bool = True

        # ── 派生張量（惰性）──
        self._hex_bits: Optional[torch.Tensor] = None
        self._upper_trigrams: Optional[torch.Tensor] = None
        self._lower_trigrams: Optional[torch.Tensor] = None
        self._opposite_upper: Optional[torch.Tensor] = None
        self._hamming_matrix: Optional[torch.Tensor] = None

    @staticmethod
    def _default_patterns() -> list[int]:
        """King Wen 順序 64 卦六爻模式（跟 config.py 一致）。"""
        return [
            0b111111, 0b000000, 0b010001, 0b100010,
            0b010111, 0b111010, 0b000010, 0b010000,
            0b110111, 0b111011, 0b111000, 0b000111,
            0b101111, 0b111101, 0b001000, 0b000100,
            0b011001, 0b100110, 0b100101, 0b101001,
            0b100001, 0b011110, 0b100111, 0b111001,
            0b111100, 0b001111, 0b000110, 0b011000,
            0b010010, 0b101101, 0b001110, 0b011100,
            0b001100, 0b110000, 0b000011, 0b110011,
            0b101011, 0b110101, 0b101110, 0b011101,
            0b110010, 0b010011, 0b100011, 0b110001,
            0b011110, 0b011011, 0b110110, 0b001001,
            0b011101, 0b101110, 0b101100, 0b001101,
            0b110100, 0b001011, 0b101010, 0b010101,
            0b011001, 0b100110, 0b110001, 0b100011,
            0b110010, 0b010011, 0b011111, 0b111110,
        ]

    # ── 惰性初始化屬性 ──

    @property
    def hex_bits(self) -> torch.Tensor:
        """(64, 6) 六爻位元張量，col 0=上爻，col 5=初爻"""
        if self._hex_bits is None:
            bits = torch.zeros(64, 6, dtype=torch.int64, requires_grad=False)
            for i in range(6):
                bits[:, 5 - i] = (self._raw >> i) & 1
            self._hex_bits = bits
        return self._hex_bits

    @property
    def upper_trigrams(self) -> torch.Tensor:
        """(64,) 上卦 ID (0-7)"""
        if self._upper_trigrams is None:
            b = self.hex_bits
            self._upper_trigrams = (b[:, 0] << 2) | (b[:, 1] << 1) | b[:, 2]
        return self._upper_trigrams

    @property
    def lower_trigrams(self) -> torch.Tensor:
        """(64,) 下卦 ID (0-7)"""
        if self._lower_trigrams is None:
            b = self.hex_bits
            self._lower_trigrams = (b[:, 3] << 2) | (b[:, 4] << 1) | b[:, 5]
        return self._lower_trigrams

    @property
    def opposite_upper(self) -> torch.Tensor:
        """(64,) 錯卦的上卦 ID (0-7)"""
        if self._opposite_upper is None:
            ob = 1 - self.hex_bits  # 全部位元反轉 = 錯卦
            self._opposite_upper = (ob[:, 0] << 2) | (ob[:, 1] << 1) | ob[:, 2]
        return self._opposite_upper

    @property
    def hamming_matrix(self) -> torch.Tensor:
        """(64, 64) 兩兩 Hamming 距離 [0, 1]，歸一化到 6 位元"""
        if self._hamming_matrix is None:
            b = self.hex_bits  # (64, 6)
            diff = b.unsqueeze(0) ^ b.unsqueeze(1)  # (64, 64, 6) XOR
            self._hamming_matrix = diff.sum(dim=-1).float() / 6.0
        return self._hamming_matrix

    # ── 私有：設備感知快取 ──
    _device: Optional[torch.device] = None
    """當前快取設備。若變更則自動重新傳輸所有張量。"""

    def _to_device(self, device: torch.device) -> "PrecomputedHexagramTables":
        """將所有張量傳輸到指定設備（快取，僅首次或設備變更時執行）。"""
        if self._device == device:
            return self
        self._device = device
        # 強制初始化所有惰性屬性，然後傳輸
        for attr in ["hex_bits", "upper_trigrams", "lower_trigrams",
                      "opposite_upper", "hamming_matrix"]:
            val = getattr(self, attr)  # triggers lazy init
            if val is not None:
                setattr(self, f"_{attr}", val.to(device))
        return self

    @classmethod
    def get_instance(cls, hexagram_patterns: Optional[tuple] = None) -> "PrecomputedHexagramTables":
        """獲取或創建模組級單例。"""
        if cls._instance is None:
            cls._instance = cls(hexagram_patterns)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """重置單例（測試用）。"""
        cls._instance = None
