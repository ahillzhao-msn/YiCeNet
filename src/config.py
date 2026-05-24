"""
YiCeNet configuration — all hyperparameters in one place.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class YiCeNetConfig:
    # ── Encoder (Tiny Transformer) ──
    vocab_size: int = 8000
    hidden_dim: int = 256
    intermediate_dim: int = 1024
    num_heads: int = 4
    num_encoder_layers: int = 4
    max_seq_len: int = 128
    dropout: float = 0.1

    # ── Hexagram / Trigram ──
    num_trigrams: int = 8
    num_hexagrams: int = 64
    num_lines: int = 6  # each hexagram = 6 lines (爻)

    # ── Action space ──
    num_actions: int = 50  # orchestration primitives

    # ── Gumbel-Softmax ──
    gumbel_tau_init: float = 1.0
    gumbel_tau_min: float = 0.1
    gumbel_tau_decay: float = 0.995

    # ── Value Network ──
    value_hidden: int = 128

    # ── Probe System ──
    probe_dim: int = 9  # 六探針合計維度
    """六探針合計維度：h密度(2) + logits形狀(1) + 卦象家族(3) + Q值差距(1) + 跳躍度(1) + 動作置信(1)"""

    # ── World Model v2 (Dual-Head + Power Law) ──
    wm_shared_dim: int = 128
    """世界模型共享層維度"""
    wm_hexagram_head_dim: int = 64
    """頭A：結果卦象分布（等於卦數）"""
    num_external_metrics: int = 3
    """頭B：外部向量維度（token消耗 + 續航長度 + 滿意度）"""
    wm_beta: float = 0.3
    """外部向量 loss 權重係數 β"""
    wm_slow_tau_days: float = 30.0
    """頭A 冪律衰減時間常數（天）"""
    wm_fast_tau_days: float = 3.0
    """頭B 冪律衰減時間常數（天）"""
    wm_alpha: float = 1.5
    """冪律衰減指數 α"""

    # ── External Metrics ──
    ext_continuation_weight: float = 0.4
    """續航針投影權重"""
    ext_correction_weight: float = 0.4
    """校正針投影權重"""
    ext_completion_weight: float = 0.4
    """完成針投影權重"""
    ext_projection_temperature: float = 0.5
    """投影 softmax 溫度（低=尖銳，高=平滑）"""

    # ── Training ──
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    max_epochs: int = 100
    clip_grad_norm: float = 1.0

    # ── RL ──
    ppo_clip_epsilon: float = 0.2
    ppo_epochs: int = 4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # ── Paths ──
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # ── Hexagram line patterns (King Wen order, 64 hexagrams) ──
    # Each is a 6-bit integer; bit 5 = top line (上九/上六),
    # bit 0 = bottom line (初九/初六)
    hexagram_patterns: tuple = field(default_factory=lambda: tuple(
        _king_wen_hexagrams()
    ))

    def __post_init__(self):
        assert len(self.hexagram_patterns) == 64


def _king_wen_hexagrams() -> list[int]:
    """
    Return the 64 hexagram patterns in King Wen (文王) order.

    Each hexagram is encoded as a 6-bit integer:
      bit 5 (MSB) = top line (上爻)
      bit 0 (LSB) = bottom line (初爻)
    where 1 = solid/yang (阳), 0 = broken/yin (阴)

    Source: traditional I Ching sequence.
    """
    # fmt: off
    return [
        0b111111, 0b000000, 0b010001, 0b100010,  # 1-4: 乾 坤 屯 蒙
        0b010111, 0b111010, 0b000010, 0b010000,  # 5-8: 需 讼 师 比
        0b110111, 0b111011, 0b111000, 0b000111,  # 9-12: 小畜 履 泰 否
        0b101111, 0b111101, 0b001000, 0b000100,  # 13-16: 同人 大有 谦 豫
        0b011001, 0b100110, 0b100101, 0b101001,  # 17-20: 随 蛊 临 观
        0b100001, 0b011110, 0b100111, 0b111001,  # 21-24: 噬嗑 贲 剥 复
        0b111100, 0b001111, 0b000110, 0b011000,  # 25-28: 无妄 大畜 颐 大过
        0b010010, 0b101101, 0b001110, 0b011100,  # 29-32: 坎 离 咸 恒
        0b001100, 0b110000, 0b000011, 0b110011,  # 33-36: 遯 大壮 晋 明夷
        0b101011, 0b110101, 0b101110, 0b011101,  # 37-40: 家人 睽 蹇 解
        0b110010, 0b010011, 0b100011, 0b110001,  # 41-44: 损 益 夬 姤
        0b011110, 0b011011, 0b110110, 0b001001,  # 45-48: 萃 升 困 井
        0b011101, 0b101110, 0b101100, 0b001101,  # 49-52: 革 鼎 震 艮
        0b110100, 0b001011, 0b101010, 0b010101,  # 53-56: 渐 归妹 丰 旅
        0b011001, 0b100110, 0b110001, 0b100011,  # 57-60: 巽 兑 涣 节
        0b110010, 0b010011, 0b011111, 0b111110,  # 61-64: 中孚 小过 既济 未济
    ]
    # fmt: on
