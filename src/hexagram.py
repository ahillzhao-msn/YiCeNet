"""
Hexagram (卦象) transformations — the deterministic reasoning engine.

Inspired by I Ching philosophy:
  错卦 (Cuo / Opposite):   bitwise NOT — complete opposite perspective
  综卦 (Zong / Upside-down): reverse line order — inverted viewpoint
  互卦 (Hu / Inner):        extract inner trigram — hidden internal structure
  之卦 (Zhi / Changing):    flip one line — targeted mutation
"""

import torch


def hexagram_to_lines(h: torch.Tensor) -> torch.Tensor:
    """
    Convert hexagram index (batch) to 6-bit line tensor.

    Args:
        h: (...,) hexagram indices, int in [0, 63]

    Returns:
        lines: (..., 6) binary tensor, bit 5 = top line, bit 0 = bottom line
    """
    # h is (...,) → (..., 6)
    return (h.unsqueeze(-1) >> torch.arange(5, -1, -1, device=h.device)) & 1


def lines_to_hexagram(lines: torch.Tensor) -> torch.Tensor:
    """
    Convert 6-bit line tensor back to hexagram index.

    Args:
        lines: (..., 6) binary tensor

    Returns:
        h: (...,) hexagram indices in [0, 63]
    """
    weights = torch.tensor(
        [32, 16, 8, 4, 2, 1], dtype=lines.dtype, device=lines.device
    )
    return (lines * weights).sum(dim=-1).long()


def cuo_hexagram(opposite: torch.Tensor) -> torch.Tensor:
    """
    错卦 — Opposite hexagram.
    Every line flips: yin↔yang.

    Args:
        opposite: (...,) hexagram indices

    Returns:
        (...,) hexagram indices of the opposite hexagram
    """
    # bitwise NOT, mask to 6 bits
    return (~opposite) & 0b111111


def zong_hexagram(lines: torch.Tensor) -> torch.Tensor:
    """
    综卦 — Upside-down hexagram.
    Read the hexagram from top to bottom reversed.

    Args:
        lines: (..., 6) binary tensor, [top→bottom]

    Returns:
        (...,) hexagram indices
    """
    # Reverse the 6 bits: bottom becomes top
    reversed_lines = lines.flip(-1)
    return lines_to_hexagram(reversed_lines)


def hu_hexagram(lines: torch.Tensor) -> torch.Tensor:
    """
    互卦 — Inner hexagram.
    Extract the inner trigram:
      - lower inner trigram = lines 2,3,4 (0-indexed LSB order)
      - upper inner trigram = lines 3,4,5
    ┌───┬───┬───┬───┬───┬───┐
    │ L0│ L1│ L2│ L3│ L4│ L5│  ← lines[0..5]
    └───┴───┴───┴───┴───┴───┘
              └───┴───┴───┘  → lower inner (lines 2,3,4)
                  └───┴───┴───┘ → upper inner (lines 3,4,5)

    The resulting hexagram has upper inner as the top 3 lines,
    lower inner as the bottom 3 lines.

    In traditional I Ching: take lines 2,3,4 as lower trigram,
    lines 3,4,5 as upper trigram, then combine into a new hexagram.

    Args:
        lines: (..., 6) binary tensor

    Returns:
        (...,) hexagram indices
    """
    # Extract inner trigrams
    # lines are ordered [top→bottom], so:
    # In terms of positions (0-indexed from top):
    #   Top: lines[0], lines[1], lines[2]
    #   Middle: lines[1], lines[2], lines[3] 
    #   Bottom: lines[2], lines[3], lines[4]
    # 
    # But in I Ching, the hai hexagram is constructed as:
    # Upper trigram = lines 3,4,5 (3rd-5th from bottom) = lines[3], lines[2], lines[1] from top
    # Actually let me be careful.
    #
    # In traditional hexagram, lines are indexed from bottom (初) to top (上):
    #   初(1), 二(2), 三(3), 四(4), 五(5), 上(6)
    # 
    # Hu (互卦) takes:
    #   Lower inner trigram = lines 2,3,4 (二三四爻)
    #   Upper inner trigram = lines 3,4,5 (三四五爻)
    #
    # In our tensor where index 0 = top (上), index 5 = bottom (初):
    #   lines[5] = 初, lines[4] = 二, lines[3] = 三, lines[2] = 四, lines[1] = 五, lines[0] = 上
    #
    # So lower inner trigram (二三四爻) = lines[4], lines[3], lines[2]
    # And upper inner trigram (三四五爻) = lines[3], lines[2], lines[1]

    lower_inner = torch.stack([lines[..., 4], lines[..., 3], lines[..., 2]], dim=-1)
    upper_inner = torch.stack([lines[..., 3], lines[..., 2], lines[..., 1]], dim=-1)

    # Combine: upper inner as top 3, lower inner as bottom 3
    new_lines = torch.cat([upper_inner, lower_inner], dim=-1)
    return lines_to_hexagram(new_lines)


def zhi_hexagram(lines: torch.Tensor, flip_position: torch.Tensor) -> torch.Tensor:
    """
    之卦 — Changing hexagram.
    Flip one specific line (爻) based on attention / bottleneck.

    Args:
        lines: (..., 6) binary tensor
        flip_position: (...,) integer in [0,5] indicating which line to flip

    Returns:
        (...,) hexagram indices
    """
    # Create one-hot mask for the flip position
    mask = torch.zeros_like(lines)
    arange = torch.arange(lines.size(-1), device=lines.device)
    flip_mask = (arange == flip_position.unsqueeze(-1)).float()
    
    # Flip: 1 - x for the chosen position
    new_lines = torch.where(flip_mask.bool(), 1 - lines, lines)
    return lines_to_hexagram(new_lines)


def generate_candidates(
    hexagram_idx: torch.Tensor,
    attention_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Generate candidate hexagram set via 错/综/互/变 transformations.

    Produces up to 8 candidates:
      0: original (本卦)
      1: 错卦 (opposite)
      2: 综卦 (upside-down)
      3: 互卦 (inner)
      4-7: 之卦 (change each of the 4 least certain lines, or 4 random)

    Args:
        hexagram_idx: (B,) current hexagram index
        attention_weights: (B, 6) optional — used to pick which lines to flip
                          Lower attention → less important → not bottleneck

    Returns:
        candidates: (B, K, 6) binary line representations of candidates
                    where K = 8
    """
    B = hexagram_idx.shape[0]
    device = hexagram_idx.device

    # Convert to lines
    lines = hexagram_to_lines(hexagram_idx)  # (B, 6)

    # Transform 1: 错卦 (opposite) — bitwise NOT
    cuo = cuo_hexagram(hexagram_idx)
    cuo_lines = hexagram_to_lines(cuo)  # (B, 6)

    # Transform 2: 综卦 (upside-down)
    zong_lines = zong_hexagram(lines)  # (B,) → need lines
    # zong returns hexagram indices, convert back to lines
    zong = lines_to_hexagram(lines.flip(-1))  # simpler: just invert order

    # Transform 3: 互卦 (inner)
    hu = hu_hexagram(lines)

    # Transform 4-7: 之卦 — flip the 4 most-uncertain lines
    if attention_weights is None:
        # No attention → flip lines 0,1,4,5 (top 2 + bottom 2)
        flip_positions = torch.tensor([0, 1, 4, 5], device=device)
    else:
        # Flip the lines with lowest attention (least important → bottleneck?)
        # Actually in I Ching, the changing lines are the ones that are "moving"
        # In our context, we want to explore variants of the most uncertain lines
        # Use attention: higher attention = more important to keep
        # So flip the ones with lowest attention
        _, flip_positions = torch.topk(
            -attention_weights, k=4, dim=-1
        )  # (B, 4)

    zhi_candidates = []
    for i in range(4):
        pos = (
            flip_positions[:, i]
            if attention_weights is not None
            else flip_positions[i].expand(B)
        )
        zhi = zhi_hexagram(lines, pos)
        zhi_candidates.append(zhi)

    # Stack all: (B, 8)
    all_candidates = torch.stack(
        [
            hexagram_idx,               # 0: 本卦 (original)
            cuo,                        # 1: 错卦 (opposite)
            lines_to_hexagram(lines.flip(-1)),  # 2: 综卦 (upside-down)
            hu,                         # 3: 互卦 (inner)
            zhi_candidates[0],          # 4: 之卦 (change line 1)
            zhi_candidates[1],          # 5: 之卦 (change line 2)
            zhi_candidates[2],          # 6: 之卦 (change line 3)
            zhi_candidates[3],          # 7: 之卦 (change line 4)
        ],
        dim=-1,
    )  # (B, 8)

    return all_candidates
