"""
YiCeNet (易策网络) — 卦象顯示格式化。

獨立的顯示層，YiCeNet 自持。
KAFED 只需調用 format_prediction(result, mode="compact") 並傳遞結果。

遇事不決問周易——YiCeNet 可在任何語境下直接調用。
"""

# 64 卦 Unicode 符號（King Wen 序，hexagram_number 1-64 → U+4DC0+）
HEXAGRAM_SYMBOLS: list[str] = [
    "䷀", "䷁", "䷂", "䷃", "䷄", "䷅", "䷆", "䷇",
    "䷈", "䷉", "䷊", "䷋", "䷌", "䷍", "䷎", "䷏",
    "䷐", "䷑", "䷒", "䷓", "䷔", "䷕", "䷖", "䷗",
    "䷘", "䷙", "䷚", "䷛", "䷜", "䷝", "䷞", "䷟",
    "䷠", "䷡", "䷢", "䷣", "䷤", "䷥", "䷦", "䷧",
    "䷨", "䷩", "䷪", "䷫", "䷬", "䷭", "䷮", "䷯",
    "䷰", "䷱", "䷲", "䷳", "䷴", "䷵", "䷶", "䷷",
    "䷸", "䷹", "䷺", "䷻", "䷼", "䷽", "䷾", "䷿",
]

# 卦名（從 yicenet_engine 引入，此處保留備用）
from yicenet.yicenet_engine import HEXAGRAM_NAMES  # noqa: F401


def hexagram_symbol(hexagram_number: int) -> str:
    """hexagram_number (1-64) → Unicode 卦象符號。"""
    if 1 <= hexagram_number <= 64:
        return HEXAGRAM_SYMBOLS[hexagram_number - 1]
    return ""


def _get_main_id(result: dict) -> int:
    """獲取主卦 ID (0-63)。優先 selected_hexagram_id，其次 hexagram_id。"""
    sid = result.get("selected_hexagram_id")
    if sid is not None:
        return sid
    return result.get("hexagram_id", 0)


def format_prediction(result: dict, mode: str = "compact") -> str:
    """格式化 YiCeNet 預測結果。

    Args:
        result: engine.predict() 返回的原始字典
        mode: "compact" — 精簡（用於 flow chain）
              "detailed" — 完整（用於回應頭部）

    Returns:
        格式化字串
    """
    hid = _get_main_id(result)
    num = hid + 1  # hexagram_number = id + 1
    symbol = hexagram_symbol(num)
    name = result.get("selected_hexagram_name", "") or result.get("hexagram_name", "")
    qs = result.get("q_values", [])
    best_q = max(qs) if qs else 0.0
    candidates = result.get("candidates", [])

    if mode == "compact":
        # 精簡模式：符號 + 卦名
        parts = []
        if symbol:
            parts.append(symbol)
        if name:
            parts.append(name[:6])
        else:
            parts.append(f"#{num}")
        return " ".join(parts)

    # detailed 模式：完整資訊
    lines = []
    if symbol and name:
        lines.append(f"卦: {symbol} {name}（第{num}卦）")
    elif name:
        lines.append(f"卦: {name}（第{num}卦）")

    lines.append(f"  最佳 Q: {best_q:.4f}")

    if candidates:
        top5 = sorted(candidates, key=lambda c: c.get("q_value", 0), reverse=True)[:5]
        cand_strs = []
        for c in top5:
            cid = c.get("hexagram_id", 0)
            cnum = cid + 1
            csym = hexagram_symbol(cnum)
            cname = c.get("hexagram_name", f"#{cnum}")
            cq = c.get("q_value", 0)
            cand_strs.append(f"{csym} {cname}({cq:.3f})")
        lines.append(f"  候選: {' | '.join(cand_strs)}")

    action = result.get("action_name", "")
    if action:
        lines.append(f"  推薦: {action}")

    return "\n".join(lines)
