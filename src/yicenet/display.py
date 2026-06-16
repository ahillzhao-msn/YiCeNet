"""
YiCeNet (易策网络) — 卦象顯示格式化。

獨立的顯示層，YiCeNet 自持。
# 外部調用: format_prediction(result, mode="compact")

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

# 64 卦卦辞（简判，传统《周易》）
# 每卦 ≤10 字，简短用于 compact 模式的 * 判词
_HEXAGRAM_JUDGMENTS: list[str] = [
    "元亨利贞",          # 1 乾
    "元亨利牝马之贞",    # 2 坤
    "元亨利贞",          # 3 屯
    "亨",                # 4 蒙
    "有孚光亨利贞",      # 5 需
    "有孚窒惕",          # 6 讼
    "贞吉",              # 7 师
    "吉",                # 8 比
    "亨密云不雨",        # 9 小畜
    "履虎尾不咥人亨",   # 10 履
    "小往大来吉亨",      # 11 泰
    "不利君子贞",        # 12 否
    "亨",                # 13 同人
    "元亨",              # 14 大有
    "亨君子有终",        # 15 谦
    "利建侯行师",        # 16 豫
    "元亨利贞",          # 17 随
    "元亨利涉大川",      # 18 蛊
    "元亨利贞",          # 19 临
    "盥而不荐",          # 20 观
    "亨利用狱",          # 21 噬嗑
    "亨小利有攸往",      # 22 贲
    "不利有攸往",        # 23 剥
    "亨出入无疾",        # 24 复
    "元亨利贞",          # 25 无妄
    "利贞",              # 26 大畜
    "贞吉观颐",          # 27 颐
    "栋桡利有攸往亨",    # 28 大过
    "习坎有孚维心亨",    # 29 坎
    "利贞亨",            # 30 离
    "亨取女吉",          # 31 咸
    "亨无咎利贞利有攸往", # 32 恒
    "亨小利贞",          # 33 遯
    "利贞",              # 34 大壮
    "康侯用锡马蕃庶",    # 35 晋
    "利艰贞",            # 36 明夷
    "利女贞",            # 37 家人
    "小事吉",            # 38 睽
    "利西南不利东北",    # 39 蹇
    "亨",                # 40 解
    "有孚元吉无咎",      # 41 损
    "利有攸往利涉大川",  # 42 益
    "扬于王庭",          # 43 夬
    "女壮勿用取女",      # 44 姤
    "亨王假有庙",        # 45 萃
    "元亨利贞",          # 46 升
    "亨贞大人吉",        # 47 困
    "改邑不改井",        # 48 井
    "元亨利贞",          # 49 革
    "元吉亨",            # 50 鼎
    "亨",                # 51 震
    "艮其背",            # 52 艮
    "女归吉利贞",        # 53 渐
    "征凶无攸利",        # 54 归妹
    "亨王假之",          # 55 丰
    "小亨旅贞吉",        # 56 旅
    "小亨利有攸往",      # 57 巽
    "亨",                # 58 兑
    "亨王假有庙",        # 59 涣
    "亨",                # 60 节
    "豚鱼吉利涉大川",    # 61 中孚
    "亨利贞",            # 62 小过
    "亨小利贞",          # 63 既济
    "亨小狐汔济",        # 64 未济
]


def hexagram_judgment(hexagram_number: int) -> str:
    """hexagram_number (1-64) → 卦辞（简判）。"""
    if 1 <= hexagram_number <= 64:
        return _HEXAGRAM_JUDGMENTS[hexagram_number - 1]
    return ""


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
    """Format YiCeNet prediction result.

    Args:
        result: engine.predict() 返回的原始字典
        mode:   "compact" — 精简（用于 flow chain）
                "detailed" — 完整（用于回应头部）

    Returns:
        格式化字串

    Note:
        LOOM 调用时显式传入 mode 参数，不读 config。
        独立运行时可用 get_display_config() 读取用户偏好后传入。
    """
    hid = _get_main_id(result)
    num = hid + 1  # hexagram_number = id + 1
    symbol = hexagram_symbol(num)
    name = result.get("selected_hexagram_name", "") or result.get("hexagram_name", "")
    qs = result.get("q_values", [])
    best_q = max(qs) if qs else 0.0
    candidates = result.get("candidates", [])

    if mode == "compact":
        # 精简模式：[䷟ 恒] * 亨无咎利贞 — 卦象+卦名 * 简判
        parts = []
        if symbol:
            parts.append(symbol)
        if name:
            parts.append(name[:6])
        else:
            parts.append(f"#{num}")
        result_str = f"[{' '.join(parts)}]"
        judgment = hexagram_judgment(num)
        if judgment:
            result_str += f" * {judgment}"
        return result_str

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
