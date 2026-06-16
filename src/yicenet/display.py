"""
YiCeNet (易策网络) — 卦象顯示層。

Interface-driven, config-injected, platform-aware.

Entry points:
    get_display(cfg: DisplayConfig | None = None) -> IDisplay
        Factory. Reads ~/.yicenet/config.yaml when cfg is None.

    format_prediction(result, mode="compact") -> str
        Backward-compatible one-shot helper.

Implementations:
    TerminalDisplay  — human-readable, compact or detailed
    JsonDisplay      — structured JSON for machine consumers
    SilentDisplay    — no-op for testing / silent mode

Display injection points:
    Hermes pre_llm_call  → TerminalDisplay.render()  (text injected into prompt)
    Claude Code hook     → JsonDisplay.render()       (JSON context for Claude)
    standalone / debug   → TerminalDisplay detailed
"""
from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

from .interfaces import IDisplay

if TYPE_CHECKING:
    from .types import PredictionResult, DisplayConfig


# ── Static data ────────────────────────────────────────────────────────────────
# Defined here (not re-exported from yicenet_engine) so this module is
# standalone (no torch dependency at import time).

HEXAGRAM_NAMES: list[str] = [
    "乾", "坤", "屯", "蒙", "需", "讼", "师", "比",
    "小畜", "履", "泰", "否", "同人", "大有", "谦", "豫",
    "随", "蛊", "临", "观", "噬嗑", "贲", "剥", "复",
    "无妄", "大畜", "颐", "大过", "坎", "离", "咸", "恒",
    "遯", "大壮", "晋", "明夷", "家人", "睽", "蹇", "解",
    "损", "益", "夬", "姤", "萃", "升", "困", "井",
    "革", "鼎", "震", "艮", "渐", "归妹", "丰", "旅",
    "巽", "兑", "涣", "节", "中孚", "小过", "既济", "未济",
]

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

# 64-卦卦辞（简判，传统《周易》，≤12字）
_HEXAGRAM_JUDGMENTS: list[str] = [
    "元亨利贞",              # 1  乾
    "元亨利牝马之贞",        # 2  坤
    "元亨利贞",              # 3  屯
    "亨",                    # 4  蒙
    "有孚光亨利贞",          # 5  需
    "有孚窒惕",              # 6  讼
    "贞吉",                  # 7  师
    "吉",                    # 8  比
    "亨密云不雨",            # 9  小畜
    "履虎尾不咥人亨",        # 10 履
    "小往大来吉亨",          # 11 泰
    "不利君子贞",            # 12 否
    "亨",                    # 13 同人
    "元亨",                  # 14 大有
    "亨君子有终",            # 15 谦
    "利建侯行师",            # 16 豫
    "元亨利贞",              # 17 随
    "元亨利涉大川",          # 18 蛊
    "元亨利贞",              # 19 临
    "盥而不荐",              # 20 观
    "亨利用狱",              # 21 噬嗑
    "亨小利有攸往",          # 22 贲
    "不利有攸往",            # 23 剥
    "亨出入无疾",            # 24 复
    "元亨利贞",              # 25 无妄
    "利贞",                  # 26 大畜
    "贞吉观颐",              # 27 颐
    "栋桡利有攸往亨",        # 28 大过
    "习坎有孚维心亨",        # 29 坎
    "利贞亨",                # 30 离
    "亨取女吉",              # 31 咸
    "亨无咎利贞利有攸往",    # 32 恒
    "亨小利贞",              # 33 遯
    "利贞",                  # 34 大壮
    "康侯用锡马蕃庶",        # 35 晋
    "利艰贞",                # 36 明夷
    "利女贞",                # 37 家人
    "小事吉",                # 38 睽
    "利西南不利东北",        # 39 蹇
    "亨",                    # 40 解
    "有孚元吉无咎",          # 41 损
    "利有攸往利涉大川",      # 42 益
    "扬于王庭",              # 43 夬
    "女壮勿用取女",          # 44 姤
    "亨王假有庙",            # 45 萃
    "元亨利贞",              # 46 升
    "亨贞大人吉",            # 47 困
    "改邑不改井",            # 48 井
    "元亨利贞",              # 49 革
    "元吉亨",                # 50 鼎
    "亨",                    # 51 震
    "艮其背",                # 52 艮
    "女归吉利贞",            # 53 渐
    "征凶无攸利",            # 54 归妹
    "亨王假之",              # 55 丰
    "小亨旅贞吉",            # 56 旅
    "小亨利有攸往",          # 57 巽
    "亨",                    # 58 兑
    "亨王假有庙",            # 59 涣
    "亨",                    # 60 节
    "豚鱼吉利涉大川",        # 61 中孚
    "亨利贞",                # 62 小过
    "亨小利贞",              # 63 既济
    "亨小狐汔济",            # 64 未济
]

_CHAIN_SEP = "→"
_CHAIN_TAIL = 4   # max hexagrams shown in chain prefix


# ── Platform detection ─────────────────────────────────────────────────────────


def _supports_unicode() -> bool:
    """True if stdout can encode U+4DC0–U+4DFF (I Ching hexagram unified block)."""
    try:
        "䷟".encode(sys.stdout.encoding or "utf-8")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


# ── Low-level helpers (public for callers who need raw lookups) ────────────────


def hexagram_symbol(hexagram_number: int, use_unicode: bool = True) -> str:
    """hexagram_number (1-64) → Unicode glyph, or empty string when use_unicode=False.

    Callers that need an ASCII fallback label should use #N notation directly;
    the symbol slot is intentionally empty so wrappers avoid double-bracketing.
    """
    if 1 <= hexagram_number <= 64 and use_unicode:
        return HEXAGRAM_SYMBOLS[hexagram_number - 1]
    return ""


def hexagram_judgment(hexagram_number: int) -> str:
    """hexagram_number (1-64) → 卦辞简判。"""
    if 1 <= hexagram_number <= 64:
        return _HEXAGRAM_JUDGMENTS[hexagram_number - 1]
    return ""


def _main_id(result: dict) -> int:
    """Extract primary hexagram_id (0-indexed) from result dict."""
    sid = result.get("selected_hexagram_id")
    return sid if sid is not None else result.get("hexagram_id", 0)


def _main_name(result: dict) -> str:
    return result.get("selected_hexagram_name", "") or result.get("hexagram_name", "")


# ── Display implementations ────────────────────────────────────────────────────


class TerminalDisplay(IDisplay):
    """Human-readable renderer for terminal output and prompt injection.

    compact:  [䷟ 恒] * 亨无咎利贞
              乾→屯→[䷟ 恒] * 亨无咎利贞   (with hexagram_chain)
    detailed: multi-line with confidence, candidates, hint
    """

    def __init__(
        self,
        mode: str = "compact",
        hexagram_chain: bool = False,
        unicode_symbols: bool = True,
    ) -> None:
        self._mode = mode if mode in ("compact", "detailed") else "compact"
        self._hexagram_chain = hexagram_chain
        self._unicode = unicode_symbols

    @property
    def needs_chain(self) -> bool:
        return self._hexagram_chain

    def render(self, result: dict, chain: list[int] | None = None) -> str:
        if self._mode == "detailed":
            return self._render_detailed(result)
        return self._render_compact(result, chain)

    def render_chain(self, hexagram_ids: list[int]) -> str:
        """Render chain history prefix. hexagram_ids: 0-indexed."""
        if not hexagram_ids:
            return ""
        tail = hexagram_ids[-_CHAIN_TAIL:]
        names = [HEXAGRAM_NAMES[hid] for hid in tail if 0 <= hid < 64]
        return _CHAIN_SEP.join(names)

    # ── private renderers ──────────────────────────────────────────────────

    def _render_compact(self, result: dict, chain: list[int] | None) -> str:
        hid = _main_id(result)
        num = hid + 1
        sym = hexagram_symbol(num, use_unicode=self._unicode)
        name = _main_name(result)
        judgment = hexagram_judgment(num)

        sym_part = f"{sym} " if sym else ""
        name_part = name[:6] if name else f"#{num}"
        label = f"[{sym_part}{name_part}]"
        if judgment:
            label = f"{label} * {judgment}"

        if self._hexagram_chain and chain:
            chain_str = self.render_chain(chain)
            if chain_str:
                return f"{chain_str}{_CHAIN_SEP}{label}"
        return label

    def _render_detailed(self, result: dict) -> str:
        hid = _main_id(result)
        num = hid + 1
        sym = hexagram_symbol(num, use_unicode=self._unicode)
        name = _main_name(result)
        action = result.get("action_name", "")
        qs = result.get("q_values", [])
        env_conf = result.get("env_confidence")
        ctx_status = result.get("context_status", "")
        ctx_hint = result.get("context_hint", "")

        lines: list[str] = []

        # ── Header ─────────────────────────────────────────
        sym_str = f"{sym} " if sym else ""
        name_str = name if name else f"#{num}"
        lines.append(f"卦: {sym_str}{name_str}（第{num}卦）")

        # ── Confidence + action ─────────────────────────────
        meta: list[str] = []
        if env_conf is not None:
            badge = f" [{ctx_status}]" if ctx_status else ""
            meta.append(f"置信: {env_conf:.2f}{badge}")
        if action:
            meta.append(f"推荐: {action}")
        if meta:
            lines.append(f"  {' · '.join(meta)}")

        # ── Top-5 candidates ────────────────────────────────
        candidates = result.get("candidates", [])
        if candidates:
            top5 = sorted(candidates, key=lambda c: c.get("q_value", 0), reverse=True)[:5]
            parts = []
            for c in top5:
                cid = c.get("hexagram_id", 0)
                cnum = cid + 1
                csym = hexagram_symbol(cnum, use_unicode=self._unicode)
                cname = c.get("hexagram_name") or (HEXAGRAM_NAMES[cid] if 0 <= cid < 64 else f"#{cnum}")
                cq = c.get("q_value", 0)
                parts.append(f"{csym}{cname}({cq:.3f})")
            lines.append(f"  候选: {' | '.join(parts)}")
        elif qs:
            lines.append(f"  最佳Q: {max(qs):.4f}")

        # ── Context hint ────────────────────────────────────
        if ctx_hint:
            lines.append(f"  提示: {ctx_hint}")

        return "\n".join(lines)


class JsonDisplay(IDisplay):
    """Structured JSON renderer for machine consumers (Claude Code hook injection).

    render() → JSON string with key inference fields.
    render_chain() → JSON array string of hexagram_ids.
    """

    def render(self, result: dict, chain: list[int] | None = None) -> str:
        hid = _main_id(result)
        num = hid + 1
        payload: dict = {
            "hexagram_id": hid,
            "hexagram_number": num,
            "hexagram_name": _main_name(result),
            "hexagram_symbol": hexagram_symbol(num),
            "judgment": hexagram_judgment(num),
            "action_name": result.get("action_name", ""),
            "env_confidence": result.get("env_confidence"),
            "context_status": result.get("context_status", ""),
        }
        if result.get("context_hint"):
            payload["context_hint"] = result["context_hint"]
        if chain is not None:
            payload["hexagram_chain"] = chain
        return json.dumps(payload, ensure_ascii=False)

    def render_chain(self, hexagram_ids: list[int]) -> str:
        return json.dumps(hexagram_ids)


class SilentDisplay(IDisplay):
    """No-op renderer. For testing and silent/headless mode."""

    def render(self, result: dict, chain: list[int] | None = None) -> str:
        return ""

    def render_chain(self, hexagram_ids: list[int]) -> str:
        return ""


# ── Factory ────────────────────────────────────────────────────────────────────


def get_display(cfg: "DisplayConfig | None" = None) -> IDisplay:
    """Create IDisplay from config.

    Reads ~/.yicenet/config.yaml via get_display_config() when cfg is None.
    Unicode capability auto-detected from sys.stdout.encoding when
    unicode_symbols is absent from cfg.
    """
    if cfg is None:
        from .config import get_display_config
        cfg = get_display_config()

    mode = cfg.get("mode", "compact")

    if mode == "silent":
        return SilentDisplay()
    if mode == "json":
        return JsonDisplay()

    # Resolve unicode: explicit config > auto-detect
    unicode_cfg = cfg.get("unicode_symbols")
    use_unicode = _supports_unicode() if unicode_cfg is None else bool(unicode_cfg)

    return TerminalDisplay(
        mode=mode,
        hexagram_chain=bool(cfg.get("hexagram_chain", False)),
        unicode_symbols=use_unicode,
    )


# ── Backward-compatible entry point ───────────────────────────────────────────


def format_prediction(result: dict, mode: str = "compact") -> str:
    """Render prediction result. Legacy one-shot API; use get_display() for injection."""
    return TerminalDisplay(mode=mode, unicode_symbols=_supports_unicode()).render(result)
