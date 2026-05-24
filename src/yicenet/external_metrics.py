"""
外部向量系統 — YiCeNet v5 動態環境指標提取。

從 session 數據中提取外部環境衡量指標，作為 World Model 頭B的訓練目標。
推理時不參與決策路徑，僅作為參考獎信的影子感知。

外部向量維度 (N=3)：
  [token_cost, response_length, satisfaction]
"""

import re
from pathlib import Path
from typing import Optional


# 滿意度關鍵詞
_PRAISE_PATTERNS = [
    r"\b(good|great|perfect|excellent|amazing|wonderful|nice|awesome)\b",
    r"\b(正確|對|好|完美|讚|厲害|不錯)\b",
    r"\b(thanks|thank you|ty|thx|cheers|appreciate)\b",
    r"\b(謝謝|感謝|多謝)\b",
]

_CORRECTION_PATTERNS = [
    r"\b(no|wrong|not|incorrect|false|mistake|error|bad)\b",
    r"\b(不對|錯了|錯誤|不是|不對吧|你錯了)\b",
    r"\b(that's not|that is not|this is wrong)\b",
    r"\b(重新|再來|重做|換一個)\b",
]

_COMPLETION_PATTERNS = [
    r"\b(yes|ok|okay|done|got it|understood|copied|fine)\b",
    r"\b(好|行|可以|明白了|收到|了解)\b",
    r"\b(繼續|接著|下一步|next|continue)\b",
    r"\b(意思|就是說|也就是|所以)\b",  # 理解後的進一步提問
]

_ABANDON_PATTERNS = [
    r"\b(bye|goodbye|exit|quit|end|stop|done for now)\b",
    r"\b(再見|結束|沒事了|先這樣)\b",
]


def _check_patterns(text: str, patterns: list[str]) -> bool:
    """檢查文本是否匹配任一模式。"""
    if not text:
        return False
    text_lower = text.lower().strip()
    for pat in patterns:
        if re.search(pat, text_lower):
            return True
    return False


def compute_satisfaction(next_text: Optional[str], current_text: str) -> float:
    """
    從 follow-up 文本提取滿意度分數。

    Returns:
        float: -1.0 (強烈不滿) 到 1.0 (非常滿意)
    """
    if next_text is None or not next_text.strip():
        # 無 follow-up：可能已結束／被放棄
        if _check_patterns(current_text, _COMPLETION_PATTERNS):
            return 0.5  # 任務完成
        return -0.5  # 被放棄

    if _check_patterns(next_text, _CORRECTION_PATTERNS):
        return -1.0
    if _check_patterns(next_text, _PRAISE_PATTERNS):
        return 1.0
    if _check_patterns(next_text, _COMPLETION_PATTERNS):
        return 0.5
    if _check_patterns(next_text, _ABANDON_PATTERNS):
        return 0.0  # 中性結束
    # 正常延續
    return 0.3


def estimate_token_cost(text: str) -> float:
    """
    估算 token 消耗（近似值，用 4 字元/token 粗估）。

    Returns:
        float: 正規化後的 token 成本（0-1 之間）
    """
    char_count = len(text)
    est_tokens = max(1, char_count / 4.0)
    # 對數壓縮：短文本成本低，長文本邊際遞增
    normalized = min(1.0, est_tokens / 512.0)
    return normalized


def estimate_response_length(response_text: str) -> float:
    """
    用戶回應長度作為「續航意願」的代理指標。

    Returns:
        float: 正規化回應長度（0-1 之間）
    """
    # 排除超短回覆（可能是誤觸或跳過）
    char_count = len(response_text.strip()) if response_text else 0
    if char_count < 2:
        return 0.0
    # 對數壓縮（長的回答代表更強參與意願）
    normalized = min(1.0, char_count / 500.0)
    return normalized


def extract_external_vector(
    user_text: str,
    response_text: Optional[str],
    next_user_text: Optional[str],
) -> list[float]:
    """
    從 session 數據中提取外部向量。

    Args:
        user_text: 用戶當前的輸入
        response_text: 系統的回應（用於估算 token 成本）
        next_user_text: 用戶下一條消息（用於評估滿意度 + 續航）

    Returns:
        list[float]: [token_cost, 續航意願, 滿意度]  ℝ³
    """
    token_cost = estimate_token_cost(response_text or user_text)
    response_len = estimate_response_length(next_user_text or "")
    satisfaction = compute_satisfaction(next_user_text, user_text)

    return [token_cost, response_len, satisfaction]
