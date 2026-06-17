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


# ── Feedback signal patterns ───────────────────────────────────────────────────
# ASCII patterns use \b word boundaries.
# CJK patterns omit \b: Chinese/Japanese have no ASCII word boundaries,
# and re.search without anchors is correct for intra-word substring matching.

_PRAISE_PATTERNS = [
    # English — affirmative / appreciative
    r"\b(good|great|perfect|excellent|amazing|wonderful|nice|awesome|brilliant|fantastic|superb|outstanding)\b",
    r"\b(well done|well said|spot on|exactly right|nailed it|love it|that works)\b",
    r"\b(thanks|thank you|ty|thx|cheers|appreciate|appreciated|that helped|very helpful)\b",
    # Traditional Chinese
    r"(正確|對|完美|讚|厲害|不錯|真棒|太棒了|很好|非常好|完全正確|就是這樣|你說得對|說得好)",
    r"(謝謝|感謝|多謝|謝了|感謝你|謝謝你|辛苦了|辛苦)",
    # Simplified Chinese
    r"(正确|完美|赞|厉害|不错|真棒|太棒了|很好|非常好|完全正确|就是这样|你说得对|说得好)",
    r"(谢谢|感谢|多谢|谢了|感谢你|谢谢你|辛苦了|辛苦)",
    # Colloquial (works for both scripts)
    r"(太好了|好棒|牛逼|牛|6啊|666|哈哈好|对对|好的好的)",
]

_CORRECTION_PATTERNS = [
    # English — explicit negation / correction
    r"\b(no|wrong|incorrect|false|mistake|error|nope|nah|not right|not quite)\b",
    r"\b(that's not|that is not|this is wrong|you're wrong|you are wrong|actually no|hold on|wait no)\b",
    r"\b(redo|retry|try again|fix this|fix it|do it again|start over)\b",
    # Traditional Chinese
    r"(不對|錯了|錯誤|不是|不對吧|你錯了|說錯了|理解錯了|不是這樣|不對啊)",
    r"(重新|再來|重做|換一個|重試|再試一次|再改改)",
    # Simplified Chinese
    r"(不对|错了|错误|不是|不对吧|你错了|说错了|理解错了|不是这样|不对啊)",
    r"(重新|再来|重做|换一个|重试|再试一次|再改改|不对不对)",
]

_COMPLETION_PATTERNS = [
    # English — acknowledgment / continuation
    r"\b(yes|ok|okay|done|got it|understood|copied|fine|got|sure|agreed|correct|right|clear|makes sense|i see)\b",
    r"\b(continue|next|proceed|go on|go ahead|move on|keep going|carry on)\b",
    # Traditional/Simplified Chinese acknowledgment (same characters for both)
    r"(好|行|可以|明白|明白了|收到|了解|懂了|知道了|嗯|嗯嗯|对对|知道)",
    r"(繼續|接著|下一步|好吧|没问题|沒問題|继续|好的)",
    r"(意思|就是說|也就是|所以|就是说|也就是说)",  # inferential follow-up signals understanding
]

_ABANDON_PATTERNS = [
    # English
    r"\b(bye|goodbye|exit|quit|end|stop here|that's all|that's it|all done|no more|enough|i'm done|that will do|cya)\b",
    # Traditional Chinese
    r"(再見|結束|沒事了|先這樣|好了|算了|不用了|到此為止|先這樣吧|就這樣)",
    # Simplified Chinese
    r"(再见|结束|没事了|先这样|好了|算了|不用了|到此为止|先这样吧|就这样)",
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
