# YiCeNet 交叉記憶皮質 — Cross-Attention Memory Cortex

> 2026-06-08 · 從單輪分類器到時序注意力引擎的架構躍遷

---

## 一、問題：LOOM 的三層容器與單層注入的矛盾

LOOM 的設計包含了完整的三層架構：

```
Conversation (跨 idle/重啟) ← 飛輪訓練單元
    └── Session (技術切片)   ← 30min idle 超時
            └── Turn (原子)  ← 單輪 recommend→solidify
```

但當 LLM 需要上下文時，LOOM **只用了最薄的一層（Turn）**：

```
Turn N 的注入:
  5W1H(當前輪) + YiCeNet(當前輪) + KAFED(當前輪) → flat text → LLM
```

上兩層（Session、Conversation）近乎閒置。其後果直接反映在 API latency 數據中：

| API Call | Context | Latency | 信息密度(out/in) |
|----------|---------|---------|-----------------|
| #12      | 92K     | 2.7s    | 0.18%           |
| #21      | 101K    | 17.8s   | 1.40%           |
| #31      | 115K    | 22.7s   | 0.75%           |
| #38 (另一session)| 219K | 16.9s   | 0.62%           |

**每輪增長 ~2-3K tokens，但有效信息率 < 1.5%**。98.5% 的 token 是冗餘歷史。

---

## 二、架構躍遷：YiCeNet 作為交叉記憶皮質

### 核心轉變

```
前：YiCeNet =  || 每輪獨立的卦象分類器 
                 || 64 卦之一 → LOOM 注入
                 || 無歷史視野

後：YiCeNet =  || 跨輪記憶皮質（Memory Cortex）
                 || encoder bank + cross-attention → 上下文處方
                 || LOOM 按處方執行注入
```

YiCeNet 的角色從**預測器**變為**時序注意力引擎**——不是問「這一輪會怎樣」，而是問「從歷史中找到與當前輪真正相關的模式」。

### 三層注意力結構

```
                     ┌─────────────────────────────────┐
                     │         L3: Conversation         │
                     │  trajectory embedding → 物化知識  │
                     │  conv 關閉時 solidify 到 KAFED    │
                     └────────────┬────────────────────┘
                                  │ mean pool / autoencoder
                     ┌────────────▼────────────────────┐
                     │         L2: Session              │
                     │  trajectory = concat(q₁...qₜ)    │
                     │  384d × t → 固定維度先驗注入     │
                     └────────────┬────────────────────┘
                                  │ attention 權重
                     ┌────────────▼────────────────────┐
                     │   L1: 卦鏈注意力 (Turn-level)    │
                     │  qₜ · Kᵢ  →  softmax → 選擇性回顧│
                     │  高權重：保留摘要                 │
                     │  低權重：僅記卦象不記內容         │
                     └────────────┬────────────────────┘
                                  │
                                  ▼
                        ┌─────────────────────┐
                        │   LOOM 注入 LLM      │
                        │   卦鏈 + 相關摘要     │
                        │   + 當前上下文        │
                        └─────────────────────┘
```

### L1：卦鏈注意力（核心）

每個 Turn t 產生一個 encoder 輸出作爲 query qₜ ∈ ℝ³⁸⁴。過去 N 個 Turn 各自有 key kᵢ = TinyEncoder(msgᵢ)：

```
attentionₜ[i] = softmax(qₜ · kᵢ · KAFED_w / √384)

if attentionₜ[i] > θ_high  → Turn i 的完整摘要注入 LLM
if attentionₜ[i] < θ_low   → 跳過，只記錄卦象 ID
if θ_low ≤ attentionₜ[i] ≤ θ_high → 單行事實摘要
```

這裡的 `KAFED_w` 是 KAFED 域名相關性調製（如果當前輪查 PM，且歷史中也有 PM 相關輪次，它們的權重自動提昇）。這創造了**隨 query 變化的動態歷史窗口**——不是固定 truncation，不是單純的摘要，而是基於當前語義相關性的選擇性回顧。

**計算耗時**：純 numpy 運算，N=100 輪時 < 5ms。

### L2：Session 壓縮

整個 session 的 encoder 輸出構成 trajectory embedding：

```
Trajectory = concat(q₁, q₂, ..., qₜ) ∈ ℝ³⁸⁴·ᵗ
```

可壓縮為固定維度（如 mean pooling 或自編碼器），作為 session 級先驗注入 LLM：

```
ctx: "這場對話共 31 輪，卦鏈軌跡=[䷓,䷿,䷸,䷙,...,䷳]，
      注意力集中於輪次 #5,#12,#21（關於 KAFED Finder）"
```

這個先驗本身 ≈ 80 tokens，但讓 LLM**不需要自己從 115K tokens 中推斷「之前說了什麼」**。

### L3：Conversation 物化

Conversation 關閉時，完整 trajectory embedding + 獎勵信號 → solidify 到 KAFED。飛輪閉環：

```
Loom close → trajectory(384d×N) + reward
     → WM fine-tune（提升下一次的 attention 準確度）
     → 知識條目寫入 KAFED（跨 conversation 可檢索）
```

---

## 三、YiCeNet 返回的「上下文處方」

LOOM 的調用方式不變——每輪調用 `yicenet_predict(task_brief)`。**變的是返回內容**：

```python
# 當前返回（單輪卦象）
{
    "hexagram_id": 18,
    "hexagram_name": "臨",
    "q_value": 0.5,
    "candidates": [45, 12, 19, 16, 22, 50],
    "chain": [18],
    "display_compact": "䷒ 臨",
    "action_id": 23,
    "action_name": "multi_step_form"
}

# 未來返回（上下文處方）
{
    "hexagram": {
        "id": 18, "name": "臨", "symbol": "䷒",
        "chain": [15, 9, 18],  # 完整卦鏈
        "attention_entropy": 0.42,  # 注意力熵值
        "drift": "stable"  # 穩定/漂移/跳躍/錯卦
    },
    "context_prescription": {
        "mode": "compress",  # compress | expand | full
        "retain_turns": [5, 12, 21],  # 高 attention 輪次需保留
        "summarize_before_turn": [2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20],
        "discard_turns": [],  # 低 attention + 低信息量的輪次
        "session_trajectory": "䷓→䷿→䷸→䷙→䷳→䷒",  # session 級脈絡
        "key_insight": "attention 集中於 KAFED Finder 討論的輪次 #5, #12, #21"
    },
    "compression_ratio": 0.72,  # 可壓縮掉的 context 比例
    "action": {
        "id": 23,
        "name": "context_compress"
    }
}
```

LOOM 收到處方後，執行：

| 處方指令 | LOOM 執行 |
|---------|----------|
| `retain_turns` | 這些輪次的 tool result + 摘要原樣保留 |
| `summarize_before_turn` | 這些輪次壓縮為 1-3 行事實摘要 |
| `discard_turns` | 完全移除，僅卦鏈 ID 留在 trajectory 中 |
| `compress` mode | 總 context 壓縮後注入，保留關鍵輪次 |
| `expand` mode | 新領域/新 topic，需更多上下文 |

---

## 四、效果預測

### Context 增長對比

```
當前：每輪 +2-3K tokens → 31 輪 = 115K tokens → latency 23s
    信息密度: 0.75%

交叉注意力後：
  retain_turns: ~5 輪 × 500 tokens = 2.5K
  summaries: ~20 輪 × 100 tokens = 2K
  session_trajectory: ~80 tokens
  當前上下文: ~1K tokens
  ---------------------------------
  總注入: ~5.5K tokens ≈ 壓縮 95%

  信息密度: ~25% (假設 1300/5500)
  latency: DeepSeek API 在 5K context 下約 0.5-1.5s
```

### 增長斜率變化

| 階段 | context 增長率 | 累積到 115K 所需輪數 |
|------|---------------|--------------------|
| 當前 | ~2.5K/輪 | 46 輪 |
| 交叉注意力後 | ~0.3K/輪（僅關鍵輪摘要） | ~380 輪 |
| 當 attention 成熟後 | ~0.1K/輪 | ~1150 輪 |

### Attention 熵隨時間的預期曲線

```
H 高 (分散)  ▂▃▄▅▆▇███▇▆▅▄▃▂__________  WM 正在學習模式
                   ▂▃▄▅▆▇████▇▆▅▄▃▂___  模式已學習，注意力集中
                           ▂▃▄▅▆▇████   完全收斂
                                   → 時間
```

期望行為：session 剛開始時 attention entropy 高（WM 尚未辨識模式），隨著輪次推進熵逐漸下降（WM 學會了「哪些歷史真正重要」）。**低熵 = 高信息密度的注入**。

---

## 五、設計邊界（2026-06-08 釐清）

### 5.1 Data Explosion？—— 計算驗證

MemoryBank 存的是 **TinyEncoder 的 384d 輸出**，不是原始對話歷史。

```
每輪存儲:
  encoder_output: 384 × float32 = 1,536 bytes ≈ 1.5KB
  hexagram_id:    1 × int8 = 1 byte
  summary:        可選，短文本 ≈ 100-300 bytes

1,000 輪對話:  1.5MB（向量）+ 0.3MB（摘要）= 1.8MB
10,000 輪對話: 15MB + 3MB = 18MB
```

Cross-attention 計算量：
```
q(384) · Kᵀ(384 × 10000) = 3.84M 次乘加 → < 5ms
```

比當前每輪的 ChromaDB query（~730ms）快兩個數量級。**不是爆炸**。

### 5.2 Session 開始/結束信號——LOOM 提供

YiCeNet **不感知** session 邊界。LOOM 的 hook 系統已經提供精確的生命週期：

```
on_session_start(session_id)     → MemoryBank.init(session_id)
pre_llm_call(user_message)       → store encoder output
post_llm_call(session_id)        → mark turn complete
close_conversation(session_id)   → MemoryBank.flush()
```

MemoryBank 只暴露 `store_turn()` 和 `get_attention()` 兩個方法。何時開始、何時結束、何時清空——由 LOOM 決定。

### 5.3 與 WM + Flywheel 的關係：正交

| 組件 | 作用域 | 學什麼 | 訓練需求 | 持久性 |
|------|--------|--------|---------|--------|
| **MemoryBank** | 會話內 | 無（僅存儲） | 無 | 會話終結即清空 |
| **Cross-Attention** (L1) | 會話內 | 無（純 cosine + softmax） | **無需訓練** | 無狀態 |
| **WM** | 跨會話 | 卦象演進的普遍模式 | RL fine-tune | checkpoint 持久 |
| **Flywheel** | 跨會話 | 獎勵驅動的策略進化 | RL fine-tune | registry.json |

**L1 交叉注意力不是機器學習**——不需要訓練，不需要梯度，不需要 flywheel。

```
L1: q · Kᵀ → softmax → threshold → retain/summarize/discard
    純 numpy 運算，確定性且可解釋
```

WM 學的是 **經驗模式**（「當卦鏈出現某種軌跡時，用戶接下來通常會做什麼」）。
MemoryBank + cross-attention 做的是 **語義匹配**（「當前 query 跟之前哪一輪的 encoder 輸出最像」）。
兩者作用域不同，不衝突。

Cross-attention 的結果可作為 WM 的額外輸入特徵（Phase 2 可選），但這是增強不是依賴。

### 5.4 核心原則：記憶皮質 = 語境依賴，WM = 經驗非語境

```
MemoryBank   ← 上下文 (context) ← 會話內，易失
   ↓
   語義查詢：q · Kᵀ
   ↓
   產出：注意力分布 + 壓縮處方
   ↓
   會話結束 → 拋棄

WM           ← 經驗 (experience) ← 跨會話，持久
   ↓
   時間序列預測：trajectory → next_hexagram
   ↓
   產出：卦象演進的概率 + 驚訝度
   ↓
   flywheel → 持續改進
```

將注意力權重放進 RL 去學會汙染 WM 的跨會話經驗——**這是錯誤的設計**。
L1 交叉注意力保持為純計算，確定性且可調試。

---

## 六、WM 的協作角色（Phase 2，非核心）

WM 不參與 L1 交叉注意力。它的角色在更高的抽象層：

### 6.1 驚訝度 → 上下文策略建議

WM 的 `compute_endogenous_weight()`（已實現）可作為上下文策略的參考信號：

| WM 驚訝度 | 含義 | 對 cross-attention 的影響 |
|-----------|------|--------------------------|
| < 0.15 | 模式穩定，預測準確 | 可信任 attention 的壓縮建議 |
| 0.15-0.50 | 部分漂移，新信息湧現 | 降低 threshold，保留更多上下文 |
| > 0.50 | 模式轉變 | 切換到 expand mode，讓 LLM 看到更多上下文 |

WM **建議**策略，但不干涉 attention 的計算。

### 6.2 飛輪閉環（可選）

Cross-attention 的結果（哪些輪次被集中關注）可作為 WM 的輸入特徵，幫助 WM 預測下一輪的注意力分布。但這是 Phase 2 的可選增強，不是核心路徑。

---

## 七、LOOM 的邊界（不變的部分）

遷移到「YiCeNet 記憶皮質 + LOOM 執行層」的模式後，LOOM 的邊界保持清晰：

| LOOM 做 | YiCeNet 做 |
|---------|-----------|
| 每輪調用 `yicenet_predict()` | 存儲每輪 encoder 輸出到 memory bank |
| 接收上下文處方 | 計算 cross-attention → 生成處方 |
| 按處方壓縮/展開 messages | 維護 trajectory embedding |
| 注入結果到 LLM | 預測下一輪 attention 分布 |
| solidify 知識到 KAFED | WM 飛輪閉環 |
| conversation 管理 | 無狀態——純計算引擎 |

LOOM 每輪的 `recommend()` 內部調用保持不變，只是第 2 步「卦」返回的不再是單一卦象，而是一份上下文處方。LOOM 的任務就是執行這個處方。

---

## 八、實現路徑

### Phase 1：Memory Bank + Attention（純計算，無 RL 依賴）

```
可獨立驗證的 MVP：
  1. MemoryBank 類：存儲 {turn_id, encoder_output(384d), hexagram_id, summary}
  2. CrossAttention.compute(q_t, K, V) → attention_weights
  3. ContextPrescription(weights, threshold) → retain/summarize/discard lists
```

測試方法：對現有 session DB 中的歷史輪次跑 attention 驗證——attention 高權重的輪次是否語義相關？

### Phase 2：WM 協作（可選）

```
  4. 驚訝度參考 → 調節 threshold 和 mode
  5. Cross-attention 結果可作為 WM 特徵（非必需）
  6. 飛輪閉環
```

### Phase 3：LOOM 集成

```
  7. LOOM.pre_llm_call 改為執行 yicenet_return.context_prescription
  8. message 列表按處方壓縮後注入 LLM
  9. solidify 時寫入 attention 結果供飛輪消費
```

---

## 九、結論

YiCeNet 從卦象分類器到交叉記憶皮質的躍遷，是 LOOM 從「三層容器單層使用」到「全層級跨注意力洞察」的關鍵。不是增加新組件，而是**重新定位現有組件的角色**——TinyEncoder 的 384d 輸出一直是跨輪的，只是從未被當作時序聚類的查詢向量使用。

64 卦的離散空間不是限制，而是**注意力可解釋性的設計優勢**——不像黑盒 transformer 的 attention weights 難以調試，卦象 ID 域讓注意力模式可以直接對應到周易語義，便於人工審視和校準。

核心原則：**LOOM 的任務不是理解歷史，而是執行 YiCeNet 對歷史的注意力指引。**

---

*附：本文檔對應 2026-06-08 cross-attention 架構討論。YiCeNet SKILL.md 為系統級引用入口。*
