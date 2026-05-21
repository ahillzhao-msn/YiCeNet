# YiCeNet 终极轻量部署方案

> 设计日期：2026-05-20
> 状态：设计稿，待实现

## 部署原则

- **终极轻量**：全部组件总内存 <500MB，镜像 <500MB
- **零外部依赖**：不用 Kubernetes / Prometheus / Redis / Grafana
- **无网络开销优先**：Plan B = 进程内 C 扩展加载；兜底 Plan A = HTTP 服务
- **Hermes 原生编排**：训练流水线由 Hermes 自身 cron 驱动

---

## 一、组件架构

```
                          ╔═══════════════╗
                          ║   Hermes 引擎  ║
                          ╚═════╤═════════╝
                                │
              ┌─────────────────┼─────────────────┐
              │  Plan B (首选)  │  Plan A (兜底)   │
              ▼                 │                  ▼
   ┌──────────────────┐        │       ┌──────────────────┐
   │ YiCeNet C 扩展   │        │       │ YiCeNet HTTP 服务│
   │ (进程内 ONNX)    │        │       │ (FastAPI :8001)  │
   │ <1ms 零开销      │        │       │ ~5ms 网络开销    │
   └────────┬─────────┘        │       └────────┬─────────┘
            │                  │                 │
            └──────────────────┼─────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  metrics.db (SQLite) │
                    │  轨迹 / 奖励 / 卦象   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
   ┌────────────────┐ ┌──────────────┐ ┌────────────────┐
   │ 训练 Worker    │ │ Streamlit    │ │Hermes 编排任务 │
   │ CPU PPO 微调   │ │ 仪表盘 :8501 │ │ 每小时/500条   │
   │ 写 registry    │ │ 效能/热力图   │ │ 触发训练+评估  │
   └────────────────┘ └──────────────┘ │ +切换判决      │
                                       └────────────────┘
```

---

## 二、容器清单（Docker Compose）

### 2.1 推理服务 — Container 1

**定位**：Plan A（HTTP 服务）时独立运行；Plan B（C 扩展）时被 Hermes 直接调用，不需独立容器

**Plan A 接口**：
```
POST /v1/predict   {text, context} → {hexagram, action, q_values, candidates}
POST /v1/mutate    {hexagram, feedback} → {new_hexagram}
GET  /v1/health    → {model_a_loaded, model_b_ready, version}
POST /v1/switch    {target: "A"|"B"} → {success}
```

**Plan A 资源**：GPU ~20MB / CPU ~50MB，镜像 ~150MB

---

### 2.2 训练 Worker — Container 2

**职责**：
1. 轮询 SQLite，拉取新轨迹
2. 执行 PPO 微调（CPU only，5.6M 模型一次更新 <0.1s）
3. 产新权重 → `checkpoints/v{version}.pt`
4. 写 `registry.json`，含版本号 + 评估指标 + 胜率
5. 若新版胜率 > 旧版 + 5% → 标记为"就绪切换"

**镜像**：~150MB（与推理服务共享 base image）

**关键设计 — A/B 切换信号**：
```json
// registry.json
{
  "active": "v3",
  "ready": {"version": "v4", "win_rate": 0.73, "timestamp": "..."},
  "history": ["v1", "v2", "v3"]
}
```
推理服务或 Hermes 工具轮询此文件，发现 `ready` 则切换。

---

### 2.3 仪表盘 — Container 3

**框架**：Streamlit + Plotly，数据源 = SQLite 共享卷

**面板**：
1. **效能曲线** — reward 滑动平均、token 消耗、延迟
2. **卦象热力图** — 横轴时间，纵轴 64 卦，色深=选中频率
3. **太极罗盘** — 综合成功率 + 八卦原型雷达图
4. **A/B 胜率对比** — 切换事件时间线
5. **温度 τ 自适应曲线**

**启动命令**：
```bash
streamlit run dashboard.py --server.port 8501 --server.headless true
```

---

## 三、数据层（共享卷）

### SQLite: `metrics.db`

```sql
-- 推理轨迹
CREATE TABLE trajectories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    state_hash TEXT,         -- 256-dim state → sha256
    hexagram_id INTEGER,     -- 0-63
    candidate_values TEXT,   -- JSON array of 8 Q-values
    action_id INTEGER,
    reward REAL,
    user_continued BOOLEAN,
    latency_ms REAL,
    token_cost REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 训练事件
CREATE TABLE evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT,
    avg_reward REAL,
    win_rate REAL,
    episodes INTEGER,
    duration_sec REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 卦象使用统计
CREATE TABLE hexagram_usage (
    date TEXT,
    hexagram_id INTEGER,
    count INTEGER DEFAULT 0,
    avg_q_value REAL,
    PRIMARY KEY (date, hexagram_id)
);
```

### 模型注册: `registry.json`

```json
{
  "active": {"version": "v3", "path": "checkpoints/v3.pt", "avg_reward": 1.53},
  "ready": null,
  "fallback": {"version": "v1", "path": "checkpoints/v1.pt"},
  "history": [
    {"version": "v1", "avg_reward": 1.20, "created": "..."},
    {"version": "v2", "avg_reward": 1.45, "created": "..."},
    {"version": "v3", "avg_reward": 1.53, "created": "..."}
  ]
}
```

---

## 四、Plan B 详解：进程内推理引擎

### 4.1 方案结论

**Plan B 已验证可行，无需 C 扩展。**

经过实际测试：
- ONNX 导出因模型中的位运算（`>>`）不兼容 PyTorch 2.12 ONNX exporter
- 但**纯 PyTorch 进程内加载**完全满足 Plan B 的核心目标：零网络开销、直接函数调用

| 指标 | 实测值 |
|------|--------|
| 推理延迟 | **4.3 ms** (GPU) / ~15 ms (CPU) |
| GPU 内存 | **43 MB** |
| 额外依赖 | 无（复用 Hermes 已有的 PyTorch） |
| A/B 切换 | 已验证：切换 <1ms，不重启进程 |

### 4.2 实现

`src/yicenet_engine.py` — 单文件，~300 行：

- `YiCeNetEngine.predict(text)` → 返回卦象 + 动作 + Q 值
- `YiCeNetEngine.switch_model(path)` → 热切换权重
- 全局单例 `get_engine()` → Hermes 工具直接调用

---

## 五、Hermes 编排集成

### 5.1 工具注册

Hermes 通过 `tools/` 目录自发现工具。YiCeNet 工具已注册：

```bash
# 激活（已完成）
ln -sf ~/YiCeNet/src/hermes_tool.py ~/.hermes/hermes-agent/tools/yicenet_tool.py

# 验证
hermes tools list | grep yicenet
# → yicenet_predict  ☯   [file]
# → yicenet_switch    🔄  [file]
```

工具通过 Hermes 的 `registry.register()` 注册，随 Hermes 启动自动加载。
**只在有 checkpoint 文件时才可见**（`check_fn` 守卫）。

### 5.2 使用方式

在 Hermes session 中直接调用：

```
yicenet_predict(task_brief="search knowledge base")
→ {"hexagram_id": 35, "hexagram_name": "晋", "action_name": "route_to_service", ...}

yicenet_predict(task_brief="rigid step A→B→C", deterministic=True)
→ {"deterministic": true, ...}  # bypass Gumbel noise

yicenet_switch(checkpoint="~/YiCeNet/checkpoints/yicenet_v1.pt")
→ {"success": true, "active": "...yicenet_v1.pt"}
```

### 5.2 编排训练流水线

Hermes cron job 或 orchestrator 定期执行：

```python
# 由 Hermes cron 触发
step1: 检查 metrics.db 新轨迹数量
step2: if 新轨迹 > 500:
          curl 训练 worker → 启动 200 步 PPO
step3: 评估新模型 → 写 registry.json
step4: 通知切换器 → 若胜率达标则切换
step5: 读 SQLite → 推送到 Streamlit 数据
```

---

## 六、资源总账

| 组件 | 镜像大小 | 运行时内存 | GPU |
|------|---------|-----------|-----|
| 推理服务 (Plan A) | ~150MB | ~80MB | 可选 |
| 训练 Worker | ~150MB | ~200MB | 否 |
| Streamlit 仪表盘 | ~200MB | ~150MB | 否 |
| SQLite | 0 | ~10MB | 否 |
| **Plan A 总计** | **~500MB** | **~440MB** | **~20MB** |
| **Plan B 新增** (进程内 ONNX) | 0 | +22MB | 复用 |

---

## 七、实施进展

| Phase | 状态 | 说明 |
|-------|------|------|
| **0** — ONNX 导出验证 | ✅ | ONNX 不可行（位运算不兼容），但纯 PyTorch 进程内满足 Plan B |
| **1** — 进程内推理引擎 | ✅ | `src/yicenet_engine.py` — 4.3ms GPU, 43MB, A/B 热切换 |
| **2** — Hermes 工具注册 | ✅ | `src/hermes_tool.py` + symlink, 双工具已注册 |
| **3** — 项目迁移 | ✅ | 从 `/mnt/c/` 迁至 `~/YiCeNet/`（Linux 原生 FS，性能提升） |
| **4** — Streamlit 仪表盘 | ✅ | 三面板：效能曲线 / 卦象热力图 / 太极罗盘，数据源 SQLite |
| **5** — Hermes 编排集成 | ✅ | 双 cron 任务：每 2h 训练检查 + 每日报告 |
| **6** — A/B 热切换闭环 | ✅ | registry.json + auto-load + check_for_switch() |
| **E1** — exploration_override | ✅ | `deterministic=True` 旁路 Gumbel 噪声 |
| **E2** — 奖励信号消歧 | ✅ | terminal_type: success/abandoned/timeout，差异化惩罚 |
