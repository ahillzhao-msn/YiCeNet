# YiCeNet 易策网络 — 架构

> 版本: v16.0.0 · 参数: 5.67M · 推理: 7ms CPU
>
> 详细设计报告: [ARCHITECTURE-REDESIGN.md](docs/ARCHITECTURE-REDESIGN.md)
> 架构图: [yicenet-architecture.html](docs/yicenet-architecture.html)

## 系统组成

```
接入层       Hermes Plugin / MCP / Claude Code Hooks / Python API
    ↓
hook_engine  PlatformAdapter · HookOrchestrator · FeedbackSignals · extract_feedback
    ↓
EngineProvider (统一工厂)
    ↓
YiCeNetEngine — predict() / prescribe() / analyze()
    ↓
ProviderRegistry — ITokenizer · IMemoryBank · IDisplay
    ↓
MemoryBank ←→ FileBackend (WAL JSONL, per-session, cross-process)
    ↓
YiCeNet Model — TinyEncoder(4) · GumbelRouter · ValueNetwork(3)
    ↓
飞轮          flywheel_buffer.jsonl → flywheel_run() → RL fine-tune
```

## 反馈闭环时序

```
Turn N:  UserPromptSubmit → before_prediction()  [extract Turn N-1 feedback]
                         → engine.predict()       [store TurnRecord in MemoryBank]
         Stop            → on_turn_complete()     [write response_snippet to FileBackend]
                         → flush_session()        [clear memory, keep file]

Turn N+1: UserPromptSubmit → before_prediction()
                          → get_last_turn()       [reload from FileBackend]
                          → extract_feedback()    [corrected/praised/abandoned from prompt]
                          → submit_trajectory()   [write to flywheel_buffer.jsonl]
```

反馈信号比较（修复前后）：

| producer | corr% | prsd% | abnd% | avg_sat | avg_tc |
|---|---|---|---|---|---|
| loom-hooks（旧） | 0.0 | 0.0 | 0.0 | +0.69 | 2,304,324 ← 畸形 |
| hermes（新） | 3.5 | ~0 | 14.0 | +0.23 | 0.127 ← 正常 |

## 核心设计原则

| # | 原则 | 评分 |
|---|------|------|
| 1 | 接口优于实现 (Protocol / ABC) | ✅ |
| 2 | 依赖注入 (ProviderRegistry) | ⚠️ model 仍 self.new |
| 3 | 职责分离 (predict/prescribe/analyze) | ✅ |
| 4 | 去重 (EngineProvider) | ✅ |
| 5 | 平台适配器 (PlatformAdapter Protocol) | ✅ |
| 6 | 类型契约 (TypedDict / dataclass) | ✅ |
| 7 | 持久化与业务分离 (FileBackend ⊂ MemoryBank) | ✅ |

## 重构阶段

| Phase | 提交 | 改动 |
|-------|------|------|
| Phase 1 | `423a52b` | types.py + EngineProvider |
| Phase 2 | `7c60e1a` | 6 核心 ABC |
| Phase 3 | `138591e` | ProviderRegistry + QwenTokenizerAdapter |
| Phase 4 | `07519d2` | prescribe() + analyze() 职责拆分 |
| Phase 5 | `467d22d` | install/ + tools/ + Claude Code hooks |
| Phase 5b | `796e141` | bootstrap 简化 |
| Display | `b957bdf` | IDisplay ABC |
| **v16.0** | `29d795d`–`4189065` | hook_engine · FileBackend WAL · 真实反馈信号 |

## 已知遗留

- `providers/__init__.py` 使用绝对 import (P1)
- `_compute_chain_signals()` 绕过 ProviderRegistry 使用全局 MemoryBank (P1)
- `ProviderRegistry.default()` 每次新建实例 (P3)
- `yicenet_engine.py` probe_list 可能为 None (P0)
