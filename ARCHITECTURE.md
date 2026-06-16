# YiCeNet 易策网络 — 架构

> 版本: v15.6.1 · 参数: 5.67M · 推理: 7ms CPU
> 
> 详细设计报告: [ARCHITECTURE-REDESIGN.md](docs/ARCHITECTURE-REDESIGN.md)
> 架构图: [yicenet-architecture.html](docs/yicenet-architecture.html)

## 系统组成

```
接入层       Hermes Plugin / MCP / Claude Hooks / Python API
    ↓
EngineProvider (统一工厂)
    ↓
YiCeNetEngine — predict() / prescribe() / analyze()
    ↓
ProviderRegistry — ITokenizer · IMemoryBank · IDisplay
    ↓
YiCeNet Model — TinyEncoder(4) · GumbelRouter · ValueNetwork(3)
```

## 核心设计原则

| # | 原则 | 评分 |
|---|------|------|
| 1 | 接口优于实现 (7 ABC) | ✅ |
| 2 | 依赖注入 (provider registry) | ⚠️ model 仍 self.new |
| 3 | 职责分离 (predict/prescribe/analyze) | ✅ |
| 4 | 去重 (EngineProvider) | ✅ |
| 5 | 平台适配器 (install/) | ✅ |
| 6 | 类型契约 (TypedDict) | ✅ |

## 重构阶段

| Phase | 提交 | 改动 |
|-------|------|------|
| Phase 1 | `423a52b` | types.py + EngineProvider (消灭重复 _get_engine) |
| Phase 2 | `7c60e1a` | 6 核心 ABC (ITokenizer/IEncoder/IRouter/IValueNetwork/IMemoryBank/IEngine) |
| Phase 3 | `138591e` | ProviderRegistry + QwenTokenizerAdapter + IMemoryBank |
| Phase 4 | `07519d2` | prescribe() + analyze() 职责拆分 |
| Phase 5 | `467d22d` | install/ + tools/ + Claude Code hooks |
| Phase 5b | `796e141` | bootstrap 简化 (776→505行) |
| Display | `b957bdf` | IDisplay ABC + TerminalDisplay/JsonDisplay/SilentDisplay |

## 已知遗留

- `providers/__init__.py` 使用绝对 import (P1)
- `_compute_chain_signals()` 绕过 ProviderRegistry 使用全局 MemoryBank (P1)
- `hermes_hook.py` 加载全量 engine 仅做 prescribe (P2)
- `scripts/install-yicenet.sh` 旧 plugin.yaml 与新 HermesInstaller 冲突 (P2)
- `ProviderRegistry.default()` 每次新建实例 (P3)
- `yicenet_engine.py` probe_list 可能为 None (P0 — 修复中)
