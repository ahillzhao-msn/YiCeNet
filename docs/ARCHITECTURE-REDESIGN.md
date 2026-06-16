# YiCeNet 易策网络 — 架构重构技术设计报告

版本: v1.1 · 日期: 2026-06-16 · 基于: YiCeNet v15.6.1 (~5.67M params)

> **状态: 重构全部完成并推 GH 🎯**
>
> Phase 1–5 + display 架构 8 个 commit 已全部推至 GitHub。
> 45 测试通过。7 个 ABC 接口。3 个 display 后端。
> 遗留 5 个 P1–P3 问题后续迭代。

---

## 前言: 本次设计审计的方法论与原则

### 为什么做这次审计

本次审计源于一个实际运行问题和一个架构追问:

1. **Claude Code MCP 接口卡死** — 首次调用 `yicenet_predict()` 耗时 26 秒，tokenizer 冷启动阻塞在 HF Hub 网络层。修复过程引出了更深层的问题: MCP 作为 pull 接口是否适合 YiCeNet 的 push 式上下文注入场景？

2. **"真的需要 MCP 吗？"** — 用户指出 Claude Code 的 PreMessageSend hook 可以实现 Hermes pre_llm_call 等同的自动注入模式，MCP 子进程 + stdio 协议是过度工程。

从这两个原点出发，审计逐步扩展到: 配置边界（LOOM config 污染）、显示模式 API 契约、组件耦合程度、接口抽象层级、全局单例模式、扩展机制缺失等全系统架构问题。

### 审计方法

审计覆盖 `src/yicenet/` 全部 30 个源文件，基于 **8 维评估框架**:

| 维度 | 考察内容 | 量化方式 |
|------|---------|---------|
| **核心封装** | 每个类/函数的职责数；是否违反 SRP | 函数行数、职责计数 |
| **内部接口** | 返回值是否有类型契约；隐性参数语义 | TypedDict 使用率、docstring 完备度 |
| **逻辑耦合** | 调用链是否硬编码；能否替换组件 | import 图扇出数、`new` 语句计数 |
| **接口边界** | 有无抽象层隔离；重复代码 | ABC 数量、重复代码段行数 |
| **外部接口** | API 是否扁平；参数是否合理 | 函数签名参数数、副作用标志位 |
| **扩展模式** | 是否支持多态替换 | ABC/Protocol/DI 的覆盖率 |
| **跨系统隔离** | 平台相关代码是否分离 | platform.system() 散布密度 |
| **共享模式** | 公共类型是否集中 | 散落 vs 聚合的 dataclass 分布 |

每项评估用 A-D 评分，结合代码行级证据（行数/引用数/import 图）。

### 审计过程中确立的设计原则

1. **接口优于实现** — 核心组件通过 ABC 声明契约，实现可替换。DataSource 是唯一成功的示范，需推广到全系统。

2. **依赖注入而非 self.new** — 构造时接收组件接口，运行时通过注册表组合。不写 `self._model = YiCeNet()`。

3. **职责分离: 一个方法做一件事** — predict() 做预测，prescribe() 做上下文处方，analyze() 做环境分析。不通过 `return_prescription` 标志位在同一方法内注入副作用。

4. **一处定义，多处引用** — 重复 >10 行即提取共享模块。hermes_tool 和 mcp_server 的 `_get_engine()` 是反面案例。

5. **平台无关核心 + 平台适配器** — 核心不感知平台细节。bootstrap.py 的 Installer ABC 模式是正确方向。

6. **类型即契约** — 不使用裸 dict 作为公共接口返回值。TypedDict / dataclass 声明字段的存在性与类型。

### 审计范围

- 覆盖: `src/yicenet/*.py` 全部 30 文件
- 保持不动: hexagram.py, display.py, env_context.py, cross_attention.py, probes.py, datasource/*, world_model.py, rl_train.py, metrics.py（B+ 以上）
- 审计时间: 2026-06-16
- 基线版本: YiCeNet v15.6.0

---

## 第一部分: 背景与动机

### 1.1 问题域

YiCeNet 是一个 I-Ching 启发的小型神经网络 (5.6M 参数, 22MB FP32)，用于对话式导航的结构化推理。它通过 Gumbel-Softmax 路由将用户意图映射到 64 个卦象结构原型，经错综互变生成 8 个候选，由值网络打分，最后由 ActionDecoder 输出 50 个行为原语中的一个。

### 1.2 当前架构问题

基于对全部 30 个源文件的代码级审计，整体评分 C+（"局部干净，整体控制器"），主要问题:

| 维度 | 分数 | 核心问题 |
|------|------|---------|
| 核心封装 | C | predict() 237 行 7 职责; bootstrap.py 920 行 15 职责; YiCeNetConfig 三组合一 |
| 内部接口 | C | 所有核心返回值是裸 dict; 无 TypedDict/TypedReturn; session_id="" 隐含语义 |
| 逻辑耦合 | C | God Class 模式; 7 个硬编码依赖; 零依赖注入; 8 个全局单例不一致 |
| 接口边界 | C | MemoryBank 无 ABC 可替换; hermes_tool+mcp_server 95% 重复 _get_engine() |
| 外部接口 | B | 扁平简约但 return_prescription 是副作用标志 |
| 扩展模式 | D | 仅 1 处 ABC (ProbeExtractor); DataSource 是最佳但唯一的成功案例 |
| 跨系统隔离 | B | bootstrap.py 混写所有平台逻辑 |
| 共享模式 | B | 无 types.py; Prescription/TurnRecord/Sample 散落各模块 |

---

## 第二部分: 已完成修复（前序工作）

在重构开始前已处理以下问题:

### 2.1 配置边界清理

**问题**: YiCeNet 的 config.yaml 含有 `loom: hexagram_mode: bus_stop` 段，从未被 YiCeNet 代码读取。

**解决**: 
- 从 `~/.yicenet/config.yaml` 移除 `loom:` 段
- 在 `~/LOOM/loom.yaml` 新增 `display.hexagram_mode: bus_stop` — 显示配置属于 LOOM 显示层
- 从 yicenet/yicenet-workflow SKILL.md 清理 loom 引用
- 文件: `~/.yicenet/config.yaml`, `~/LOOM/loom.yaml`, SKILL.md

### 2.2 显示模式 API 契约

**问题**: `format_prediction()` 的 mode 参数与 config 名称不一致（config 用 `bus_stop`，代码用 `compact`）。

**解决**: 
- 移除 `bus_stop` 别名，统一使用 `compact` | `detailed`
- compact 格式从 `䷳ 艮` 改为 `[䷳ 艮] * 亨无咎利贞`（方括号引用 + 卦辞简判）
- 新增 `hexagram_judgment()` 函数和 64 卦完整卦辞表
- LOOM 调用 `format_prediction(result, mode="compact")` 显式传参，不读 YiCeNet config
- 文件: `display.py`, `config.py`, `~/.yicenet/config.yaml`

### 2.3 Hermes Plugin 冷启动修复

**问题**: Hermes plugin 启动时未设置 `TRANSFORMERS_OFFLINE`，导致 tokenizer 首次加载阻塞在 HF Hub 网络层。

**解决**: 
- Plugin `__init__.py` 在 import 时加载 `~/.yicenet/config.yaml` runtime section
- 设置 `TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`, `TQDM_DISABLE=1`
- Flywheel buffer 路径改用 `yicenet_data_dir()` 而非硬编码 `~/.hermes/data/yicenet/`
- 文件: `~/.hermes/plugins/yicenet-hooks/__init__.py`

### 2.4 Tokenizer 本地缓存

**问题**: Qwen2.5-0.5B tokenizer 每次首次调用 `AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)` 可能联网，导致 26s 冷启动阻塞。

**解决**: 
- 新增 `download_tokenizer()` 通过 huggingface_hub 下载 5 个文件到 `~/.yicenet/tokenizer/qwen2.5-0.5b/`
- `_get_qwen_tokenizer()` 优先从本地路径加载 (trust_remote_code=False, 零联网)
- 回退到 HF Hub (TRANSFORMERS_OFFLINE 保护下)
- Bootstrap Phase 4b 自动下载
- 导出 `download_tokenizer()` 和 `tokenizer_available()` 为公共 API
- 文件: `tokenizer.py`, `bootstrap.py`, `__init__.py`

### 2.5 脚本更名

**问题**: `install-yicenet-hooks.sh` 安装的是 Hermes 插件而非 YiCeNet 自身。

**解决**: 更名为 `install-hermes-hooks.sh`，工作流和发布资产同步更新。
- 文件: `scripts/install/install-hermes-hooks.sh`, `.github/workflows/build-release.yml`

---

## 第三部分: 架构问题深度分析

### 3.1 耦合中心: yicenet_engine.py

当前 `predict()` 方法的数据流（237 行）:

```
predict(text, τ, det, session_id, return_prescription, env)
  → tokenizer.encode()           # tokenizer.py
  → _compute_chain_signals()     # memory_bank.py
  → build_env_vec()              # env_context.py
  → model.encode_context()       # model.py
  → model.router.divine()        # model → router
  → evaluate_candidates()        # model + hexagram
  → decode_action()              # decoder
  → extract_probes_tensor()      # probes + constants
  → compute_env_confidence()     # env_context + probes
  → [if prescription]            # memory + cross_attention + ContextPrescription
    memory.store_turn()
    CrossAttention.compute()
    ContextPrescription.generate()
```

7 个不同子系统的调用链全部硬编码在同一个方法体内。**没有接口边界，没有注入，没有单元测试隔离。**

### 3.2 重复模式: hermes_tool.py vs mcp_server.py

两个文件各自维护 `_get_engine()`:

```python
# hermes_tool.py (12 行)
def _get_engine():
    global _engine
    if _engine is None:
        checkpoint_dir = yicenet_checkpoint_dir()
        reg = json.loads(reg_path.read_text())
        ckpt = f"{checkpoint_dir}/{reg['active']['path']}"
        _engine = YiCeNetEngine(checkpoint=ckpt, project_root=str(yicenet_home()))
    return _engine

# mcp_server.py (12 行，几乎相同)
def _get_engine():
    global _engine
    if _engine is None:
        checkpoint_dir = yicenet_checkpoint_dir()
        reg = json.loads(reg_path.read_text())
        ckpt = active.get("path") or glob("*_v*")[-1]
        _engine = YiCeNetEngine(checkpoint=ckpt, project_root=str(yicenet_home()))
    return _engine
```

**95% 重复。新增一个平台（如 VSCode extension）将产生第三份副本。**

### 3.3 单例泛滥

| 模块 | 单例对象 | 实现方式 |
|------|---------|---------|
| memory_bank.py | MemoryBank | `_instance = None` + `get_memory_bank()` |
| tokenizer.py | _TOK | `_TOK = None` + `_get_qwen_tokenizer()` |
| probes.py | _extractor | `_extractor = None` + `get_extractor()` |
| config.py | _config + _user_config | `_config = None` + `get_config()` |
| constants.py | PrecomputedTables | 模块级全局 |
| yicenet_engine.py | _engine_cache | `_engine_cache = {}` |
| hermes_tool.py | _engine | `_engine = None` |
| mcp_server.py | _engine | `_engine = None` |

**8 处全局可变异步状态。测试时无法隔离，环境变量污染被各测试共享。**

### 3.4 抽象不足

全系统仅 1 处使用 ABC（ProbeExtractor）。DataSource 是唯一的成功扩展模式:

```python
# datasource/__init__.py — 正确的示范
class DataSource(ABC):
    @abstractmethod
    def scan_since(self, timestamp) -> list[Sample]: ...

class HermesDataSource(DataSource): ...
class ClaudeCodeDataSource(DataSource): ...
class FlywheelBufferSource(DataSource): ...
```

flywheel 通过 `scan_all_sources()` 多态调用，新增平台只需添加 DataSource 实现。**但这个模式只在 datasource 中存在，核心架构未被复用。**

---

## 第四部分: 目标架构设计

### 4.1 核心原则

1. **接口优于实现** — 每个核心组件通过 ABC 声明契约
2. **依赖注入而非 self.new** — Engine 构造时接收所有组件
3. **职责分离** — predict/prescribe/analyze 拆为独立方法
4. **一处定义，多处引用** — 重复代码提取为共享模块
5. **平台无关核心 + 平台适配器** — bootstrap / install 抽象化

### 4.2 新文件结构

```
src/yicenet/
├── engine.py                    # 重构: YiCeNetEngine (predict+prescribe+analyze)
├── engine_provider.py           # NEW: 统一 engine 工厂 (消灭重复 _get_engine)
├── interfaces.py                # 扩展: 6 ABC (ITokenizer, IEncoder, IRouter, IValueNetwork, IMemoryBank, IEngine)
├── types.py                     # NEW: PredictionResult TypedDict + Prescription + TurnRecord + Sample
├── config.py                    # 拆分: ModelArchConfig + TrainingConfig + PathConfig
├── providers/                   # NEW: 注册表模式
│   └── __init__.py              # ProviderRegistry (default + override)
├── install/                     # NEW: 平台适配器
│   ├── base.py                  # PlatformInstaller ABC
│   ├── hermes.py                # HermesInstaller
│   └── claude.py                # ClaudeCodeInstaller
├── (保持不动)
│   ├── model.py                 # YiCeNet 模型骨架
│   ├── hexagram.py              # 纯函数卦象推理
│   ├── display.py               # 格式化层
│   ├── tokenizer.py             # Qwen BPE (实现 ITokenizer)
│   ├── memory_bank.py           # 会话记忆 (实现 IMemoryBank)
│   ├── cross_attention.py       # numpy 注意力
│   ├── env_context.py           # 16-dim 环境向量
│   ├── probes.py                # 9 探针系统
│   ├── flywheel.py              # 飞轮训练
│   ├── datasource/*             # DataSource ABC (保持不变)
│   ├── world_model.py           # 双头世界模型
│   ├── rl_train.py              # RL v5 微调
│   └── metrics.py               # SQLite 日志
├── hermes_tool.py               # 简化: 调 EngineProvider.get_engine()
├── mcp_server.py                # 简化: 同上
└── bootstrap.py                 # 简化: 编排器
```

### 4.3 类型系统

```python
# types.py — 所有公共类型聚合

class PredictionResult(TypedDict):
    """predict() 的契约式返回。调用方看着这个类就知道所有 key。"""
    hexagram_id: int                # 0-63
    hexagram_name: str
    hexagram_number: int            # 1-64
    hexagram_pattern: int           # 6-bit int
    selected_hexagram_id: int       # always set
    selected_hexagram_name: str
    candidates: list[dict]          # [{hexagram_id, hexagram_name, q_value}]
    action_id: int                  # always set
    action_name: str
    q_values: list[float]           # always 8 elements
    probes: list[float]             # always 9 elements
    env_confidence: float
    context_status: str             # "sufficient"|"partial"|"thin"
    context_hint: str

@dataclass
class Prescription:
    retain_turns: list[int]
    summarize_turns: list[int]
    discard_turns: list[int]
    mode: str                       # "expand" | "compress"
    attention_entropy: float
    compression_ratio: float
    key_insight: str

@dataclass
class TurnRecord:
    turn_id: int
    encoder_output: np.ndarray      # (384,)
    hexagram_id: int
    summary: str
    timestamp: float

@dataclass
class Sample:
    conversation_id: str
    source: str
    user_text: str
    assistant_text: str
    next_user_text: str | None
    signals: dict[str, float]
    embedding: list[float] | None
```

### 4.4 接口层

```python
# interfaces.py — 6 个核心接口

class ITokenizer(ABC):
    """可替换的 tokenizer。默认 Qwen2.5-0.5B BPE。"""
    @abstractmethod
    def encode(self, text: str, max_len: int = 128) -> tuple[Tensor, Tensor]: ...
    @abstractmethod
    def get_vocab_size(self) -> int: ...
    @abstractmethod
    def download(self, hf_token: str = "") -> bool: ...

class IEncoder(ABC):
    """可替换的编码器。默认 TinyEncoder (4 层 Transformer)。"""
    @abstractmethod
    def encode_context(self, input_ids: Tensor, mask: Tensor, env_vec: Tensor | None = None) -> Tensor: ...

class IRouter(ABC):
    """可替换的路由策略。默认 Gumbel-Softmax Router。"""
    @abstractmethod
    def divine(self, h: Tensor, tau: float = 1.0, hard: bool = False) -> tuple[Tensor, Tensor]: ...

class IValueNetwork(ABC):
    """可替换的值网络。默认 3-layer MLP (256→128→64→1)。"""
    @abstractmethod
    def score(self, candidate_embeds: Tensor) -> Tensor: ...

class IMemoryBank(ABC):
    """可注入的短期记忆。不再全局单例。"""
    @abstractmethod
    def init_session(self, session_id: str): ...
    @abstractmethod
    def store_turn(self, session_id: str, turn_id: int, encoder_output: np.ndarray, hexagram_id: int, summary: str = ""): ...
    @abstractmethod
    def get_session_keys(self, session_id: str) -> tuple[np.ndarray, list[dict]]: ...

class IEngine(ABC):
    """核心推理引擎（门面模式，非 God Class）。"""
    @abstractmethod
    def predict(self, task_brief: str, temperature: float = 0.1, deterministic: bool = False, environment: dict | None = None) -> PredictionResult: ...
    @abstractmethod
    def prescribe(self, task_text: str, session_id: str, turn_id: int = 0, turn_summary: str = "") -> Prescription: ...
    @abstractmethod
    def analyze(self, task_brief: str) -> EnvAnalysis: ...
    @abstractmethod
    def switch_model(self, checkpoint: str) -> bool: ...
```

### 4.5 Engine 重构

```python
# engine.py — predict/prescribe/analyze 职责分离

class YiCeNetEngine:
    """构造时注入所有依赖。"""

    def __init__(
        self,
        model: YiCeNet,                     # 模型骨架
        tokenizer: ITokenizer,              # 注入 tokenizer
        memory_bank: IMemoryBank,           # 注入 memory（非单例）
        encoder: IEncoder | None = None,    # 可替换 encoder
        router: IRouter | None = None,      # 可替换 router
        value_net: IValueNetwork | None = None,  # 可替换 value net
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._memory = memory_bank
        self._encoder = encoder or model.encoder
        self._router = router or model.router
        self._value_net = value_net or model.value_net
        self._prev_hx = None

    def predict(self, task_brief: str, temperature: float = 0.1,
                deterministic: bool = False,
                environment: dict | None = None) -> PredictionResult:
        """纯推理：编码 → 路由 → 候选评估 → 动作 → 探针。约 7ms。"""
        # ① 编码
        input_ids, mask = self._tokenizer.encode(task_brief)
        env_vec = build_env_vec(environment)
        h = self._encoder.encode_context(input_ids, mask, env_vec)

        # ② 路由
        hex_idx, probs = self._router.divine(h, tau=temperature, hard=deterministic)

        # ③ 候选评估 (错综互变 → 值网络)
        cand_idxs = generate_candidates(hex_idx.item())
        cand_embeds = self._model.hexagram_embed(cand_idxs)
        cand_values = self._value_net.score(cand_embeds)
        best_idx = cand_values.argmax(dim=-1)
        best_hex_id = cand_idxs.gather(1, best_idx)

        # ④ 动作解码
        action_ids = self._model.decode_action(best_hex_id)

        # ⑤ 探针
        probes = extract_probes_tensor(h, ..., hex_idx, self._prev_hx, ...)
        self._prev_hx = h

        # ⑥ 信心评估
        confidence, status, hint = compute_env_confidence(probes.tolist(), cand_values[0].tolist())

        return PredictionResult(
            hexagram_id=hex_idx.item(),
            hexagram_name=HEXAGRAM_NAMES[hex_idx.item()],
            selected_hexagram_id=best_hex_id.item(),
            selected_hexagram_name=HEXAGRAM_NAMES[best_hex_id.item()],
            candidates=[...],
            action_id=action_ids.item(),
            action_name=ACTION_NAMES[action_ids.item()],
            q_values=cand_values[0].tolist(),
            probes=probes.tolist(),
            env_confidence=confidence,
            context_status=status,
            context_hint=hint,
        )

    def prescribe(self, task_text: str, session_id: str,
                  turn_id: int = 0, turn_summary: str = "") -> Prescription:
        """上下文处方 = predict + memory + cross-attention。
           调用者显式调用，不再通过 return_prescription 标志位。"""
        # predict 复用作编码
        self._memory.init_session(session_id)
        result = self.predict(task_text)
        h = ...  # 复用上次编码结果或重新编码
        self._memory.store_turn(session_id, turn_id, h, result["hexagram_id"], turn_summary)
        keys, meta = self._memory.get_session_keys(session_id)
        weights = CrossAttention.compute(h.squeeze(), keys)
        return ContextPrescription.generate(weights, meta, result["probes"])

    def analyze(self, task_brief: str) -> EnvAnalysis:
        """快速环境分析（~3ms）。仅编码+探针，不走路由。"""
        input_ids, mask = self._tokenizer.encode(task_brief)
        h = self._encoder.encode_context(input_ids, mask)
        probes = extract_probes_tensor(h, ...)
        return EnvAnalysis(h=h, probes=probes, confidence=...)
```

### 4.6 EngineProvider: 解决重复

```python
# engine_provider.py — 统一 engine 工厂

class EngineProvider:
    """一处写死，多处复用。取代 hermes_tool._get_engine() + mcp_server._get_engine()。"""

    _engine: IEngine | None = None
    _active_version: str = ""

    @classmethod
    def get_engine(cls) -> IEngine:
        if cls._engine is None:
            cls._engine = cls._build_engine()
        return cls._engine

    @classmethod
    def check_switch(cls) -> bool:
        """注册表检查：ready 是否优于 active + 3% → 热切换。"""
        reg = load_registry()
        if reg.get("ready") and reg["ready"]["win_rate"] >= reg["active"]["win_rate"] + 0.03:
            cls._engine = cls._build_engine(checkpoint=reg["ready"]["path"])
            return True
        return False

    @classmethod
    def _build_engine(cls, checkpoint: str = "") -> IEngine:
        ckpt_path, ver = resolve_checkpoint(checkpoint)
        model = YiCeNet.from_pretrained(ckpt_path)
        registry = ProviderRegistry.default()
        return YiCeNetEngine(
            model=model,
            tokenizer=registry.tokenizer,
            memory_bank=registry.memory,
            encoder=registry.encoder,
            router=registry.router,
            value_net=registry.value_net,
        )

# 替换后:
# hermes_tool.py:  engine = EngineProvider.get_engine()
# mcp_server.py:  engine = EngineProvider.get_engine()
# 新增平台:       engine = EngineProvider.get_engine()
```

### 4.7 ProviderRegistry: 组件注册表

```python
# providers/__init__.py

class ProviderRegistry:
    """单例，但可被测试代码干净地覆盖。"""

    tokenizer: ITokenizer
    encoder: IEncoder
    router: IRouter
    value_net: IValueNetwork
    memory: IMemoryBank

    @classmethod
    def default(cls) -> "ProviderRegistry":
        return ProviderRegistry(
            tokenizer=QwenTokenizer(),      # 实现 ITokenizer
            encoder=TinyEncoder(),           # 实现 IEncoder
            router=GumbelRouter(),           # 实现 IRouter
            value_net=ValueNetwork(),        # 实现 IValueNetwork
            memory=MemoryBank(),             # 实现 IMemoryBank
        )

    @classmethod
    def override(cls, **kwargs) -> "ProviderRegistry":
        """测试/离线用。替换指定组件，其余保持默认。"""
        base = cls.default()
        for k, v in kwargs.items():
            setattr(base, k, v)
        return base

# 测试:
registry = ProviderRegistry.override(
    tokenizer=MockTokenizer(),              # 不加载 22MB checkpoint
    memory=MockMemoryBank(),                # 不依赖 ~/.hermes/state.db
)
engine = YiCeNetEngine(model, registry.tokenizer, registry.memory, ...)
```

### 4.8 Config 拆分

```python
# config.py — 一拆三

@dataclass
class ModelArchConfig:        # 模型初始化用
    vocab_size: int = 8000
    hidden_dim: int = 256
    intermediate_dim: int = 1024
    num_heads: int = 4
    num_encoder_layers: int = 4
    max_seq_len: int = 128
    dropout: float = 0.1
    num_trigrams: int = 8
    num_hexagrams: int = 64
    num_actions: int = 50
    probe_dim: int = 9
    value_hidden: int = 128
    wm_shared_dim: int = 128
    wm_hexagram_head_dim: int = 64
    num_external_metrics: int = 3
    hexagram_patterns: tuple = field(default_factory=lambda: tuple(_king_wen_hexagrams()))

@dataclass
class TrainingConfig:         # 训练时用
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    max_epochs: int = 100
    clip_grad_norm: float = 1.0
    ppo_clip_epsilon: float = 0.2
    ppo_epochs: int = 4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    gumbel_tau_init: float = 1.0
    gumbel_tau_min: float = 0.1
    gumbel_tau_decay: float = 0.995
    wm_beta: float = 0.3
    wm_slow_tau_days: float = 30.0
    wm_fast_tau_days: float = 3.0
    wm_alpha: float = 1.5

@dataclass
class PathConfig:             # 路径解析用
    home: Path = field(default_factory=lambda: _resolve_home())
    data_dir: Path = None
    checkpoint_dir: Path = None
    log_dir: Path = None

    def _resolve_home(self) -> Path:
        """三阶路径: env var > editable install source > ~/.yicenet/"""
        if env_val := os.getenv("YICENET_HOME"):
            return Path(env_val)
        if _is_editable_install():
            return PROJECT_ROOT
        return Path.home() / ".yicenet"

# 使用:
model = YiCeNet(ModelArchConfig())          # 模型初始化
trainer = RLTrainer(model, TrainingConfig())  # 训练
paths = PathConfig()                          # 路径
paths.data_dir / "flywheel_buffer.jsonl"      # 构建路径
```

### 4.9 Install 平台化

```python
# install/base.py

class PlatformInstaller(ABC):
    """每个平台一个实现。"""

    @abstractmethod
    def detect(self) -> bool: ...         # 检测平台是否存在
    @abstractmethod
    def install_package(self) -> bool: ... # pip install 到该平台 venv
    @abstractmethod
    def register_hooks(self): ...         # 注册 lifecycle hooks
    @abstractmethod
    def unregister(self): ...             # 清理痕迹

# install/hermes.py
class HermesInstaller(PlatformInstaller):
    def detect(self) -> bool: ...         # which hermes
    def install_package(self) -> bool: ... # pip install into Hermes venv
    def register_hooks(self): ...
        # writer ~/.hermes/plugins/yicenet-hooks/plugin.yaml
        # writer ~/.hermes/plugins/yicenet-hooks/__init__.py (env + hooks)
    def unregister(self): ...
        # rm -rf ~/.hermes/plugins/yicenet-hooks/

# install/claude.py
class ClaudeCodeInstaller(PlatformInstaller):
    def detect(self) -> bool: ...         # which claude
    def install_package(self) -> bool: ... # pip install
    def register_hooks(self): ...
        # PreMessageSend: 注入 hexagram 上下文
        # PostToolUse: flywheel reward
        # 移除 MCP server 配置
    def unregister(self): ...
        # rm from ~/.claude/settings.json

# bootstrap.py (简化版 — 仅编排)
def main():
    init_data_root()
    ensure_checkpoints()
    if not tokenizer_available():
        download_tokenizer()
    target = detect_target()
    installers = {"hermes": HermesInstaller, "claude-code": ClaudeCodeInstaller}
    for name, cls in installers.items():
        if target == name or (target == "auto" and cls.detect()):
            cls().install_package()
            cls().register_hooks()
    register_scheduler()
```

---

## 第五部分: 外部系统接口规范

### 5.1 Hermes Plugin (保持)

使用 Hermes 原生 lifecycle hooks。YiCeNet 通过 plugin 注入，Hermes 不反向依赖 YiCeNet:

| Hook | Hermes 侧 | YiCeNet 侧 |
|------|----------|------------|
| pre_llm_call | 注入 hexagram 上下文 | format_prediction(engine.predict(), mode) |
| post_tool_call | 记录工具调用 | MemoryBank.store_turn() |
| post_llm_call | 飞轮 reward | submit_trajectory() |

### 5.2 Claude Code (移除 MCP，改用 Hooks)

**当前问题**: MCP stdio 子进程冷启动 26s，Claude Code 超时。

**替代方案**（已验证 Hermes 等同模式）:

| Claude Hook | 等同 Hermes | 注入内容 |
|-------------|-------------|---------|
| PreMessageSend | pre_llm_call | hexagram 上下文 (engine.predict()) |
| PostToolUse | post_tool_call | flywheel reward |
| Stop | — | MemoryBank flush |

**PreMessageSend Hook 伪代码**:

```python
# Claude settings.json
{
  "hooks": {
    "PreMessageSend": [{
      "matcher": "",
      "hooks": [{
        "type": "command",
        "command": "python3 -c \"
import os; os.environ['YICENET_HOME']='~/.yicenet'
from yicenet.tools.claude_hook import pre_message_send
pre_message_send()
\""
      }]
    }],
    "PostToolUse": [{
      "matcher": "",
      "hooks": [{
        "type": "command",
        "command": "...flywheel reward..."
      }]
    }]
  }
}
```

**优势**: 
- Engine 进程启动时预热（非每次调用冷启动）
- 自动注入，Claude 不需要显式调用 MCP 工具
- 消除 stdio 子进程 26s 延迟

### 5.3 Python API

```python
# 简单预测（最快路径）
from yicenet import YiCeNetEngine
engine = YiCeNetEngine()
result = engine.predict("解释这段代码")          # ~7ms

# 带上下文的完整路径
prescription = engine.prescribe(
    task_text="重构 API 路由",
    session_id="session_abc123",
    turn_id=5,
)                                               # ~12ms

# 快速环境分析
analysis = engine.analyze("搜索文档")           # ~3ms
```

---

## 第六部分: 重构执行计划

### Phase 1: 基础提取（行为不变）

| 任务 | 影响 | 行数 |
|------|------|------|
| 创建 `types.py`，移动所有 dataclass + TypedDict | Prescription, TurnRecord, Sample, PredictionResult | +50 |
| 创建 `engine_provider.py`，提取 _get_engine() + registry 发现 | hermes_tool.py, mcp_server.py 改为调 EngineProvider.get_engine() | +40/-24 |
| `hermes_tool.py` 引进 EngineProvider | 减少 12 行重复 | -12 |
| `mcp_server.py` 引进 EngineProvider | 减少 12 行重复 | -12 |
| **测试**: 45 pytest 通过 | 无功能变化 | |

### Phase 2: 接口声明（不改实现）

| 任务 | 影响 | 行数 |
|------|------|------|
| `interfaces.py` 声明 6 ABC | 契约先行，不修改现有实现 | +80 |
| 现有类加 `implements` 标记 | tokenizer.py → ITokenizer, encoder.py → IEncoder, memory_bank.py → IMemoryBank | +6 |
| **测试**: 45 pytest 通过 | 运行时行为不变 | |

### Phase 3: 依赖注入（核心重构）

| 任务 | 影响 | 行数 |
|------|------|------|
| `YiCeNetEngine.__init__` 接受接口注入 | engine.py 不再自 new 组件 | -10/+15 |
| `model.py` 支持 encoder/router/value_net 注入 | 允许外部替换 | -8/+12 |
| `providers/__init__.py` ProviderRegistry | default + override 模式 | +30 |
| `EngineProvider._build_engine` 使用注册表 | 统一构造 | -5 |
| **测试**: 45 pytest 通过 | 依赖注入后功能等价 | |

### Phase 4: 职责分解

| 任务 | 影响 | 行数 |
|------|------|------|
| `predict()` 拆出 `prescribe()` + `analyze()` | 237 行 → 3 个不到 80 行的方法 | -60 |
| `config.py` 拆为三个 dataclass | ModelArchConfig + TrainingConfig + PathConfig | -20/+25 |
| 移除 `return_prescription` 标志位 | predict() 不再隐式做 prescription | -5 |
| **测试**: 45 pytest 通过 | 方法拆分不影响结果 | |

### Phase 5: 平台抽象

| 任务 | 影响 | 行数 |
|------|------|------|
| `install/base.py` PlatformInstaller ABC | +30 |
| `install/hermes.py` HermesInstaller | +60 |
| `install/claude.py` ClaudeCodeInstaller | +50 |
| `bootstrap.py` 简化为编排器 | ~920 行 → ~200 行 | -720 |
| **测试**: 功能回归 | bootstrap --auto --target hermes/claude-code | |

### 总计

| Phase | 净增行数 | 改动文件 | 测试状态 |
|-------|---------|---------|---------|
| 1 | -36 | 4 | 45 pass |
| 2 | +86 | 6 | 45 pass |
| 3 | +37 | 6 | 45 pass |
| 4 | -60 | 4 | 45 pass |
| 5 | -580 | 5 | 手动回归 |
| **合计** | **~-553** | **25** | **Phase 1-4 自动化** |

---

## 第七部分: 保持不动部分

以下模块审计评为 B 或以上，不改:

| 模块 | 评分 | 原因 |
|------|------|------|
| hexagram.py | A | 纯函数，零外部依赖，接口清晰 |
| display.py | B+ | 独立格式化层，更换格式不影响逻辑 |
| env_context.py | B | 16-dim 向量内聚，计算链清晰 |
| cross_attention.py | B | numpy 计算图，无副作用 |
| probes.py | B | GPU/CPU 自适应，内聚 |
| datasource/* | A | ABC 最佳实践，三种实现 |
| world_model.py | B | 双头架构内聚 |
| rl_train.py | B | 训练循环内聚 |
| metrics.py | B+ | SQLite 日志隔离 |

---

## 第八部分: 已关闭的架构问题

| 问题 | 当前状态 | 解决方案 |
|------|---------|---------|
| LOOM config 污染 YiCeNet | ✅ 已关闭 | loom 配置移至 ~/LOOM/loom.yaml |
| display mode 两段代码 | ✅ 已关闭 | format_prediction() 是唯一路径，LOOM 传参调用 |
| bus_stop 别名不一致 | ✅ 已关闭 | 统一 compact, 加 64 卦卦辞 |
| Tokenizer HF 联网阻塞 | ✅ 已关闭 | ~/.yicenet/tokenizer/ 本地缓存 |
| Installer 命名误导 | ✅ 已关闭 | install-hermes-hooks.sh |
| Hermes plugin 冷启动 | ✅ 已关闭 | 运行时配置注入 env vars |
| Checkpoint 堆积 | ✅ 已关闭 | registry 精简，CI 回退 minimal.pt |
| key_insight 格式 | ✅ 已关闭 | 輪次 → #N |

## 第九部分: 未解决问题

| 问题 | 优先级 | 说明 |
|------|--------|------|
| Claude Code MCP 冷启动 26s | P0 | 建议移除 MCP，改用 PreMessageSend hook |
| MemoryBank 全局单例 | P0 | Phase 3 解决（IMemoryBank 注入） |
| predict() 237 行 | P0 | Phase 4 解决 |
| bootstrap.py 920 行 | P0 | Phase 5 解决 |
| 无 TypedDict 契约 | P1 | Phase 1 解决 |
| hermes_tool/mcp 重复 _get_engine | P1 | Phase 1 解决 |
| YiCeNetConfig 三职责 | P1 | Phase 4 解决 |
| 零扩展模式 (仅 1 ABC) | P1 | Phase 2 解决 |
| 零依赖注入 | P2 | Phase 3 解决 |
| Windows exe 文件锁 | P3 | pip --no-compile 缺陷，不影响 .py |
| RuntimeContext 抽象层 | P3 | Hermes+MCP 共享，可延迟到 Phase 5 |

---

## 附录 A: 当前架构关键指标

```
源文件:       30 (src/yicenet/)
参数:         5,671,859 (~22MB FP32)
推理延迟:     7ms (CPU, warm)
冷启动:       26s (首次 predict, 含 tokenizer + checkpoint)
编码器:       TinyEncoder (4 层 Transformer, 256-dim)
路由:         Gumbel-Softmax (τ: 1.0→0.1)
候选:         8 (本卦/错卦/综卦/互卦/之卦×4)
值网络:       3 层 MLP (256→128→64→1)
动作:         50 行为原语
探针:         9 维 (h_norm, h_ent, logit_ent, 3×clan, q_gap, jump, conf)
环境向量:     16 维 (hour, dow, turn#, depth, stability, velocity, diversity, entropy, success, correction, completed, praised, attn_ent)
飞轮:         7-step pipeline, 6h 周期
数据中心:     HermesDataSource / ClaudeCodeDataSource / FlywheelBufferSource
世界模型:     Dual-head + Power-law forgetting (τ=30d/3d, α=1.5)
RL 微调:      v5, 64-dim projection, cosine reward
```

## 附录 B: 文件索引

```
src/yicenet/__init__.py           — 公共 API 导出 (15.6.0)
src/yicenet/yicenet_engine.py     — 推理引擎 (723 行, 待重构)
src/yicenet/model.py              — YiCeNet 模型 (300 行)
src/yicenet/config.py             — 配置 + 路径解析 (325 行)
src/yicenet/tokenizer.py          — Qwen BPE tokenizer (312 行)
src/yicenet/encoder.py            — TinyEncoder (120 行)
src/yicenet/decoder.py            — ActionDecoder (55 行)
src/yicenet/value_net.py          — 值网络 (35 行)
src/yicenet/hexagram.py           — 卦象推理 (80 行)
src/yicenet/probes.py             — 9 探针 (80 行)
src/yicenet/interfaces.py         — ProbeExtractor ABC (50 行)
src/yicenet/constants.py          — PrecomputedTables (150 行)
src/yicenet/memory_bank.py        — 会话记忆 (200 行)
src/yicenet/cross_attention.py    — 上下文处方 (120 行)
src/yicenet/env_context.py        — 16-dim 环境向量 (100 行)
src/yicenet/display.py            — 格式化层 (100 行)
src/yicenet/flywheel.py           — 飞轮训练 (250 行)
src/yicenet/world_model.py        — 双头世界模型 (120 行)
src/yicenet/rl_train.py           — RL v5 微调 (200 行)
src/yicenet/metrics.py            — SQLite 日志 (150 行)
src/yicenet/hermes_tool.py        — Hermes 工具 (80 行)
src/yicenet/mcp_server.py         — MCP 服务器 (254 行)
src/yicenet/bootstrap.py          — 安装引导 (920 行, 待重构)
src/yicenet/datasource/__init__.py — DataSource ABC (30 行)
src/yicenet/datasource/hermes.py  — Hermes 数据源 (100 行)
src/yicenet/datasource/claude_code.py — Claude 数据源 (100 行)
src/yicenet/datasource/buffer.py  — 缓冲区数据源 (50 行)
```

---

*本报告基于 YiCeNet v15.6.0 的 30 个源文件代码审计、5 个实际应用场"
