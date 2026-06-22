# Context Collector — Architecture Design

> 卦师察言观色。这里的"色"不在卦辞里，在求卦人的两轮对话之间。
>
> 版本: 1.0 · 状态: Draft · 作者: Hermes Architect (review by CC Implementer)

---

## 1. Architecture Overview

```
Three platform modes, one interface, one data endpoint.

                    ┌─────────────────────┐
                    │  ContextCollector   │ ← ABC (interface)
                    │  (interface.py)     │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
    ┌─────────────────┐ ┌──────────────┐ ┌──────────────┐
    │ DaemonCollector │ │ SubprocColl. │ │ ExplicitColl.│
    │ (Hermes daemon) │ │ (CC subproc) │ │  (MCP args)  │
    └────────┬────────┘ └──────┬───────┘ └──────┬───────┘
             │                 │                 │
             └─────────────────┼─────────────────┘
                               │
                               ▼
                    TurnRecord.metadata
                    ["context_vector"] = ℝ²⁷
                               │
                               ▼
                    MemoryBank → flywheel → WM training
```

### 1.1 Why Three Implementations

| Mode | Process Model | Signal Richness | Sharing Mechanism |
|------|--------------|-----------------|-------------------|
| **Daemon** | Single long-lived process | Full 27-dim | In-memory accumulator |
| **Subprocess** | Each hook = new process | ~5-dim (expandable) | File-based side-channel |
| **Explicit** | Stateless tool calls | Variable (per call) | All signals as args |

The same `ContextCollector` ABC, same `build_vector()` output format, different backends
for how signals accumulate across hook invocations.

### 1.2 Three-Layer Separation

```
┌─────────────────────────────────────────────────────┐
│ Layer 3: Adapter (HermesAdapter / ClaudeCodeAdapter) │ ← owns lifecycle
├─────────────────────────────────────────────────────┤
│ Layer 2: Collector (Daemon / Subprocess / Explicit)  │ ← accumulates signals
├─────────────────────────────────────────────────────┤
│ Layer 1: SignalVector (TypedDict, 27 fields)         │ ← output contract
└─────────────────────────────────────────────────────┘
```

---

## 2. Interface Layer — ContextCollector ABC

### 2.1 Sniffing Interface

```python
# yicenet/hook_engine/collector/interface.py

from abc import ABC, abstractmethod
from typing import Optional

class ContextCollector(ABC):
    """每轮环境信号的聚合器接口。
    
    Three implementations match three platform process models.
    """

    @abstractmethod
    def sniff_user(self, text: str, is_first_turn: bool = False) -> None:
        """pre_llm_call: user message text + session freshness."""
        ...

    @abstractmethod
    def sniff_api(self, prompt_tokens: int, completion_tokens: int,
                  duration_ms: float = 0.0) -> None:
        """post_api_request: real token counts + API latency."""
        ...

    @abstractmethod
    def sniff_tool(self, name: str, exit_code: int,
                   duration_ms: float, result_size_bytes: int = 0) -> None:
        """post_tool_call: each tool execution (may be called 0..N times)."""
        ...

    @abstractmethod
    def sniff_response(self, text: str) -> None:
        """post_llm_call: assistant response text."""
        ...

    @abstractmethod
    def sniff_hexagram(self, q_max: float, q_gap: float,
                       entropy: float) -> None:
        """After YiCeNet prediction: hexagram Q-value distribution."""
        ...

    @abstractmethod
    def sniff_timing(self, user_interval_sec: Optional[float] = None,
                     prev_metadata: Optional[dict] = None) -> None:
        """Next-turn pre_llm_call: inter-turn interval + prior turn state."""
        ...

    @abstractmethod
    def build_vector(self, prev_metadata: Optional[dict] = None) -> dict:
        """Assemble ℝ²⁷ normalized signal vector for this turn.
        
        Returns a dict with 'tok_' prefix keys, values in [0,1] or [-1,1].
        """
        ...
```

### 2.2 SignalVector TypedDict

```python
# yicenet/hook_engine/collector/types.py

from typing import TypedDict

class SignalVector(TypedDict):
    """A normalized ℝ²⁷ vector encoding one turn's environment signals."""

    # ── User input (2) ──
    tok_user_input_len: float       # [0, 1]  字符数/512
    tok_is_first_turn: float        # {0, 1}

    # ── API consumption (3) ──
    tok_prompt_tokens: float        # [0, 1]  prompt_tokens/4096
    tok_completion_tokens: float    # [0, 1]  completion_tokens/4096
    tok_api_duration: float         # [0, 1]  duration_ms/10000

    # ── Tool execution (6) ──
    tok_tool_count: float           # [0, 1]  N/10
    tok_tool_success_rate: float    # [0, 1]
    tok_tool_retry_count: float     # [0, 1]  N/5
    tok_tool_duration: float        # [0, 1]  total_ms/30000
    tok_tool_output_size: float     # [0, 1]  total_bytes/1M
    tok_tool_diversity: float       # [0, 1]  unique_tools/8

    # ── Response properties (3) ──
    tok_response_len: float         # [0, 1]  chars/4000
    tok_has_code: float             # {0, 1}
    tok_code_block_count: float     # [0, 1]  N/10

    # ── Hexagram (3) ──
    tok_hex_conf: float             # [0, 1]  max Q-value
    tok_hex_q_gap: float            # [0, 1]  Q₁ - Q₂
    tok_hex_entropy: float          # [0, 1]  entropy/4.0

    # ── Cross-turn timing (3) ──
    tok_user_speed: float           # [0, 1]  interval_sec/60
    tok_user_speed_ratio: float     # [0, 2]  current/prev_mean
    tok_mood_trend: float           # [-1, 1] satisfaction_delta

    # ── Prior turn state (3) ──
    tok_drift_trend: float          # [-1, 1] hex_q_delta
    tok_is_prev_correction: float   # {0, 1}
    tok_is_prev_praise: float       # {0, 1}

    # ── Current-turn feedback (4) ──
    tok_user_satisfaction: float    # [-1, 1]  composite signal
    tok_is_correction: float        # {0, 1}
    tok_is_praise: float            # {0, 1}
    tok_is_abandon: float           # {0, 1}
```

### 2.3 Old `external_metrics.py` Relationship

The existing `external_metrics.py` regex-based signal derivation is **not removed**.
It becomes a sub-component of the satisfaction computation:

```
ContextCollector._compute_satisfaction()
    ├── signal-based score (tool success, code density, user engagement)  ← primary
    └── external_metrics pattern matching                                 ← fallback/reference
```

The old `compute_satisfaction()` and `extract_external_vector()` remain for
backward compatibility but are no longer the sole signal source for WM training.

---

## 3. Platform Implementations

### 3.1 DaemonContextCollector

#### When Used
- **Hermes** daemon process (the `_adapter` singleton in `hermes_hook.py`)
- **Claude Code daemon mode** (hook_server IPC)

#### How It Works

All sniff methods write to in-memory instance variables. `build_vector()` computes
cross-turn derivations from `prev_metadata`. The instance lives as long as the
adapter's turn lifecycle.

```
pre_llm_call → new_turn() → DaemonCollector created
  post_api_request → .sniff_api()
  post_tool_call(+) → .sniff_tool()  (called per tool)
  post_llm_call → .sniff_response()
                → .build_vector(prev_metadata) → ℝ²⁷ → MemoryBank metadata
```

#### Implementation

```python
class DaemonContextCollector(ContextCollector):
    """In-process accumulator. Full 27-dim vector.
    
    All signal collection happens within a single Python process.
    The collector is created at turn start and garbage-collected at turn end.
    """

    def __init__(self):
        self._user_text = ""
        self._is_first_turn = False
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._api_duration_ms = 0.0
        self._tools: list[dict] = []
        self._response_text = ""
        self._response_has_code = False
        self._code_block_count = 0
        self._hexagram_q_max = 0.0
        self._hexagram_q_gap = 0.0
        self._hexagram_entropy = 0.0
        self._turn_timestamp = 0.0
        self._user_interval_sec: Optional[float] = None
        self._prev_metadata: Optional[dict] = None

    def sniff_user(self, text, is_first_turn=False):
        self._user_text = text
        self._is_first_turn = is_first_turn
        self._turn_timestamp = self._turn_timestamp or time.time()

    def sniff_api(self, prompt_tokens, completion_tokens, duration_ms=0.0):
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._api_duration_ms = duration_ms

    def sniff_tool(self, name, exit_code, duration_ms, result_size_bytes=0):
        self._tools.append({
            "name": name,
            "exit_code": exit_code,
            "duration_ms": duration_ms,
            "result_size": result_size_bytes,
        })

    def sniff_response(self, text):
        self._response_text = text
        self._response_has_code = "```" in text or "~~~" in text
        self._code_block_count = (text.count("```") // 2) or (text.count("~~~") // 2)

    def sniff_hexagram(self, q_max, q_gap, entropy):
        self._hexagram_q_max = q_max
        self._hexagram_q_gap = q_gap
        self._hexagram_entropy = entropy

    def sniff_timing(self, user_interval_sec=None, prev_metadata=None):
        self._user_interval_sec = user_interval_sec
        self._prev_metadata = prev_metadata

    def build_vector(self, prev_metadata=None) -> SignalVector:
        # (full implementation as already coded in context_collector.py)
        ...
```

### 3.2 SubprocessContextCollector

#### When Used
- **Claude Code subprocess mode** (default: each hook = new process)
- Any platform where hook lifecycle is split across process boundaries

#### The Side-Channel Problem

```
pre_message_send ──▶ process A ──▶ sniff_user() ──▶ writes to file ──▶ exits
                                                       │
post_tool_use ────▶ process B ──▶ reads file ──▶ sniff_tool() ──▶ writes back ──▶ exits
                                                       │
stop ─────────────▶ process C ──▶ reads file ──▶ sniff_response() ──▶ build_vector() ──▶ exits
```

Each CC hook is a separate subprocess that creates a fresh adapter. The
ContextCollector must persist across these invocations via a **file-based side-channel**.

#### File Format

```
~/.yicenet/data/cc-ctx/{session_id}.{turn_id}.jsonl

Each line = one signal event:  {"event": "user|api|tool|response|hexagram|timing", ...}
Line 1: user event
Line 2-N: tool/api events
Last line: timing/response
```

#### Implementation

```python
class SubprocessContextCollector(ContextCollector):
    """File-based accumulator for cross-process hook invocations.
    
    Each hook subprocess:
      init_session() → loads existing file for (session, turn)
      sniff_*()     → appends event as JSON line
      build_vector()→ reads all events, assembles vector, optionally cleans up
    """

    _BASE = Path.home() / ".yicenet" / "data" / "cc-ctx"

    def __init__(self, session_id: str, turn_id: int):
        self._session_id = session_id
        self._turn_id = turn_id
        self._events: list[dict] = []  # in-memory buffer for current process
        self._loaded = False

    def sniff_user(self, text, is_first_turn=False):
        self._append_event({"type": "user", "text": text, "is_first": is_first_turn})

    def sniff_tool(self, name, exit_code, duration_ms, result_size_bytes=0):
        self._append_event({"type": "tool", "name": name, "exit_code": exit_code,
                            "duration_ms": duration_ms, "result_size": result_size_bytes})

    def sniff_api(self, prompt_tokens, completion_tokens, duration_ms=0.0):
        self._append_event({"type": "api", "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens, "duration_ms": duration_ms})

    def sniff_response(self, text):
        self._append_event({"type": "response", "text": text})

    def sniff_hexagram(self, q_max, q_gap, entropy):
        self._append_event({"type": "hexagram", "q_max": q_max,
                            "q_gap": q_gap, "entropy": entropy})

    def sniff_timing(self, user_interval_sec=None, prev_metadata=None):
        self._append_event({"type": "timing", "interval_sec": user_interval_sec})

    def _append_event(self, event: dict):
        self._events.append(event)
        self._flush()

    def _flush(self):
        """Append new events to file (atomic append, no locking needed for JSONL)."""
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            for e in self._events:
                f.write(json.dumps(e) + "\n")
        self._events = []

    def _load(self):
        """Load all events from file (called lazily on build_vector)."""
        if self._loaded:
            return
        self._loaded = True
        path = self._path()
        if path.exists():
            with open(path) as f:
                for line in f:
                    self._events.append(json.loads(line))

    def build_vector(self, prev_metadata=None) -> SignalVector:
        self._load()
        # Replay events into a DaemonContextCollector for computation logic reuse
        inner = DaemonContextCollector()
        for e in self._events:
            t = e.get("type")
            if t == "user":   inner.sniff_user(e["text"], e.get("is_first", False))
            elif t == "api":  inner.sniff_api(e["prompt_tokens"], e["completion_tokens"])
            elif t == "tool": inner.sniff_tool(e["name"], e["exit_code"],
                                               e["duration_ms"], e.get("result_size", 0))
            elif t == "response": inner.sniff_response(e["text"])
            elif t == "hexagram": inner.sniff_hexagram(e["q_max"], e["q_gap"], e["entropy"])
            elif t == "timing":   inner.sniff_timing(e.get("interval_sec"))
        inner.sniff_timing(prev_metadata=prev_metadata)
        return inner.build_vector(prev_metadata=prev_metadata)

    def _path(self) -> Path:
        return self._BASE / f"{self._session_id}.{self._turn_id}.jsonl"

    def cleanup(self):
        """Called after build_vector to remove the temp file."""
        try:
            self._path().unlink()
        except OSError:
            pass
```

#### Key Design Decision: Reuse Daemon's Computation

`SubprocessContextCollector.build_vector()` replays file events into a
`DaemonContextCollector` instance. The signal computation logic (normalization
formulas, cross-turn derivations) lives in ONE place — DaemonContextCollector.
Subprocess collector is just a file persistence layer.

### 3.3 ExplicitContextCollector

#### When Used
- **MCP server** (tools called with explicit signal arguments)
- **Unit tests** (construct test vectors directly)

#### Implementation

```python
class ExplicitContextCollector(ContextCollector):
    """All signals passed via constructor or explicit methods.
    
    No accumulation mode — every signal must be provided explicitly.
    Used by MCP tools and test harnesses.
    """

    def __init__(self, signals: Optional[dict] = None):
        self._data = signals or {}

    def _stub(self, *args, **kwargs):
        pass  # no-op: data is pre-populated or not used

    sniff_user = sniff_api = sniff_tool = sniff_response = _stub
    sniff_hexagram = sniff_timing = _stub

    def build_vector(self, prev_metadata=None) -> SignalVector:
        return SignalVector(
            tok_user_input_len=self._data.get("tok_user_input_len", 0.0),
            tok_is_first_turn=self._data.get("tok_is_first_turn", 0.0),
            # ... all 27 fields from self._data, defaulting to 0.0
        )
```

---

## 4. Integration Points

### 4.1 HooksAdapter (Base Class)

```python
# yicenet/tools/hooks_adapter.py (modified)

class HooksAdapter(ABC):
    
    # New method: factory for platform-appropriate collector  
    def create_collector(self, payload: dict) -> ContextCollector:
        """Override in subclasses to return platform-specific collector.
        
        Default: DaemonContextCollector (works for Hermes, daemon-mode CC).
        """
        from yicenet.hook_engine.collector.daemon import DaemonContextCollector
        return DaemonContextCollector()

    def new_turn(self, payload: dict) -> None:
        self._ctx = self.create_collector(payload)
        self._ctx.sniff_user(
            self.prompt(payload),
            is_first_turn=(self.turn_id(payload) == 0),
        )

    @property
    def ctx(self) -> Optional[ContextCollector]:
        return getattr(self, "_ctx", None)
```

### 4.2 HermesAdapter

```python
# yicenet/tools/hermes_hook.py (modified)

class HermesAdapter(HooksAdapter):
    # Inherits DaemonContextCollector from HooksAdapter default.
    # No override needed — daemon process model fits Hermes perfectly.
    
    # Hook wiring (already done):
    #   pre_llm_call → predict_for_turn_payload → new_turn → sniff_user
    #   post_api_request → ctx.sniff_api
    #   post_tool_call → ctx.sniff_tool
    #   post_llm_call → orchestrator.on_turn_complete → sniff_response → build_vector
```

### 4.3 ClaudeCodeAdapter

```python
# yicenet/tools/claude_hook.py (modified)

class ClaudeCodeAdapter(HooksAdapter):
    
    def create_collector(self, payload: dict) -> ContextCollector:
        if self.process_model == "daemon":
            return DaemonContextCollector()
        else:
            # subprocess mode: file-based persistence across hook invocations
            return SubprocessContextCollector(
                session_id=self.session_id(payload),
                turn_id=self.turn_id(payload),
            )
```

**CC hook wiring changes:**
- `pre_message_send` → `new_turn()` → sniff_user → sniff_hexagram → **flush to file**
- `post_tool_use` (new hook) → **read from file** → sniff_tool → **flush to file**
- `stop` → read from file → sniff_response → build_vector → MemoryBank → **cleanup file**

### 4.4 MCPAdapter

```python
# yicenet/tools/mcp_adapter.py (modified)

class MCPAdapter:
    
    def predict_with_signals(self, payload: dict) -> dict:
        """New combined endpoint: predict + accept explicit signals."""
        signals = payload.get("signals", {})
        collector = ExplicitContextCollector(signals)
        result = self._do_predict(payload)
        # Merge hexagram info into collector
        collector.sniff_hexagram(
            result.get("env_confidence", 0.0), 0.0, 0.0
        )
        return {**result, "context_vector": collector.build_vector()}
```

### 4.5 Hermes Plugin (`_hermes_stub.py`)

```python
# yicenet/tools/_hermes_stub.py (modified — minimal changes)

# The stub forwards to hermes_hook functions.
# HermesPlugin.__init__.py does NOT need modification for ContextCollector
# because hermes_hook.post_tool_call() and post_api_request() already
# feed into _adapter.ctx internally.

# Add: post_api_request forwarder (already done in previous iteration)
def post_api_request(context):
    from yicenet.tools.hermes_hook import post_api_request as _fn
    return _fn(context)
```

### 4.6 MemoryBank Integration

```python
# yicenet/hook_engine/orchestrator.py (already modified)

class HookOrchestrator:
    
    def on_turn_complete(self, payload: dict) -> None:
        session_id = self._adapter.session_id(payload)
        bank = self._bank()
        bank.init_session(session_id)
        
        last = bank.get_last_turn(session_id)
        if last is None:
            return
        
        response = self._adapter.assistant_response(payload)
        metadata: dict = {
            "response_snippet": response[:300],
            "response_char_count": len(response),
        }
        
        # Context vector from collector
        ctx = getattr(self._adapter, "ctx", None)
        if ctx is not None:
            ctx.sniff_response(response)
            # Load prev_metadata from bank for cross-turn derivations
            prev_turn = bank.get_turn(session_id, last.turn_id - 1) if last.turn_id > 0 else None
            prev_meta = (prev_turn.metadata or {}).get("context_vector") if prev_turn else None
            metadata["context_vector"] = ctx.build_vector(prev_metadata=prev_meta)
            # Clean up subprocess temp file if applicable
            if hasattr(ctx, "cleanup"):
                ctx.cleanup()
        
        raw_signals = self._adapter.platform_signals(payload)
        if raw_signals:
            metadata["platform_signals"] = raw_signals
        
        bank.update_turn_metadata(session_id, last.turn_id, metadata)
        
        if self._adapter.process_model == "subprocess":
            bank.flush_session(session_id)
```

---

## 5. File Structure

```
src/yicenet/
├── hook_engine/
│   ├── __init__.py
│   ├── orchestrator.py           # (modified — already done)
│   ├── context_collector.py      # ← REMOVED: replaced by collector/ package
│   ├── collector/                # ← NEW package
│   │   ├── __init__.py
│   │   ├── interface.py           # ContextCollector ABC
│   │   ├── types.py               # SignalVector TypedDict
│   │   ├── daemon.py              # DaemonContextCollector
│   │   └── subprocess.py          # SubprocessContextCollector
│   │   └── explicit.py            # ExplicitContextCollector
│   └── extractor.py              # (unchanged — pattern matching helpers)
│
├── tools/
│   ├── hooks_adapter.py          # (modified — add create_collector)
│   ├── hermes_hook.py            # (modified — already done)
│   ├── claude_hook.py            # (modified — add create_collector override)
│   ├── mcp_adapter.py            # (modified — add predict_with_signals)
│   └── _hermes_stub.py           # (modified — add post_api_request fwd)
│
tests/
├── test_hooks_adapter.py         # (modified — add collector tests)
├── test_collector/
│   ├── test_interface.py          # ABC contract tests
│   ├── test_daemon.py             # DaemonContextCollector unit tests
│   ├── test_subprocess.py         # SubprocessContextCollector unit + integration
│   └── test_explicit.py           # ExplicitContextCollector
```

---

## 6. Migration Path

### Phase 1 — Package + Interface (CC's first task)
1. Create `collector/` package with `interface.py` and `types.py`
2. Move `context_collector.py` into `daemon.py` as `DaemonContextCollector`
3. Create `SubprocessContextCollector` and `ExplicitContextCollector` stubs
4. Add `create_collector()` to `HooksAdapter`
5. Override in `ClaudeCodeAdapter` (subprocess → SubprocessCollector)
6. Update imports in `hermes_hook.py`, `orchestrator.py`

### Phase 2 — CC Hook Integration
7. Wire SubprocessCollector into CC `post_tool_use` hook script
8. Test: file-based side-channel survives process boundaries
9. Register new CC hook `post_tool_use` in installer

### Phase 3 — MCP Integration
10. Add `predict_with_signals()` to `MCPAdapter`
11. Update MCP tool schemas to accept `signals` parameter

### Phase 4 — WM Training Update
12. Modify `flywheel.py` / `datasource` to read `context_vector` from TurnRecord metadata
13. Update World Model input dimension from 3 to 27
14. Retrain WM with hold-out validation

### Phase 5 — Cleanup
15. Deprecate old `external_metrics.extract_external_vector()`
16. Remove `external_metrics.py` or reduce to utility functions
17. Remove old `scripts/install/__init__.py` plugin hooks (consolidated into `_hermes_stub.py`)

---

## 7. Testing Strategy

| Test | Scope | Method |
|------|-------|--------|
| ABC contract | Interface enforcement | Try to instantiate ABC → TypeError |
| Daemon full 27-dim | All sniff methods + build_vector | Unit: feed mock data, verify each `tok_` field |
| Daemon cross-turn | prev_metadata | Feed prev vector, verify mood_trend, drift_trend |
| Subprocess file I/O | JSONL read/write | Write events as separate process emulations, then build_vector |
| Subprocess cleanup | Cleanup after build | Verify .jsonl deleted |
| Explicit pre-built | Direct construction | Pass dict, verify output equals input |
| HooksAdapter integration | new_turn → ctx | Verify ctx created and sniff_user called |
| Orchestrator integration | on_turn_complete | Verify context_vector in metadata |
| Performance | 1000 turns | Verify <1ms per build_vector |

---

## 8. Key Design Decisions

1. **ABC not Protocol.** ContextCollector defines behavior, not just shape. ABC
   + abstractmethod enforces all sniff methods at instantiation time.

2. **Daemon as computation kernel.** `SubprocessContextCollector.build_vector()`
   creates a temp `DaemonContextCollector` to run the same normalization formulas.
   One truth, zero duplication.

3. **File-based, not DB-based, for subprocess side-channel.** JSONL is atomic at
   the line level, needs no locking for append-only writes, and is trivially
   debuggable (`cat file.jsonl`).

4. **No new Hermes plugin scripts.** `_hermes_stub.py` already wraps
   `hermes_hook.py`. All changes are in `hermes_hook.py` — the plugin only
   needs to re-deploy the stub (which it already does).

5. **External metrics not removed.** The regex-based patterns in
   `external_metrics.py` become a secondary signal in the satisfaction
   computation, not the primary one.

6. **SignalVector is a TypedDict, not a dataclass.** TypedDict documents the
   contract without enforcing runtime overhead. Runtime validation is optional
   (pydantic can be added later if needed).

---

## 9. Out of Scope (for this design)

| Item | Reason |
|------|--------|
| RL training loop changes | WM training is a separate concern (Phase 4) |
| Old Hermes plugin `scripts/install/__init__.py` | Two separate plugin implementations; consolidate in Phase 5 |
| CC `post_tool_use` hook `__init__` | CC hook registration is installer code, not core architecture |
| Frontend/UI for vector visualization | User mentioned CC is strong at UI — future enhancement |
