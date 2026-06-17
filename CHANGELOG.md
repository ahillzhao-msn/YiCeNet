# Changelog

All notable changes to YiCeNet (易策网络) will be documented in this file.

## [16.0.0] — 2026-06-17

Major release: real feedback signal pipeline, platform-independent hook architecture,
MemoryBank persistence, and performance fixes.

### Added

- **`hook_engine` package** (`src/yicenet/hook_engine/`) — platform-independent
  layer that decouples feedback extraction and orchestration from platform specifics.
  - `PlatformAdapter` Protocol: `platform_id`, `process_model`, `session_id()`,
    `prompt()`, `assistant_response()`, `platform_signals()`
  - `FeedbackSignals` frozen dataclass: `continued/corrected/completed/praised/abandoned/satisfaction/token_cost`
  - `extract_feedback(next_prompt, last_turn)` — pure function, no I/O; infers
    all signals from the next turn's user text (1-turn delay design)
  - `signals_from_platform(raw)` — lifts platform-provided dict to `FeedbackSignals`
  - `build_trajectory(signals, last_turn, session_id, platform)` — assembles
    flywheel sample dict
  - `HookOrchestrator`: `before_prediction()` submits Turn N's real feedback at
    the start of Turn N+1; `on_turn_complete()` stores response metadata

- **`FileBackend`** (`src/yicenet/memory_bank.py`) — JSONL write-through
  persistence for `MemoryBank`, designed for cross-process state sharing:
  - WAL (Write-Ahead Log) `update()`: O(1) patch-line append instead of O(n)
    full file rewrite; `load()` merges patches in a single pass
  - `compact(session_id)`: folds WAL patches into base records (called from
    `cleanup_stale()` for live sessions)
  - `cleanup_stale(max_age_hours=48)`: deletes stale session files, compacts
    live ones; TTL configurable via `memory.session_ttl_hours` in config.yaml
  - `configure_memory_bank_for(adapter)`: idempotent singleton factory — attaches
    `FileBackend` for subprocess adapters, uses config flag for daemon adapters

- **`TurnRecord.metadata`** — flexible `dict` field populated incrementally
  across hooks: `response_snippet`, `response_char_count`, `platform_signals`

- **`ClaudeCodeAdapter`** (`process_model="subprocess"`) and **`HermesAdapter`**
  (`process_model="daemon"`) implementing `PlatformAdapter`

- **`HermesAdapter.platform_signals()`** — derives `corrected/praised/satisfaction`
  directly from `conversation_history` at `post_llm_call` time; no text guessing

- **`configure_memory_bank_for()`** call added to `_claude_runner.py` and
  `_hermes_stub.py` so `FileBackend` is active before any engine code runs

- **CJK pattern expansion** (`src/yicenet/external_metrics.py`):
  - All four pattern sets now carry both Traditional and Simplified Chinese variants
  - Removed `\b` anchors from CJK patterns (no ASCII word boundaries in Chinese)
  - Praise: 谢谢/感谢/太棒了/辛苦了 + English *well done / spot on / nailed it*
  - Correction: 不对啊/再来一次/不对不对 + English *hold on / try again / fix it*
  - Completion: 明白了/嗯嗯/没问题 + English *i see / makes sense / go ahead*
  - Abandon: 先这样吧/就这样 + English *enough / i'm done / that will do*
  - Coverage: 15/15 CJK test cases PASS (previously 0% Simplified Chinese hit rate)

### Changed

- **`flywheel.py`**: `submit_trajectory()` writes directly to
  `~/.yicenet/data/flywheel_buffer.jsonl` via cross-platform file lock
  (`msvcrt` on Windows, `fcntl` on POSIX); eliminates in-memory buffer loss
  on process exit. `flywheel_run()` calls `_rotate_buffer()` after training
  (keeps last 500 records).

- **`MemoryBank.store_turn()`** gains `timestamp` and `metadata` optional params.
  New methods: `update_turn_metadata()`, `get_last_turn()`

- **`ClaudeCodeAdapter.post_tool_use()`** is now a no-op; removed `PostToolUse`
  from `ClaudeCodeInstaller._patch_settings()` and live `~/.claude/settings.json`
  (feedback signals require the next user message; the subprocess was wasted)

- **`HermesAdapter.turn_id()`** falls back to `len(conversation_history) - 1`
  when `context["turn_id"]` is absent, preventing stale `turn_id=0` collisions

- **`hermes_hook.py:pre_llm_call()`** passes `session_id` and `turn_id` to
  `engine.predict()` so `TurnRecord.turn_id` reflects actual conversation position

### Fixed

- **World model reward collapse**: `post_tool_use()` previously hardcoded
  `continued=True, corrected=False` for every sample — 100% win_rate was
  meaningless. Real signals now flow from `before_prediction()` via 1-turn delay.

- **Token cost overflow**: old producers (`loom-hooks`, `yicenet-hooks`) emitted
  `avg_tc ≈ 2,300,000` (raw token count, not normalized cost). Fixed in
  `estimate_token_cost()` — now returns `[0, 1]` normalized float.

- **Simplified Chinese not matched**: `_PRAISE_PATTERNS` only had Traditional
  Chinese (`謝謝`); simplified `谢谢` never matched. Fixed by adding both scripts.

## [15.6.0] — 2026-06-16

### Added

- **Tokenizer 本地缓存** — `~/.yicenet/tokenizer/qwen2.5-0.5b/` 存储
  Qwen2.5-BPE tokenizer（5 文件，~11 MB）。`_get_qwen_tokenizer()` 从本地
  路径加载，`trust_remote_code=False`，永不联网。
  (`src/yicenet/tokenizer.py`)
- **`download_tokenizer()`** — 公开函数，通过 huggingface_hub 或 HTTPS
  回退下载 tokenizer 文件。
- **`tokenizer_available()`** — 检查本地是否已缓存。
- **Bootstrap Phase 4b** — `yicenet-bootstrap --auto` 自动下载 tokenizer。
  (`src/yicenet/bootstrap.py`)

### Changed

- **`install-yicenet-hooks.sh` → `install-hermes-hooks.sh`** — 脚本更名，
  反映其安装在 Hermes 而非 YiCeNet 的事实。工作流和发布资产同步更新。
  (`scripts/install/install-hermes-hooks.sh`,
  `.github/workflows/build-release.yml`)

## [15.5.5] — 2026-06-15

### Added

- **Display config** — `~/.yicenet/config.yaml` gains `display:` section with
  `hexagram_chain` (卦链开关) and `mode` (`bus_stop` | `detailed`).
  `get_display_config()` reads it; `format_prediction()` accepts `bus_stop` as
  alias for `compact`. (`src/yicenet/config.py`, `src/yicenet/display.py`)
- **`LOOM.yml display.hexagram_mode`** — Display mode config moved from
  YiCeNet to LOOM's own config where it belongs. (`~/LOOM/loom.yml`)

### Changed

- **Hermes plugin env setup** — Plugin now reads `~/.yicenet/config.yml`
  `runtime:` section at init time, sets `TRANSFORMERS_OFFLINE=1`,
  `HF_HUB_OFFLINE=1`, `TQDM_DISABLE=1` before engine loads. Buffer path
  uses `yicenet_data_dir()` instead of hardcoded `~/.hermes/data/yicenet/`.
  (`~/.hermes/plugins/yicenet-hooks/__init__.py`)

## [15.5.4] — 2026-06-15

### Fixed

- **cross_attention.py key_insight format** — Removed redundant `輪次` prefix
  from bus-stop header. Format changed from `焦點: 輪次 #N` to `焦點: #N`
  (`src/yicenet/cross_attention.py:223`)

### Changed

- **Production deployment default** — `YICENET_HOME` env var should now be
  **unset** for production use; engine auto-resolves to `~/.yicenet/` on
  non-editable installs. Source-tree development still uses `YICENET_HOME`.

### Cleanup

- **`~/.yicenet/data/` directory** — Removed misplaced files: `config.yaml`,
  `SOUL.md`, empty `checkpoints/`, `data/`, `logs/` subdirectories.
  `flywheel_buffer.jsonl`, `metrics.db`, `qwen_to_yicenet.json` retained at
  `~/.yicenet/data/`.
- **`checkpoints/` cleanup** — Deleted 3 stale 22MB checkpoints
  (`yicenet_v15.pt`, `yicenet_v14.pt`, `yicenet_rl_best.pt`). Promoted
  `yicenet_v18.pt` (avg_reward=0.9903, highest) to active. Simplified
  `registry.json` to active + fallback only (no history). Now tracked in
  git for CI release builds.

## [15.5.3] — 2026-06-15

### Changed

- **Bus-stop hexagram display** — LOOM header now uses compact `〈歷史摘要 |
  mode | 焦點: #N, 壓縮比 X%〉` format (公交站台风格).
- **Cross-attention key_insight cleanup** — Unified all three entropy branches
  (low/medium/high) to use 公交站台 key_insight format; removed entropy value
  exposure from user-facing header. Single-turn sessions show `焦點: #0, 壓縮比
  0%` instead of exposing internal entropy.
- **Claude Code MCP rework** — MCP tools restructured for consistent
  `predict/attend/feedback/switch` interface.
- **Runtime config** — `~/.yicenet/config.yaml` now supports `runtime:` section
  for `transformers_offline`, `hf_hub_offline`, `tqdm_disable`.

## [15.5.2] — 2026-06-15

### Changed

- Release workflow incorporates bootstrap download against GitHub Assets API.

## [15.5.1] — 2026-06-15

### Fixed

- `NameError: sp.run` in `bootstrap.py` — `_remove_crontab` used `sp.run()`
  without importing `sp`; replaced with `subprocess.run()`.

### Added

- Release workflow now bundles wheel + model checkpoints + install scripts into
  a single GitHub Release artifact.

## [15.5.0] — 2026-06-15

### Added

- **SOUL template system** — `SOUL-template.md` defines YiCeNet's identity
  (心·道/骨·法/皮·儒/用·⽤). `~/.yicenet/SOUL.md` created during bootstrap
  Phase 5. SOUL summary injectable into host system prompt.
- **Cross-platform flywheel scheduler** — Native OS scheduling via crontab
  (Linux/macOS) and Task Scheduler (Windows). No longer depends on Hermes cron.
- **`yicenet-uninstall` CLI** — Clean removal of Hermes plugin, Claude Code MCP,
  flywheel scheduler. `--clean-data` flag deletes `~/.yicenet/`.
- **`yicenet-bootstrap` Phase 5** — Creates `~/.yicenet/` with `config.yaml`,
  `SOUL.md`, and `data/`, `logs/`, `checkpoints/` directories.
- **Config template `soul:` section** — `DEFAULT_CONFIG_YAML` includes SOUL
  integration settings (`enabled`, `priority_weights`, `inject_targets`,
  `injection_level`).
- **Performance benchmark section in skill doc** — Standardised benchmark with
  warmup, parameter count, per-version expected CPU latency.

### Fixed

- **Install docs** — `INSTALL.md` updated with full install/uninstall lifecycle.

## [15.4.0] — 2026-06-15

### Fixed

- **Hermes plugin flywheel buffer path** — Changed `_YICENET_BUFFER` from
  hardcoded `~/YiCeNet/data/flywheel_buffer.jsonl` to `yicenet_data_dir() /
  "flywheel_buffer.jsonl"`, ensuring compatibility with wheel-based installs
  where no source tree exists. (`scripts/install/__init__.py`)
- **MCP server fallback checkpoint** — Replaced hardcoded `yicenet_v15.pt`
  with automatic discovery of the latest checkpoint from `checkpoints/`
  directory via `glob("yicenet_v*.pt")`. (`src/yicenet/mcp_server.py`)
- **Probe index in env_context** — `compute_env_confidence()` was reading
  `probe_list[7]` (jump_distance) as q_gap; corrected to `probe_list[6]`.
  (cherry-picked from Claud Code commit c173346)

### Changed

- **`mcp>=1.0` promoted from optional to core dependency** — `pip install
  yicenet` now automatically supports MCP server mode; no `[mcp]` extras
  needed. (`pyproject.toml`)
- **`pyproject.toml` authors** — Updated from placeholder `"YiCeNet
  Contributors"` to `"ahillzhao-msn" <105655625+ahillzhao-msn@...>`.
- **Model label** — `count_parameters()` display label corrected from
  `"Env Projector (7→256)"` to `"Env Projector (16→256)"` to match actual
  dimensionality. (`src/yicenet/model.py:499`)

### Added

- **Test suite for datasource adapters and env_context** — 18 new tests
  covering `build_env_vec()`, `compute_env_confidence()`,
  `FlywheelBufferSource`, `HermesDataSource`, and `Sample` dataclass.
  (`tests/test_datasource.py`)

## [15.3.2] — 2026-06-15

### Added

- **env_vec slot dropout** — `env_context.py` now provides
  `env_vec_dropout(env_vec, p, training)` which independently masks each of
  the 16 env slots with probability p during training, so the
  EnvironmentProjector learns to handle any subset of provided signals. At
  inference time p=0 preserves full context. Default dropout rate: 0.3.
- **`train.py` integration** — `_state_to_env()`, `_batch_env_from_features()`
  helpers; pretrain Phase 3 and `rl_train_stage()` both pass env_vec with
  dropout to the model; `--env_dropout` CLI flag (default 0.3).

## [15.3.1] — 2026-06-15

### Added

- **16-dim env vector** — `ENV_DIM` expanded from 7 to 16 with four logical
  groups: time phase (sin/cos hour + day_of_week), session depth (turn,
  memory_bank_depth, hexagram_stability, last_hexagram_id), hexagram chain
  dynamics (velocity, clan_diversity, hexagram_entropy, tool_bin), quality
  signals (correction/completed/praised rates), and attention_entropy.
- **Auto-computed chain signals** — `yicenet_engine.py` now has
  `_compute_chain_signals()` that reads hexagram history from MemoryBank to
  auto-compute memory_bank_depth, hexagram_stability, hexagram_velocity,
  clan_diversity, and hexagram_entropy when session_id is provided.
- **MemoryBank query** — `get_hexagram_history(session_id)` added for
  accessing per-session hexagram trajectories.
- **Caller override** — Caller-supplied environment keys always override
  auto-computed values for the same slot.

## [15.3.0] — 2026-06-15

### Added

- **EnvironmentProjector** — `nn.Linear(16, 256)` with zero-initialised
  weights added to YiCeNet model. Adds 4,096 parameters (total now
  5,643,123). Zero-init ensures full backward compatibility with existing
  v15.x checkpoints via `strict=False`. `encode_context()` and `forward()`
  accept optional `env_vec` parameter.
- **env_context module** — Platform-agnostic 7-dim structural signal vector
  (hour_of_day, session_turn, last_hexagram_id, correction_rate, etc.).
  `build_env_vec()` and `compute_env_confidence()` provide signal
  construction and routing confidence estimation.
- **MCP server env support** — `yicenet_attend` and `yicenet_predict` tools
  accept `environment: dict = None` parameter; predict returns
  `env_confidence`, `context_status`, `context_hint`.
- **DataSource abstraction** — `datasource/` module with abstract
  `DataSource` base class and `Sample` dataclass. Three implementations:
  `HermesDataSource` (Hermes state.db), `ClaudeCodeDataSource`
  (`~/.claude/projects/**/*.jsonl`), `FlywheelBufferSource`
  (`~/.yicenet/data/flywheel_buffer.jsonl`).
- **Multi-source flywheel** — `flywheel.py` replaced Hermes-only
  `scan_new_messages()` with platform-adaptive `scan_all_sources()` that
  calls all available DataSources. Legacy `scan_new_messages()` kept as
  deprecated wrapper. `STATE_FILE` moved from `~/.hermes/data/` to
  `yicenet_home() / "state.json"`.
- **Bootstrap with --target** — `bootstrap.py` now accepts
  `--target hermes|claude-code|auto` for platform-adaptive installation.
- **Claude Code integration** — MCP server (`src/yicenet/mcp_server.py`),
  installation script (`scripts/install/install-claudecode-hooks.sh`),
  `yicenet-serve` entry point in `pyproject.toml`.
- **Three-tier path resolution** — `yicenet_home()` now
  checks `$YICENET_HOME` > editable install source tree > `~/.yicenet/`
  (default for wheel installs). Install mode detected via
  `importlib.metadata`.
- **User config overlay** — `~/.yicenet/config.yaml` overrides
  runtime-tunable params (flywheel schedule, inference temperature, eval API)
  via `get_config()` function. Env vars take highest priority.
- **ARCHITECTURE.md section 9** — Documents env awareness design and the
  two-vector architecture (9-dim probe vs 7/16-dim env).

## [15.2.0] — 2026-06-11

### Changed

- Build system migrated from poetry to hatchling. Version bump only — no
  feature changes in this release.
- GitHub Actions release workflow added for automated wheel builds.

## [15.0.1] — 2026-05-30

### Fixed

- **WorldModelV2 head_external Sigmoid** — Added `nn.Sigmoid()` to `head_external`
  to constrain `pred_ext` to `[0, 1]` range. Previously raw linear output could
  produce arbitrary-magnitude values, causing `loss_B` to explode to ~5e11 while
  `loss_A` remained at ~0.0029. Sigmoid has no trainable parameters so old
  checkpoints load without any key mismatch. (#0291df2)

- **Flywheel evaluations table** — Added `CREATE TABLE IF NOT EXISTS` for
  `evaluations`, `hexagram_usage`, and `trajectories` tables in
  `_record_evaluation()` to prevent `no such table: evaluations` crash on first
  flywheel evaluation run. (#0291df2)

### Changed

- `src/yicenet/__init__.py`: `__version__` bumped from `15.0.0` → `15.0.1` to
  match `pyproject.toml`
- `ARCHITECTURE.md`: Version header updated from `15.0.0` → `15.0.1`

## [15.0.0] — 2026-05-29

### Added

- Initial project scaffolding and poetry build system
- `WorldModelV2` with dual-head architecture and power-law forgetting
- Flywheel continuous training loop with SQLite state tracking
- External Producer API for flywheel data injection
- yicenet-hooks plugin for Hermes post-hook reward signal
- Full semantic versioning and changelog
