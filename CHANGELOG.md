# Changelog

All notable changes to YiCeNet (易策网络) will be documented in this file.

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
