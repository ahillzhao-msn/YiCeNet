# Changelog

All notable changes to YiCeNet (易策网络) will be documented in this file.

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
