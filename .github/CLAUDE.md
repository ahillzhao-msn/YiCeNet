# YiCeNet — Project Context for Claude Code

## Architecture

- **YiCeNetEngine**: load_model() picks version from ~/.yicenet/checkpoints/registry.json
- **WorldModelV3**: 36-dim input (9 probes + 27 context_vector) + 64 hex one-hot = 100 dim shared → 128 → HeadA(64) + HeadB(3)
- **ContextCollector**: 27-dim TurnSignal TypedDict in hook_engine/collector/types.py
- **MemoryBank**: FileBackend at ~/.yicenet/data/memory/{session_id}.jsonl
- **Flywheel buffer**: ~/.yicenet/data/flywheel_buffer.jsonl (4561 entries, June 8-22)
- **Checkpoints**: ~/.yicenet/checkpoints/ (registry.json + .pt files)

## Key Files

| File | Purpose |
|------|---------|
| src/yicenet/world_model.py | WorldModelV2 (old), WorldModelV3 (new) |
| scripts/rl_train.py | WM training + RL fine-tuning |
| src/yicenet/flywheel.py | Auto-training pipeline, called by cron |
| src/yicenet/hook_engine/collector/types.py | TurnSignal (27-dim) TypedDict |
| src/yicenet/yicenet_engine.py | YiCeNetEngine main class |
| tests/ | pytest, run with python -m pytest tests/ -x |
| DESIGN-phase3.md | Current phase design spec |

## Training Data

- Flywheel buffer: ~/.yicenet/data/flywheel_buffer.jsonl
- Fields: satisfaction, completed, corrected, praised, abandoned, token_cost, hexagram_evolution, timestamp
- No probe vectors in flywheel — compute from user_text using encoder
- No context_vectors in flywheel — construct from existing fields

## Naming Conventions

- All new code in src/yicenet/ (flat or subpackages)
- Use `from __future__ import annotations` for forward refs
- TypedDict for structured data, ABC for interfaces
- Snake_case for functions/vars, PascalCase for classes

## Workflow

1. Design doc in repo root DESIGN-phaseN.md
2. Implement in src/yicenet/
3. Test with python -m pytest tests/ -x --tb=short
4. Full suite: python -m pytest tests/
5. Build: python -m build --wheel
