"""
YiCeNet Online Flywheel — two-speed learning (DESIGN-phase4.md).

Pipeline:
  1. Collect new samples from all registered DataSources (platform-agnostic)
     and append them to <home>/data/flywheel_buffer.jsonl
  2. Build the training set: buffer + archive, trajectories with the §3.1
     decision fields where present (legacy rows: question + 本卦 only)
  3. Train World Model V4 (察言观色) on the older samples
  4. Evaluate it on the newest samples against the base rate and the active WM;
     promote into registry.json["world_model"] only on a held-out win
  5. Slow 卦理 channel: distil a candidate prior from a validated WM; gates,
     K-cycle stability, shadow period, then promotion (prior_update.py)

The prior is no longer RL-trained every run.

DataSources registered by default_sources():
  HermesDataSource      — Hermes state.db (only when available)
  ClaudeCodeDataSource  — ~/.claude/projects/**/*.jsonl (only when available)
  FlywheelBufferSource  — <home>/data/flywheel_buffer.jsonl (always)
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

# ── Paths ──
from yicenet.config import get_learning_config, yicenet_home, yicenet_data_dir, yicenet_checkpoint_dir

YICENET_ROOT = yicenet_home()
CHECKPOINT_DIR = yicenet_checkpoint_dir()
REGISTRY_PATH = CHECKPOINT_DIR / "registry.json"
STATE_FILE = yicenet_home() / "state.json"

def _append_locked(path: Path, line: str) -> None:
    """Append one JSONL line with a cross-platform exclusive lock.

    Uses fcntl.flock on POSIX and msvcrt.locking on Windows so that
    concurrent hook processes (Claude Code PostToolUse, Hermes pre_llm_call)
    cannot interleave partial writes into the shared buffer file.
    """
    import sys
    with open(path, "a", encoding="utf-8", buffering=1) as f:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            try:
                f.write(line)
                f.flush()
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


def _rotate_buffer(buffer_path: Path) -> None:
    """Archive the current buffer and start fresh after a successful training run.

    Keeps the last KEEP_RECENT records in the live file so the next training
    run always has some warm-start data. Older records go to a dated archive.

    Design note: training functions intentionally use ALL buffer data with
    power-law decay weighting — rotation is purely a file-size management
    concern, not a "forget trained data" operation.
    """
    KEEP_RECENT = 500
    if not buffer_path.exists():
        return
    lines = buffer_path.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(lines) <= KEEP_RECENT:
        return  # small enough, nothing to do

    archive_dir = buffer_path.parent / "archive"
    archive_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"flywheel_buffer_{stamp}.jsonl"

    # Write old records to archive
    archive_path.write_text("".join(lines[:-KEEP_RECENT]), encoding="utf-8")
    # Rewrite live file with only the recent tail
    buffer_path.write_text("".join(lines[-KEEP_RECENT:]), encoding="utf-8")
    print(f"    Buffer rotated: {len(lines) - KEEP_RECENT} records → {archive_path.name}, {KEEP_RECENT} kept")


# Text that is not the customer's question: session-id placeholders from older
# producers, and messages the platform injects into the user turn.
#   "[claude-code] a51ffae7c48e", "[loom-hooks] sid=20260811_135" — but not "[WIP] 继续".
_PLACEHOLDER = re.compile(r"^\[[a-z][a-z0-9_-]*\] (?=[\w:=.-]*\d)[\w:=.-]+$", re.ASCII)
# Whole-message notices: nothing of the customer's in them.
_NOTICE_PREFIXES = (
    "[IMPORTANT:", "[SYSTEM",                                # Hermes process / system notices
    "<system-reminder>", "<command-", "<local-command-",     # Claude Code transcripts
    "[Request interrupted", "Caveat: The messages below",
)
_COMPACTION = "[CONTEXT COMPACTION"
_COMPACTION_END = "--- END OF CONTEXT SUMMARY"
# Notice blocks Hermes wraps around the customer's words: strip them, keep the rest.
_WRAPPER_BLOCKS = re.compile(
    r"\[Your active task list was preserved across context compression\]\n(?:- \[.*(?:\n|$))*"
    r"|\[Note: [^\]\n]*\]"
    r"|\[The user attached [^\]\n]*\]"
)


def extract_question(text) -> str:
    """The customer's own words in a user turn, or "" when there are none.

    Drops session-id placeholders and whole-message notices, strips the notice
    blocks platforms wrap around a real message (task list, model switch,
    unreadable attachment).
    """
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if t.startswith(_COMPACTION):
        _, _, t = t.partition(_COMPACTION_END)
        t = t.partition("\n")[2].strip()  # whatever follows the marker line
    if not t or _PLACEHOLDER.match(t) or t.startswith(_NOTICE_PREFIXES):
        return ""
    return _WRAPPER_BLOCKS.sub("", t).strip()


def is_question(text) -> bool:
    """True when `text` holds a real customer question the flywheel can re-encode."""
    return bool(extract_question(text))


def submit_trajectory(data: dict) -> None:
    """標準介面——任何 Producer（Claude hook、Hermes hook、Loom）調用此函數投遞軌跡。

    直接寫入共享 buffer 文件，跨進程可見。
    Claude Code hook、Hermes hook、飛輪 cron 各自獨立進程，in-memory list 無法共享。

    data 格式（標準化 v1）：
    {
        "producer": "claude-code",           # 來源標識
        "version": 1,                         # 介面版本
        "conversation_id": "...",
        "user_text": "...",                   # 被評的那一輪的問題（本卦由此起，訓練時重新編碼）；
                                              # 非顧客提問（空、系統注入）的軌跡不收

        "next_user_text": "...",              # 顧客對其回答的反應（下一輪提問）
        "trajectory": {...},                  # 獎勵信號
        # v2（可選）：提問時的決策 —— turn_id, base_hexagram, chosen_hexagram,
        #   candidates, candidate_q, chosen_prob, action, context_pre, …（見 DECISION_FIELDS）
        "embedding": [...],                   # 可選：預計算嵌入向量
    }
    """
    question = extract_question(data.get("user_text"))
    if not question:
        return  # nothing to re-encode: no 本卦, no training value
    trajectory = data.get("trajectory", {})
    sample = {
        "user_text": question,
        "next_user_text": extract_question(data.get("next_user_text")),
        "producer": data.get("producer", "unknown"),
        "conversation_id": data.get("conversation_id", ""),
        "hexagram_evolution": trajectory.get("hexagram_evolution", []),
        "timestamp": time.time(),
        "token_cost": int(trajectory.get("token_cost", 0)),
        "continued": bool(trajectory.get("continued", False)),
        "corrected": bool(trajectory.get("corrected", False)),
        "completed": bool(trajectory.get("completed", False)),
        "praised": bool(trajectory.get("praised", False)),
        "abandoned": bool(trajectory.get("abandoned", False)),
        "satisfaction": 0.0,
    }
    # v2: the decision taken at question time (DESIGN-phase4 §3.1)
    from yicenet.hook_engine.extractor import DECISION_FIELDS
    for k in ("turn_id",) + DECISION_FIELDS:
        if data.get(k) is not None:
            sample[k] = data[k]
    emb = data.get("embedding", [])
    if emb:
        sample["embedding"] = emb

    try:
        buf_path = yicenet_data_dir() / "flywheel_buffer.jsonl"
        buf_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(sample, ensure_ascii=False) + "\n"
        _append_locked(buf_path, line)
    except Exception:
        pass  # never block the caller


def _loom_to_yicenet(trajectory: dict) -> dict:
    """將 Loom 獎勵信號映射為 YiCeNet 內部 reward_sig 格式。"""
    return {
        "continued": trajectory.get("n_sessions", 1) > 1,
        "corrected": (trajectory.get("correction_rate", 0) or 0) > 0,
        "completed": trajectory.get("n_turns", 0) >= 1,
        "praised": False,           # Loom 暫無法判斷
        "abandoned": False,         # Loom 暫無法判斷
        "token_cost": trajectory.get("total_tokens", 0),
        "token_efficiency": trajectory.get("token_efficiency", 0),
    }


def default_sources():
    """Return the platform-adaptive list of DataSources for this machine.

    Each source is only included when its underlying store exists:
      - HermesDataSource      if ~/.hermes/state.db is present
      - ClaudeCodeDataSource  if ~/.claude/projects/ exists
      - FlywheelBufferSource  always (creates the file on first write)

    The flywheel calls scan_since() on each source and merges the results,
    so adding a new IDE later requires only adding its DataSource here.
    """
    from yicenet.datasource.hermes import HermesDataSource
    from yicenet.datasource.claude_code import ClaudeCodeDataSource
    from yicenet.datasource.buffer import FlywheelBufferSource

    sources = []
    h = HermesDataSource()
    if h.is_available():
        sources.append(h)
    c = ClaudeCodeDataSource()
    if c.is_available():
        sources.append(c)
    sources.append(FlywheelBufferSource())  # always — universal drop-zone
    return sources


def scan_all_sources(state: dict, sources=None) -> list[dict]:
    """Collect new samples from all DataSources and normalise to buffer schema.

    Replaces the Hermes-only scan_new_messages(); called by flywheel_run().
    `state` must contain 'last_run' (Unix timestamp float or None).
    """
    since = float(state.get("last_run") or 0.0)
    if sources is None:
        sources = default_sources()

    buffer_path = yicenet_data_dir() / "flywheel_buffer.jsonl"
    buffer_path.parent.mkdir(parents=True, exist_ok=True)

    all_samples: list[dict] = []
    for src in sources:
        try:
            raw_samples = src.scan_since(since)
        except Exception:
            continue

        # FlywheelBufferSource samples are already on disk — never re-write them.
        # All other sources (Hermes, Claude Code, …) get appended to the buffer.
        should_write = src.source_id != "buffer"
        if should_write:
            wf = open(buffer_path, "a", encoding="utf-8")
        try:
            for s in raw_samples:
                question = extract_question(s.user_text)
                if not question:
                    continue  # platform-injected / empty: not a customer question
                rec = {
                    "user_text": question,
                    "producer": s.source,
                    "conversation_id": s.conversation_id,
                    "timestamp": s.timestamp or time.time(),
                    "token_cost": s.token_cost,
                    "satisfaction": s.satisfaction,
                    "continued": s.continued,
                    "corrected": s.corrected,
                    "completed": s.completed,
                    "praised": s.praised,
                    "abandoned": s.abandoned,
                    "token_efficiency": s.response_length,
                    "source_msg_id": s.source_msg_id,
                }
                if s.embedding:
                    rec["embedding"] = s.embedding
                if should_write:
                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                all_samples.append(rec)
        finally:
            if should_write:
                wf.close()

    return all_samples


def scan_new_messages(state: dict) -> list[dict]:
    """Deprecated: use scan_all_sources() instead. Removal target: v17.0.0."""
    import warnings
    warnings.warn(
        "scan_new_messages() is deprecated, use scan_all_sources() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from yicenet.datasource.hermes import HermesDataSource
    h = HermesDataSource()
    if not h.is_available():
        return []
    return scan_all_sources(state, sources=[h])


def load_state() -> dict:
    """Load flywheel state."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

    # Scan checkpoints to find the highest existing version number
    highest = _highest_checkpoint_version()

    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)
        # Reconcile: if state's counter is behind registry, fast-forward
        if state.get("version_counter", 0) <= highest:
            state["version_counter"] = highest + 1
        return state

    return {
        "last_message_id": 0,
        "total_samples": 0,
        "version_counter": highest + 1,  # next after existing checkpoints
        "last_run": None,
        "runs": [],
    }


def _highest_checkpoint_version() -> int:
    """Return the highest numeric version among existing checkpoints, or 0."""
    try:
        versions = []
        for f in CHECKPOINT_DIR.glob("yicenet_v*.pt"):
            stem = f.stem  # e.g. "yicenet_v18"
            parts = stem.split("_v")
            if len(parts) == 2:
                try:
                    versions.append(int(parts[1]))
                except ValueError:
                    pass
        if versions:
            return max(versions)
    except Exception:
        pass
    # Also check registry as fallback
    try:
        if REGISTRY_PATH.exists():
            reg = json.loads(REGISTRY_PATH.read_text())
            ver = reg.get("active", {}).get("version", "")
            if ver.startswith("v"):
                return int(ver[1:])
    except Exception:
        pass
    return 0


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def flywheel_run():
    """Execute one flywheel cycle."""
    print("=" * 60)
    print(f"YiCeNet Flywheel (two-speed) — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    state = load_state()
    print(f"  Total samples processed: {state['total_samples']}")

    # ── Step 1: Scan all DataSources for new samples ──
    # scan_all_sources() writes Hermes / Claude Code samples to buffer_path;
    # FlywheelBufferSource samples are already in the file and are NOT re-written.
    print("\n  Step 1: Scanning all DataSources...")
    new_samples = scan_all_sources(state)
    new_count = len(new_samples)
    print(f"    Found {new_count} new samples across all sources")

    if not new_samples:
        print("    No new data. Skipping.")
        state["last_run"] = time.time()
        save_state(state)
        return

    # Count current buffer size (scan_all_sources already wrote to it)
    from yicenet.config import yicenet_data_dir
    buffer_path = yicenet_data_dir() / "flywheel_buffer.jsonl"
    total_buffer = 0
    if buffer_path.exists():
        with open(buffer_path, encoding="utf-8") as f:
            total_buffer = sum(1 for _ in f)
    print(f"    Buffer now holds {total_buffer} total samples")

    # ── Step 2: Build the training set (buffer + archive, §3.1 fields where present) ──
    print("\n  Step 2: Building the training set...")
    samples = load_training_samples(buffer_path)
    n_full = sum(1 for s in samples if s.get("chosen_hexagram") is not None)
    print(f"    {len(samples)} samples ({n_full} with the 之卦 decision, "
          f"{len(samples) - n_full} legacy: question + 本卦 only)")

    # ── Steps 3–4: World model V4 — train on the time split, evaluate, gate ──
    cfg = get_learning_config()
    new_since_wm = state.get("new_since_wm", 0) + new_count
    state["new_since_wm"] = new_since_wm
    has_active_wm = bool(((_load_registry().get("world_model") or {}).get("active")))
    wm_outcome = "skipped"
    if new_since_wm >= int(cfg["wm_min_new_samples"]) or not has_active_wm:
        print(f"\n  Step 3: Training World Model V4 ({new_since_wm} new since the last WM)...")
        try:
            wm_outcome = _train_and_gate_world_model(samples, cfg)
            state["new_since_wm"] = 0
        except Exception as exc:
            wm_outcome = f"failed: {exc}"
            print(f"    WM training failed: {exc}")
    else:
        print(f"\n  Step 3: Skipping WM training ({new_since_wm} new, "
              f"need {cfg['wm_min_new_samples']}+)")

    # ── Step 5: slow 卦理 channel (distillation + gates + shadow) ──
    print("\n  Step 5: 卦理 update channel...")
    try:
        prior_status = _prior_update_cycle(samples, cfg, state)
    except Exception as exc:
        prior_status = f"failed: {exc}"
    print(f"    {prior_status}")

    # ── Rotate buffer after a successful run ──
    _rotate_buffer(buffer_path)

    # ── Update state ──
    state["total_samples"] += new_count
    state["last_run"] = time.time()
    state["runs"].append({
        "timestamp": time.time(),
        "new_samples": new_count,
        "action": "trained",
        "world_model": wm_outcome,
        "prior": prior_status,
    })
    state["runs"] = state["runs"][-200:]
    save_state(state)


# ── Training set ──────────────────────────────────────────────────────────────

def load_training_samples(buffer_path: Path) -> list[dict]:
    """Every trajectory kept: the archive (rotated out of the buffer) plus the buffer.

    Rotation only manages file size; old samples still train, with power-law decay.
    """
    files = sorted((buffer_path.parent / "archive").glob("flywheel_buffer_*.jsonl"))
    files.append(buffer_path)
    seen, samples = set(), []
    for path in files:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    s = json.loads(line)
                except ValueError:
                    continue
                question = extract_question(s.get("user_text"))
                if not question:
                    continue
                key = (s.get("conversation_id"), s.get("timestamp"), question)
                if key in seen:
                    continue
                seen.add(key)
                samples.append({**s, "user_text": question})
    return samples


def _load_registry() -> dict:
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_registry(reg: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")


def _active_prior_checkpoint() -> Optional[Path]:
    """The prior in service: the registry's active model.

    Everything the flywheel derives from the prior (question features, candidate
    sets, distillation base) comes from the model actually answering customers.
    """
    try:
        active = json.loads(REGISTRY_PATH.read_text(encoding="utf-8")).get("active", {})
        path = CHECKPOINT_DIR / active.get("path", "")
        if active.get("path") and path.exists():
            return path
    except (OSError, ValueError):
        pass
    # No registry yet: the highest version number (not the lexicographically last name).
    def version_of(p: Path) -> int:
        m = re.match(r"yicenet_v(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1
    existing = [p for p in CHECKPOINT_DIR.glob("yicenet_v*.pt") if version_of(p) >= 0]
    return max(existing, key=version_of) if existing else None


def _load_active_prior():
    from yicenet.yicenet_engine import YiCeNetEngine
    path = _active_prior_checkpoint()
    engine = YiCeNetEngine(checkpoint=str(path) if path else "", project_root=str(YICENET_ROOT))
    engine._lazy_load()
    return engine._model.eval(), (_load_registry().get("active") or {}).get("version", "")


# ── World model V4 (察言观色) ──────────────────────────────────────────────────

def _train_and_gate_world_model(samples: list[dict], cfg: dict) -> str:
    """Train on the older samples, test on the newest, promote only on held-out wins.

    Promote when the candidate beats the base rate (skill > 0) and the active WM
    (lower log-loss on the same test set).  Returns a one-line outcome.
    """
    from yicenet.world_model_v4 import (
        QuestionEncoder, WorldModelV4, base_rates, build_dataset, evaluate,
        time_split, train_world_model,
    )

    if len(samples) < 20:
        print(f"    Too few samples ({len(samples)} < 20).")
        return "too few samples"
    prior, prior_version = _load_active_prior()
    data = build_dataset(samples, QuestionEncoder(prior))
    train_idx, test_idx = time_split(data.ts, float(cfg["test_fraction"]))
    train, test = data.subset(train_idx), data.subset(test_idx)
    rates = base_rates(train)

    wm = train_world_model(train)
    metrics = evaluate(wm, test, rates, int(cfg["min_full_test"]))
    _print_wm_metrics("candidate", metrics)

    reg = _load_registry()
    section = reg.setdefault("world_model", {"active": None, "ready": None, "fallback": None, "history": []})
    active = section.get("active")
    active_ll = None
    if active and active.get("path"):
        try:
            old = WorldModelV4.load(str(CHECKPOINT_DIR / active["path"]))
            old_metrics = evaluate(old, test, rates, int(cfg["min_full_test"]))
            active_ll = old_metrics["log_loss"]
            _print_wm_metrics(f"active {active['version']}", old_metrics)
        except Exception as exc:
            print(f"    Active WM unusable ({exc}); comparing with the base rate only.")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    entry = {
        "version": f"wm-{stamp}",
        "path": f"world_model_v4_{stamp}.pt",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": "full" if metrics["rerank_skill"] > 0 else "reduced",
        "skill": metrics["skill"],
        "rerank_skill": metrics["rerank_skill"],
        "log_loss": metrics["log_loss"],
        "base_log_loss": metrics["base_log_loss"],
        "n_train": len(train),
        "n_test": metrics["n"],
        "n_full_test": metrics["n_full"],
        "encoder": prior_version,
        "test_from": float(test.ts.min()) if len(test) else None,
        "metrics": metrics,
    }
    wm.meta = {k: v for k, v in entry.items() if k != "metrics"}
    wm.save(str(CHECKPOINT_DIR / entry["path"]))

    beats_base = metrics["beats_base"]  # bootstrap CI of the gain excludes 0
    beats_active = active_ll is None or metrics["log_loss"] < active_ll
    if beats_base and beats_active:
        if active:
            section.setdefault("history", []).append(active)
        section["fallback"] = active
        section["active"] = entry
        section["ready"] = None
        outcome = f"promoted {entry['version']} ({entry['mode']}, skill={entry['skill']:+.4f})"
    else:
        section["ready"] = entry
        why = ("does not beat the base rate (CI includes 0)" if not beats_base
               else "does not beat the active WM")
        outcome = f"kept as ready: {why}"
    section["history"] = section.get("history", [])[-20:]
    _save_registry(reg)
    print(f"    {outcome}")
    return outcome


def _print_wm_metrics(label: str, m: dict) -> None:
    lo, hi = m["skill_ci"]
    print(f"    [{label}] held-out n={m['n']}: log-loss {m['log_loss']:.4f} vs recent base rate "
          f"{m['base_log_loss']:.4f} → skill {m['skill']:+.4f} (CI [{lo:+.4f}, {hi:+.4f}]); "
          f"re-rank skill {m['rerank_skill']:+.4f} (n_full={m['n_full']})")
    for o, v in m["per_outcome"].items():
        a = f"{v['auc']:.3f}" if v["auc"] is not None else "n/a"
        print(f"      {o:<10} log-loss {v['log_loss']:.4f} (base {v['base_log_loss']:.4f})  AUC {a}")


# ── Slow 卦理 channel ─────────────────────────────────────────────────────────

def _prior_update_cycle(samples: list[dict], cfg: dict, state: dict) -> str:
    """Phase 3: needs a WM validated for choosing between candidates."""
    import torch
    from yicenet.prior_update import CycleContext, Items, run_cycle
    from yicenet.world_model_v4 import (
        QuestionEncoder, WorldModelV4, build_dataset, time_split, utility_of,
    )

    reg = _load_registry()
    entry = (reg.get("world_model") or {}).get("active")
    if not entry or float(entry.get("rerank_skill", 0.0) or 0.0) <= 0:
        return "skipped: no WM validated on 之卦 choices yet (re-rank skill ≤ 0)"
    if not reg.get("shadow"):
        last = float((reg.get("prior_update") or {}).get("last_attempt", 0.0))
        if time.time() - last < float(cfg["prior_update_days"]) * 86400:
            return f"waiting: at most one 卦理 attempt per {cfg['prior_update_days']} days"

    wm = WorldModelV4.load(str(CHECKPOINT_DIR / entry["path"]))
    prior, _ = _load_active_prior()
    encoder = QuestionEncoder(prior)
    data = build_dataset(samples, encoder)
    # Candidate sets as the prior generates them now; a logged set wins when the
    # sample recorded one for the same 本卦.
    with torch.no_grad():
        _, cands, _ = prior.evaluate_candidates(data.base, encoder.raw)
    for i, s in enumerate(samples):
        logged = s.get("candidates")
        if isinstance(logged, list) and len(logged) == cands.shape[1] and s.get("base_hexagram") == int(data.base[i]):
            cands[i] = torch.tensor(logged)
    items = Items(data.h, data.base, cands, data.ctx, data.has_ctx, data.producers, data.ts)

    weights = cfg["utility_weights"]

    def util(it, chosen, ctx, has_ctx):
        p = wm.proba(it.h, it.base, chosen.long(), torch.ones(len(it)), ctx, has_ctx)
        return utility_of(p, weights)

    train_idx, test_idx = time_split(data.ts, float(cfg["test_fraction"]))
    pool = torch.randperm(len(train_idx), generator=torch.Generator().manual_seed(0))[:16]
    pool_idx = torch.as_tensor(train_idx)[pool]

    cc = CycleContext(prior=prior, util=util, train=items.subset(train_idx),
                      held_out=items.subset(test_idx),
                      ctx_pool=data.ctx[pool_idx], has_pool=data.has_ctx[pool_idx])
    shadow = reg.get("shadow")
    if shadow:
        idx = [i for i, s in enumerate(samples)
               if s.get("shadow_version") == shadow.get("version")
               and s.get("shadow_chosen") is not None and s.get("chosen_hexagram") is not None]
        if idx:
            cc.shadow_items = items.subset(idx)
            cc.shadow_logged = torch.tensor(
                [[samples[i]["chosen_hexagram"], samples[i]["shadow_chosen"]] for i in idx])

    def next_version() -> str:
        v = state["version_counter"]
        state["version_counter"] = v + 1
        return f"v{v}"

    return run_cycle(cc, REGISTRY_PATH, CHECKPOINT_DIR, cfg, next_version)


FLYWHEEL_LOG_MAX_BYTES = 5 * 1024 * 1024


def _setup_output() -> None:
    """Console: UTF-8 so CJK and arrows print on Windows.  No console (pythonw from
    the scheduled task, sys.stdout is None): append to ~/.yicenet/logs/flywheel.log.

    Only for `python -m yicenet.flywheel` — importers (daemon, hooks) keep their streams.
    """
    if sys.stdout is None or sys.stderr is None:
        log_dir = Path.home() / ".yicenet" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "flywheel.log"
        try:
            if log_path.stat().st_size > FLYWHEEL_LOG_MAX_BYTES:
                log_path.replace(log_path.with_suffix(".log.1"))
        except OSError:
            pass
        log = open(log_path, "a", encoding="utf-8", errors="replace", buffering=1)
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] flywheel run (pid {os.getpid()})")
        return
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


if __name__ == "__main__":
    _setup_output()
    flywheel_run()
