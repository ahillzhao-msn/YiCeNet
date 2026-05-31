"""
YiCeNet v5 Online Flywheel — continuous learning with power-law decay.

Pipeline:
  1. Scan Hermes session DB for new user messages since last check
  2. Extract external vectors (token_cost, response_length, satisfaction)
  3. Append to buffer with timestamps for power-law weighting
  4. Incrementally update World Model v2 (dual-head, weighted by power-law)
  5. RL fine-tune v5 (64-dim projection reward)
  6. Register new checkpoint as 'ready' in registry.json for A/B switch
"""

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

# ── Paths (dual-mode: YICENET_HOME env var > auto-detect) ──
from yicenet.config import yicenet_home, yicenet_data_dir, yicenet_checkpoint_dir

YICENET_ROOT = yicenet_home()
CHECKPOINT_DIR = yicenet_checkpoint_dir()
REGISTRY_PATH = CHECKPOINT_DIR / "registry.json"
STATE_FILE = Path.home() / ".hermes" / "data" / "yicenet_flywheel.json"
DB_PATH = str(Path.home() / ".hermes" / "state.db")

# ── Power law parameters (match config.py defaults) ──
WM_SLOW_TAU_DAYS = 30.0
WM_FAST_TAU_DAYS = 3.0
WM_ALPHA = 1.5
WM_BETA = 0.3

# ── 外部 Producer 緩衝區 ──
# 任何系統（Loom、其他）調用 submit_trajectory() 投遞結構化軌跡，
# 由 flywheel_run() 的 cron tick 消費。

FLYWHEEL_BUFFER: list[dict] = []


def submit_trajectory(data: dict) -> None:
    """標準介面——任何 Producer 調用此函數投遞軌跡。

    data 格式（標準化 v1）：
    {
        "producer": "loom",                  # 來源標識
        "version": 1,                         # 介面版本
        "conversation_id": "...",
        "trajectory": {...},                  # 獎勵信號（由 reward_for_flywheel() 產生）
        "embedding": [0.1, 0.2, ...],         # 可選：預計算的嵌入向量
        "raw_messages": [...],                # 可選：原始訊息（當 embedding 未提供時）
    }
    """
    FLYWHEEL_BUFFER.append(data)


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


def _consume_external_buffer(buffer_path: Path) -> int:
    """消費外部 Producer 投遞的軌跡，寫入 buffer 供訓練使用。

    Returns:
        寫入的樣本數
    """
    if not FLYWHEEL_BUFFER:
        return 0

    count = 0
    os.makedirs(os.path.dirname(buffer_path), exist_ok=True)
    with open(buffer_path, "a") as f:
        while FLYWHEEL_BUFFER:
            item = FLYWHEEL_BUFFER.pop(0)
            trajectory = item.get("trajectory", {})
            reward_sig = _loom_to_yicenet(trajectory)

            # 構建標準樣本
            sample = {
                "user_text": f"[{item.get('producer', 'external')}] "
                             f"{item.get('conversation_id', '?')}",
                "producer": item.get("producer", "unknown"),
                "conversation_id": item.get("conversation_id", ""),
                "hexagram_evolution": trajectory.get("hexagram_evolution", []),
                "timestamp": time.time(),
                "token_cost": reward_sig.get("token_cost", 0),
                "continued": reward_sig.get("continued", False),
                "corrected": reward_sig.get("corrected", False),
                "completed": reward_sig.get("completed", False),
                "praised": reward_sig.get("praised", False),
                "abandoned": reward_sig.get("abandoned", False),
                "satisfaction": 0.0,
            }
            # 若有 embedding，一併寫入
            emb = item.get("embedding", [])
            if emb:
                sample["embedding"] = emb

            f.write(json.dumps(sample) + "\n")
            count += 1

    return count


def load_state() -> dict:
    """Load flywheel state."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "last_message_id": 0,
        "total_samples": 0,
        "version_counter": 14,  # continue from v14 (v5 onwards)
        "last_run": None,
        "runs": [],
    }


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def scan_new_messages(state: dict) -> list[dict]:
    """
    Scan Hermes session DB for new user messages since last check.

    Returns list of {
        user_text, assistant_text, next_user_text,
        timestamp, satisfaction, token_cost, response_length,
        session_id, msg_id, continued, corrected, completed, praised, abandoned
    }
    """
    last_id = state.get("last_message_id", 0)
    conn = sqlite3.connect(DB_PATH)

    rows = conn.execute("""
        SELECT m1.id, m1.content, m1.session_id, m1.timestamp,
               m2.content as asst_content,
               (SELECT content FROM messages m3
                WHERE m3.session_id = m1.session_id
                  AND m3.role = 'user'
                  AND m3.timestamp > m2.timestamp
                ORDER BY m3.timestamp LIMIT 1) as next_user_text
        FROM messages m1
        JOIN messages m2 ON m2.session_id = m1.session_id
            AND m2.role = 'assistant'
            AND m2.timestamp > m1.timestamp
            AND m2.timestamp = (
                SELECT MIN(timestamp) FROM messages
                WHERE session_id = m1.session_id
                  AND role = 'assistant'
                  AND timestamp > m1.timestamp
            )
        WHERE m1.role = 'user'
          AND m1.id > ?
          AND m1.content IS NOT NULL
          AND length(m1.content) > 3
        ORDER BY m1.id
        LIMIT 100
    """, (last_id,)).fetchall()

    conn.close()

    from yicenet.external_metrics import (
        compute_satisfaction,
        estimate_token_cost,
        estimate_response_length,
    )

    samples = []
    for msg_id, content, session_id, ts, asst_text, next_text in rows:
        # Multi-level satisfaction
        satisfaction = compute_satisfaction(next_text, content)
        token_cost = estimate_token_cost(asst_text or content)
        response_len = estimate_response_length(next_text or "")

        # Boolean signals for hexagram projection
        from yicenet.external_metrics import (
            _check_patterns, _CORRECTION_PATTERNS, _COMPLETION_PATTERNS,
            _PRAISE_PATTERNS, _ABANDON_PATTERNS,
        )
        next_str = next_text or ""
        corrected = _check_patterns(next_str, _CORRECTION_PATTERNS)
        completed = _check_patterns(next_str, _COMPLETION_PATTERNS)
        praised = _check_patterns(next_str, _PRAISE_PATTERNS)
        abandoned = _check_patterns(next_str, _ABANDON_PATTERNS) or next_text is None
        continued = next_text is not None and not abandoned

        samples.append({
            "msg_id": msg_id,
            "user_text": content,
            "assistant_text": asst_text or "",
            "next_user_text": next_text or "",
            "session_id": session_id,
            "timestamp": ts,
            "satisfaction": satisfaction,
            "token_cost": token_cost,
            "response_length": response_len,
            "continued": continued,
            "corrected": corrected,
            "completed": completed,
            "praised": praised,
            "abandoned": abandoned,
        })

    return samples


def flywheel_run():
    """Execute one flywheel cycle (v5)."""
    print("=" * 60)
    print(f"YiCeNet v5 Flywheel — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    state = load_state()
    print(f"  Last message ID: {state['last_message_id']}")
    print(f"  Total samples processed: {state['total_samples']}")
    print(f"  Next version: v{state['version_counter']}")

    # ── Step 0: Consume external producer buffer ──
    buffer_path = yicenet_data_dir() / "flywheel_buffer.jsonl"
    ext_count = _consume_external_buffer(buffer_path)
    if ext_count:
        print(f"    Consumed {ext_count} external trajectories")

    # ── Step 1: Scan new data ──
    print("\n  Step 1: Scanning for new messages...")
    new_samples = scan_new_messages(state)
    print(f"    Found {len(new_samples)} new samples")

    if not new_samples:
        print("    No new data. Skipping.")
        state["last_run"] = time.time()
        save_state(state)
        return

    # ── Step 2: Append to training buffer ──
    print("\n  Step 2: Appending to training buffer...")
    os.makedirs(os.path.dirname(buffer_path), exist_ok=True)

    new_count = 0
    with open(buffer_path, "a") as f:
        for s in new_samples:
            f.write(json.dumps(s) + "\n")
            new_count += 1

    total_buffer = 0
    if buffer_path.exists():
        with open(buffer_path) as f:
            total_buffer = sum(1 for _ in f)

    print(f"    Appended {new_count} samples (buffer now {total_buffer})")

    # Need minimum 20 samples to bother training
    if total_buffer < 20:
        print(f"    Buffer too small ({total_buffer} < 20). Deferring training.")
        state["last_message_id"] = max(s["msg_id"] for s in new_samples)
        state["total_samples"] += new_count
        state["last_run"] = time.time()
        state["runs"].append({
            "timestamp": time.time(),
            "new_samples": new_count,
            "action": "deferred",
        })
        save_state(state)
        return

    # ── Step 3: Incremental world model v2 update ──
    print("\n  Step 3: Updating World Model v2 (power-law weighted)...")
    _update_world_model_v2(buffer_path)

    # ── Step 4: RL fine-tune v5 ──
    print("\n  Step 4: RL fine-tuning v5 (64-dim projection reward)...")
    version = f"v{state['version_counter']}"
    new_checkpoint = _rl_fine_tune_v5(version, buffer_path)

    # ── Step 5: Register as 'ready' ──
    print(f"\n  Step 5: Registering {version} as ready...")
    _register_ready(version, new_checkpoint)

    # ── Step 6: Evaluate new model ──
    print(f"\n  Step 6: Evaluating {version} on buffer data...")
    _record_evaluation(version, buffer_path, checkpoint_path=new_checkpoint)

    # ── Step 7: Auto-promote ──
    _auto_promote(buffer_path)

    # ── Update state ──
    state["last_message_id"] = max(s["msg_id"] for s in new_samples)
    state["total_samples"] += new_count
    state["version_counter"] += 1
    state["last_run"] = time.time()
    state["runs"].append({
        "timestamp": time.time(),
        "new_samples": new_count,
        "version": version,
        "action": "trained",
    })
    save_state(state)


def _update_world_model_v2(buffer_path: Path):
    """
    Incremental World Model v2 update with power-law weighting.

    Uses dual-head loss (headA=64-dim distribution, headB=ext vector)
    weighted by power-law forgetting curve.
    """
    import torch
    import random
    import torch.nn.functional as F
    from yicenet.config import YiCeNetConfig
    from yicenet.world_model import WorldModelV2, power_law_weight
    from yicenet.yicenet_engine import YiCeNetEngine
    from yicenet.tokenizer import encode
    from yicenet.rl_train import project_to_hexagram_space

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # Load existing World Model v2
    wm_path = CHECKPOINT_DIR / "world_model_best.pt"
    if wm_path.exists():
        wm = WorldModelV2.load(str(wm_path), device_str)
        print(f"    Loaded existing World Model v2 from {wm_path}")
    else:
        wm = WorldModelV2().to(device)
        print("    No existing World Model v2, starting fresh")

    # Load YiCeNet engine for probe extraction
    config = YiCeNetConfig()
    engine = YiCeNetEngine(project_root=str(YICENET_ROOT))
    engine._lazy_load()
    model = engine._model
    model.eval()

    # Load buffer
    samples = []
    with open(buffer_path) as f:
        for line in f:
            samples.append(json.loads(line))

    random.shuffle(samples)
    batch = samples[:min(100, len(samples))]

    wm.train()
    optimizer = torch.optim.AdamW(wm.parameters(), lr=1e-4)
    now = time.time()

    total_loss_a = 0.0
    total_loss_b = 0.0
    total_count = 0

    for s in batch:
        text = s["user_text"]
        ts = s.get("timestamp", now)

        # Encode and get probes
        ids, mask = encode(text, max_len=128)
        ids, mask = ids.to(device), mask.to(device)

        with torch.no_grad():
            out = model(ids, mask, tau=0.01, hard=True)
            probes_t = out["probes"].to(device)  # (9,)
            hex_id = out["hexagram_idx"]

        # Target hexagram distribution from reward signals
        reward_sig = {
            "continued": s.get("continued", True),
            "corrected": s.get("corrected", False),
            "completed": s.get("completed", False),
            "praised": s.get("praised", False),
            "abandoned": s.get("abandoned", False),
        }
        target_dist = project_to_hexagram_space(
            reward_sig,
            temperature=config.ext_projection_temperature,
            continuation_w=config.ext_continuation_weight,
            correction_w=config.ext_correction_weight,
            completion_w=config.ext_completion_weight,
        ).to(device)

        # Target external vector
        target_ext = torch.tensor(
            [s.get("token_cost", 0.5),
             s.get("response_length", 0.5),
             s.get("satisfaction", 0.0)],
            dtype=torch.float32, device=device
        )

        # Power-law weights
        w_slow = power_law_weight(ts, now, WM_SLOW_TAU_DAYS, WM_ALPHA)
        w_fast = power_law_weight(ts, now, WM_FAST_TAU_DAYS, WM_ALPHA)

        # 全內生噪聲感知：WM 自己判斷噪聲
        try:
            endo_w = wm.compute_endogenous_weight(
                probes_t.unsqueeze(0), hex_id,
                target_dist.unsqueeze(0),
            ).item()
            w_slow *= endo_w
            w_fast *= endo_w
        except Exception:
            pass  # 永不中斷飛輪

        # Forward through WM
        pred_dist, pred_ext = wm(probes_t.unsqueeze(0), hex_id)

        # Weighted loss
        loss_a = (w_slow * F.kl_div(
            pred_dist.clamp(min=1e-8).log(),
            target_dist.unsqueeze(0).clamp(min=1e-8),
            reduction="sum",
        ))
        loss_b = (w_fast * (pred_ext - target_ext.unsqueeze(0)).pow(2).mean())

        loss = loss_a + WM_BETA * loss_b

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_a += loss_a.item()
        total_loss_b += loss_b.item()
        total_count += 1

    import torch.nn.functional as F

    avg_loss_a = total_loss_a / max(total_count, 1)
    avg_loss_b = total_loss_b / max(total_count, 1)
    print(f"    Incremental update: {total_count} samples, "
          f"avg loss_A={avg_loss_a:.4f} loss_B={avg_loss_b:.4f}")

    # Save updated world model
    wm.save(str(CHECKPOINT_DIR / "world_model_best.pt"))
    print(f"    World Model v2 saved to {CHECKPOINT_DIR / 'world_model_best.pt'}")


def _rl_fine_tune_v5(version: str, buffer_path: Path) -> str:
    """Run short RL fine-tune v5 with 64-dim projection reward."""
    import torch
    import random
    import torch.nn.functional as F
    from yicenet.model import YiCeNet
    from yicenet.config import YiCeNetConfig
    from yicenet.world_model import WorldModelV2, power_law_weight
    from yicenet.tokenizer import encode
    from yicenet.rl_train import project_to_hexagram_space, compute_hexagram_reward

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load base model
    config = YiCeNetConfig()
    model = YiCeNet(config).to(device)
    # Load from existing checkpoint if available
    existing = sorted(CHECKPOINT_DIR.glob("yicenet_v*.pt"))
    if existing:
        base_path = existing[-1]  # latest by name (highest version)
        saved = torch.load(str(base_path), map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state_dict"], strict=False)
        print(f"    Loaded base model from {base_path}")

    # Load World Model v2
    wm_path = CHECKPOINT_DIR / "world_model_best.pt"
    if wm_path.exists():
        wm = WorldModelV2.load(str(wm_path), device)
    else:
        print(f"    Warning: no World Model v2 found, using random init")
        wm = WorldModelV2().to(device)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    # Load buffer
    samples = []
    with open(buffer_path) as f:
        for line in f:
            samples.append(json.loads(line))

    if len(samples) < 10:
        print(f"    Too few samples ({len(samples)}). Skipping RL.")
        return str(CHECKPOINT_DIR / f"yicenet_{version}.pt")

    # Trainable: router + value_net + state_proj
    trainable = list(model.router.parameters()) + \
                list(model.value_net.parameters())
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.encoder.state_proj.parameters():
        p.requires_grad = True
    trainable += list(model.encoder.state_proj.parameters())

    optimizer = torch.optim.AdamW(trainable, lr=2e-4)
    model.train()
    now = time.time()

    episodes = min(200, len(samples) * 5)
    for ep in range(episodes):
        s = random.choice(samples)
        text = s["user_text"]
        ts = s.get("timestamp", now)

        ids, mask = encode(text, max_len=128)
        ids, mask = ids.to(device), mask.to(device)

        with torch.no_grad():
            out = model(ids, mask, tau=max(model.tau, 0.05), hard=False)
            probes_t = out["probes"].to(device)
            hex_id = out["hexagram_idx"]

        # Target distribution from reward signals
        reward_sig = {
            "continued": s.get("continued", True),
            "corrected": s.get("corrected", False),
            "completed": s.get("completed", False),
            "praised": s.get("praised", False),
            "abandoned": s.get("abandoned", False),
        }
        target_dist = project_to_hexagram_space(
            reward_sig,
            temperature=config.ext_projection_temperature,
        ).to(device)

        # WM prediction
        with torch.no_grad():
            wm_pred_dist, _ = wm(probes_t.unsqueeze(0), hex_id)

        # Reward = distribution similarity
        reward = compute_hexagram_reward(wm_pred_dist, target_dist.unsqueeze(0))

        # Policy gradient
        probs = F.softmax(model.router.projection(out["h"]), dim=-1)
        log_prob = torch.log(probs[0, hex_id[0]].clamp(min=1e-8))
        loss = -log_prob * reward.squeeze()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.decay_temperature()

        if (ep + 1) % 50 == 0:
            print(f"      RL ep {ep+1}/{episodes} — reward={reward.item():.4f} loss={loss.item():.4f}")

    # Save
    out_path = CHECKPOINT_DIR / f"yicenet_{version}.pt"
    model.save_pretrained(str(out_path))
    print(f"    Saved {out_path}")

    # Also copy as rl_best for backward compat
    import shutil
    shutil.copy(str(out_path), str(CHECKPOINT_DIR / "yicenet_rl_best.pt"))

    return str(out_path)


def _register_ready(version: str, checkpoint_path: str):
    """Register new checkpoint as 'ready' in registry.json."""
    if not REGISTRY_PATH.exists():
        reg = {"active": None, "ready": None, "fallback": None, "history": []}
    else:
        with open(REGISTRY_PATH) as f:
            reg = json.load(f)

    reg["ready"] = {
        "version": version,
        "path": os.path.relpath(checkpoint_path, str(CHECKPOINT_DIR)),
        "avg_reward": 0.0,
        "win_rate": 0.0,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "notes": f"Flywheel v5 auto-train {time.strftime('%Y-%m-%d')}",
    }
    reg["history"].append(reg["ready"])

    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)

    print(f"    {version} registered as 'ready' in registry.json")
    print(f"    run 'yicenet_switch' or wait for auto-switch")


def _record_evaluation(version: str, buffer_path: Path, checkpoint_path: str = ""):
    """Evaluate model on buffer data and write to metrics.db for dashboard."""
    import sqlite3
    import torch
    from yicenet.yicenet_engine import YiCeNetEngine
    from yicenet.tokenizer import encode

    db_path = YICENET_ROOT / "data" / "metrics.db"
    engine = YiCeNetEngine(checkpoint=checkpoint_path, project_root=str(YICENET_ROOT))
    engine._lazy_load()
    device = next(engine._model.parameters()).device

    samples = []
    with open(buffer_path) as f:
        for line in f:
            samples.append(json.loads(line))

    if not samples:
        print("    No samples for evaluation, skipping.")
        return

    total_reward = 0.0
    wins = 0
    episode_data = []
    engine._model.eval()

    with torch.no_grad():
        for s in samples:
            text = s["user_text"]
            next_text = s.get("next_user_text", "")

            ids, mask = encode(text, max_len=128)
            ids, mask = ids.to(device), mask.to(device)

            out = engine._model(ids, mask, tau=0.1, hard=True)
            hex_idx = out["hexagram_idx"]

            # Reward from 64-dim projection
            from yicenet.rl_train import project_to_hexagram_space, compute_hexagram_reward
            from yicenet.world_model import WorldModelV2
            from yicenet.config import YiCeNetConfig

            config = YiCeNetConfig()
            wm_path = CHECKPOINT_DIR / "world_model_best.pt"
            if wm_path.exists():
                wm = WorldModelV2.load(str(wm_path), device)
                probes_t = out["probes"].to(device)
                wm_pred, _ = wm(probes_t.unsqueeze(0), hex_idx)
            else:
                wm_pred = None

            reward_sig = {
                "continued": s.get("continued", True),
                "corrected": s.get("corrected", False),
                "completed": s.get("completed", False),
                "praised": s.get("praised", False),
                "abandoned": s.get("abandoned", False),
            }
            target_dist = project_to_hexagram_space(reward_sig)

            if wm_pred is not None:
                reward_val = compute_hexagram_reward(
                    wm_pred.cpu(), target_dist.unsqueeze(0)
                ).item()
            else:
                reward_val = s.get("satisfaction", 0.0)

            total_reward += reward_val
            if reward_val > 0.5:
                wins += 1

            episode_data.append({
                "hexagram_id": hex_idx[0].item(),
                "reward": reward_val,
                "action_id": out["action_ids"][0].item(),
            })

    avg_reward = total_reward / len(samples)
    win_rate = wins / len(samples)

    # Write to metrics.db
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE IF NOT EXISTS evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version TEXT NOT NULL,
        avg_reward REAL NOT NULL,
        win_rate REAL NOT NULL,
        episodes INTEGER NOT NULL,
        duration_sec REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS hexagram_usage (
        date TEXT NOT NULL,
        hexagram_id INTEGER NOT NULL,
        count INTEGER NOT NULL DEFAULT 0,
        avg_q_value REAL NOT NULL DEFAULT 0.0,
        PRIMARY KEY (date, hexagram_id)
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS trajectories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        hexagram_id INTEGER NOT NULL,
        reward REAL NOT NULL,
        terminal_type TEXT NOT NULL,
        latency_ms INTEGER NOT NULL DEFAULT 0,
        token_cost INTEGER NOT NULL DEFAULT 0
    )""")
    conn.execute("""
        INSERT INTO evaluations (version, avg_reward, win_rate, episodes, duration_sec)
        VALUES (?, ?, ?, ?, ?)
    """, (version, avg_reward, win_rate, len(samples), 0.0))

    date_str = time.strftime("%Y-%m-%d")
    for ep in episode_data:
        conn.execute("""
            INSERT INTO hexagram_usage (date, hexagram_id, count, avg_q_value)
            VALUES (?, ?, 1, 0.0)
            ON CONFLICT(date, hexagram_id)
            DO UPDATE SET count = count + 1
        """, (date_str, ep["hexagram_id"]))

    for ep in episode_data:
        conn.execute("""
            INSERT INTO trajectories
                (session_id, hexagram_id, reward, terminal_type, latency_ms, token_cost)
            VALUES (?, ?, ?, ?, 0, 0)
        """, (
            f"flywheel_eval_{version}",
            ep["hexagram_id"],
            ep["reward"],
            "success" if ep["reward"] > 0.5 else "abandoned",
        ))

    conn.commit()
    conn.close()

    print(f"    avg_reward={avg_reward:.4f}, win_rate={win_rate:.2%}, n={len(samples)}")
    print(f"    Written to metrics.db")

    # Update registry.json ready metrics
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            reg = json.load(f)
        if reg.get("ready") and reg["ready"]["version"] == version:
            reg["ready"]["avg_reward"] = round(avg_reward, 4)
            reg["ready"]["win_rate"] = round(win_rate, 4)
            with open(REGISTRY_PATH, "w") as f:
                json.dump(reg, f, indent=2)
            print(f"    registry.json['ready'] metrics updated: {avg_reward=:.4f} {win_rate=:.2%}")


def _auto_promote(buffer_path: Path):
    """Evaluate active model, compare with ready, promote if ready wins."""
    import torch
    from yicenet.yicenet_engine import YiCeNetEngine
    from yicenet.tokenizer import encode
    from yicenet.rl_train import project_to_hexagram_space, compute_hexagram_reward

    if not REGISTRY_PATH.exists():
        print("    No registry.json, skipping auto-promote.")
        return

    with open(REGISTRY_PATH) as f:
        reg = json.load(f)

    active = reg.get("active")
    ready = reg.get("ready")
    if not active or not ready:
        print("    No active or ready entry, skipping auto-promote.")
        return

    samples = []
    with open(buffer_path) as f:
        for line in f:
            samples.append(json.loads(line))

    if not samples:
        print("    No buffer data, skipping auto-promote.")
        return

    # Evaluate active model
    print(f"\n  Step 7: Evaluating active model ({active['version']}) for comparison...")
    engine = YiCeNetEngine(project_root=str(YICENET_ROOT))
    engine._lazy_load()
    device = next(engine._model.parameters()).device

    active_wins = 0
    engine._model.eval()
    with torch.no_grad():
        for s in samples:
            ids, mask = encode(s["user_text"], max_len=128)
            ids, mask = ids.to(device), mask.to(device)
            out = engine._model(ids, mask, tau=0.1, hard=True)

            # Evaluate using 64-dim projection
            reward_sig = {
                "continued": s.get("continued", True),
                "corrected": s.get("corrected", False),
                "completed": s.get("completed", False),
                "praised": s.get("praised", False),
                "abandoned": s.get("abandoned", False),
            }
            target_dist = project_to_hexagram_space(reward_sig)
            from yicenet.world_model import WorldModelV2
            wm_path = CHECKPOINT_DIR / "world_model_best.pt"
            if wm_path.exists():
                wm = WorldModelV2.load(str(wm_path), device)
                probes_t = out["probes"].to(device)
                wm_pred, _ = wm(probes_t.unsqueeze(0), out["hexagram_idx"])
                reward_val = compute_hexagram_reward(
                    wm_pred.cpu(), target_dist.unsqueeze(0)
                ).item()
            else:
                reward_val = s.get("satisfaction", 0.0)

            if reward_val > 0.5:
                active_wins += 1

    active_win_rate = active_wins / len(samples)
    ready_win_rate = ready.get("win_rate", 0.0)

    print(f"    {active['version']} win_rate={active_win_rate:.2%}    "
          f"{ready['version']} win_rate={ready_win_rate:.2%}")

    # Promote if ready wins by >= 3%
    if ready_win_rate >= active_win_rate + 0.03:
        print(f"    ✓ {ready['version']} outperforms {active['version']} by "
              f"{ready_win_rate - active_win_rate:.1%} — promoting!")
        result = engine.check_for_switch()
        if result and result.get("should_switch"):
            print(f"    ✓ Promoted to {result['new_version']} "
                  f"(avg_reward={result['new_avg_reward']:.4f})")
        else:
            print(f"    ⚠ check_for_switch returned {result}")
    else:
        delta = (ready_win_rate - active_win_rate) * 100
        if delta > 0:
            print(f"    Ready ahead by {delta:.1f}% but below 3% threshold — keeping active.")
        else:
            print(f"    Active still ahead by {-delta:.1f}% — no switch.")


if __name__ == "__main__":
    flywheel_run()
