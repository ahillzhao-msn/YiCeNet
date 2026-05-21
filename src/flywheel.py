"""
YiCeNet Online Flywheel — daily cron job for continuous learning.

Pipeline:
  1. Scan Hermes session DB for new user messages since last check
  2. Append new data to training buffer
  3. Incrementally update world model (a few epochs on new data only)
  4. RL fine-tune on combined dataset
  5. Register new checkpoint as 'ready' in registry.json for A/B switch

State tracking: ~/.hermes/data/yicenet_flywheel.json
  - last_message_id: last processed message ID
  - total_samples: cumulative sample count
  - version_counter: next version number (v5, v6, ...)
"""

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

# ── Paths ──
YICENET_ROOT = Path(__file__).parent.parent
STATE_FILE = Path.home() / ".hermes" / "data" / "yicenet_flywheel.json"
DB_PATH = str(Path.home() / ".hermes" / "state.db")
CHECKPOINT_DIR = YICENET_ROOT / "checkpoints"
REGISTRY_PATH = CHECKPOINT_DIR / "registry.json"

sys.path.insert(0, str(YICENET_ROOT))


def load_state() -> dict:
    """Load flywheel state (last processed message, version counter)."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "last_message_id": 0,
        "total_samples": 0,
        "version_counter": 5,  # start at v5
        "last_run": None,
        "runs": [],
    }


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def scan_new_messages(state: dict) -> list[dict]:
    """
    Scan Hermes session DB for new user messages since last check.

    Returns list of {user_text, reward, token_cost, session_id, msg_id}
    """
    last_id = state.get("last_message_id", 0)
    conn = sqlite3.connect(DB_PATH)

    # Find user→assistant pairs with their follow-up
    rows = conn.execute("""
        SELECT m1.id, m1.content, m1.session_id, m1.timestamp,
               m2.id as asst_id,
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

    samples = []
    for msg_id, content, session_id, ts, asst_id, next_text in rows:
        # Determine reward from follow-up
        reward = _compute_reward(next_text, content)
        samples.append({
            "msg_id": msg_id,
            "user_text": content,
            "session_id": session_id,
            "reward": reward,
            "has_follow_up": next_text is not None,
        })

    return samples


def _compute_reward(next_user_text: str | None, current_text: str) -> float:
    """Compute simplified reward from follow-up message."""
    from src.data.dataset import _detect_correction, _detect_completion

    if next_user_text is None:
        # No follow-up → likely abandoned
        if _detect_completion(current_text):
            return 0.5
        return -1.5

    if _detect_correction(next_user_text):
        return -1.0
    if _detect_completion(next_user_text):
        return 0.5
    return 0.3  # normal continuation


def flywheel_run():
    """Execute one flywheel cycle."""
    print("=" * 60)
    print(f"YiCeNet Flywheel — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    state = load_state()
    print(f"  Last message ID: {state['last_message_id']}")
    print(f"  Total samples processed: {state['total_samples']}")
    print(f"  Next version: v{state['version_counter']}")

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
    buffer_path = YICENET_ROOT / "data" / "flywheel_buffer.jsonl"
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

    # Need minimum 20 new samples to bother training
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

    # ── Step 3: Incremental world model update ──
    print("\n  Step 3: Updating world model...")
    _update_world_model(buffer_path)

    # ── Step 4: RL fine-tune ──
    print("\n  Step 4: RL fine-tuning...")
    version = f"v{state['version_counter']}"
    new_checkpoint = _rl_fine_tune(version, buffer_path)

    # ── Step 5: Register as 'ready' in registry.json ──
    print(f"\n  Step 5: Registering {version} as ready...")
    _register_ready(version, new_checkpoint)

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

    # ── Clean buffer after training ──
    # Actually keep buffer for combined training next time
    print(f"\n  ✓ Flywheel complete: {version} ready for A/B switch")


def _update_world_model(buffer_path: Path):
    """Incremental world model update on combined old+new data."""
    import torch
    from src.world_model import WorldModel

    # Load existing world model
    wm_path = CHECKPOINT_DIR / "world_model_best.pt"
    if wm_path.exists():
        wm = WorldModel.load(str(wm_path), "cuda" if torch.cuda.is_available() else "cpu")
        print(f"    Loaded existing world model from {wm_path}")
    else:
        wm = WorldModel().to("cuda" if torch.cuda.is_available() else "cpu")
        print("    No existing world model, starting fresh")

    # Quick fine-tune on buffer data
    # (Full training pipeline would be better, but for cron we do incremental)
    from src.yicenet_engine import YiCeNetEngine
    from src.tokenizer import encode

    engine = YiCeNetEngine(project_root=str(YICENET_ROOT))
    # Force model load
    engine._lazy_load()
    device = next(engine._model.parameters()).device

    wm.train()
    optimizer = torch.optim.AdamW(wm.parameters(), lr=1e-4)

    # Load buffer
    samples = []
    with open(buffer_path) as f:
        for line in f:
            samples.append(json.loads(line))

    import random
    random.shuffle(samples)
    batch = samples[:min(50, len(samples))]

    total_loss = 0.0
    for s in batch:
        text = s["user_text"]
        target = torch.tensor([s["reward"]], dtype=torch.float32, device=device)

        # Encode and get h
        ids, mask = encode(text, max_len=128)
        ids, mask = ids.to(device), mask.to(device)

        with torch.no_grad():
            h = engine._model.encode_context(ids, mask)

        # Score all 64 hexagrams
        all_hex = torch.arange(64, device=device).unsqueeze(0)
        pred = wm.forward(h, all_hex)
        loss = torch.nn.functional.mse_loss(pred.mean().unsqueeze(0), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / max(len(batch), 1)
    print(f"    Incremental update: {len(batch)} samples, avg loss={avg_loss:.4f}")

    # Save updated world model
    wm.save(str(CHECKPOINT_DIR / "world_model_best.pt"))
    print(f"    World model saved")


def _rl_fine_tune(version: str, buffer_path: Path) -> str:
    """Run short RL fine-tune and save checkpoint."""
    import torch
    from src.model import YiCeNet
    from src.config import YiCeNetConfig
    from src.world_model import WorldModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load existing v4 as base
    config = YiCeNetConfig()
    model = YiCeNet(config).to(device)
    base_path = CHECKPOINT_DIR / "yicenet_v4.pt"
    if base_path.exists():
        saved = torch.load(str(base_path), map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state_dict"], strict=False)
        print(f"    Loaded base model from {base_path}")

    # Load world model
    wm = WorldModel.load(str(CHECKPOINT_DIR / "world_model_best.pt"), device)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    # Load buffer data
    samples = []
    with open(buffer_path) as f:
        for line in f:
            samples.append(json.loads(line))

    if len(samples) < 10:
        print(f"    Too few samples ({len(samples)}). Skipping RL.")
        return str(CHECKPOINT_DIR / f"yicenet_{version}.pt")

    # Short RL training (200 episodes on buffer data)
    from src.tokenizer import encode
    trainable = list(model.router.parameters()) + \
                list(model.value_net.parameters())
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.encoder.state_proj.parameters():
        p.requires_grad = True
    trainable += list(model.encoder.state_proj.parameters())

    optimizer = torch.optim.AdamW(trainable, lr=2e-4)
    model.train()

    episodes = min(200, len(samples) * 5)
    for ep in range(episodes):
        import random
        s = random.choice(samples)
        text = s["user_text"]

        ids, mask = encode(text, max_len=128)
        ids, mask = ids.to(device), mask.to(device)

        h = model.encode_context(ids, mask)
        hex_idx, _ = model.router(h, tau=max(model.tau, 0.1), hard=False)

        with torch.no_grad():
            wm_score = wm.forward(h, hex_idx)

        # Simple policy gradient
        probs = torch.nn.functional.softmax(model.router.projection(h), dim=-1)
        log_prob = torch.log(probs[0, hex_idx[0]].clamp(min=1e-8))
        loss = -log_prob * wm_score.squeeze()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.decay_temperature()

        if (ep + 1) % 50 == 0:
            print(f"      RL ep {ep+1}/{episodes} — loss={loss.item():.4f}")

    # Save
    out_path = CHECKPOINT_DIR / f"yicenet_{version}.pt"
    model.save_pretrained(str(out_path))
    print(f"    Saved {out_path}")

    # Also symlink/copy as rl_best for backward compat
    import shutil
    shutil.copy(str(out_path), str(CHECKPOINT_DIR / "yicenet_rl_best.pt"))

    return str(out_path)


def _register_ready(version: str, checkpoint_path: str):
    """Register new checkpoint as 'ready' in registry.json."""
    import json

    if not REGISTRY_PATH.exists():
        reg = {"active": None, "ready": None, "fallback": None, "history": []}
    else:
        with open(REGISTRY_PATH) as f:
            reg = json.load(f)

    reg["ready"] = {
        "version": version,
        "path": checkpoint_path,
        "avg_reward": 0.0,
        "win_rate": 0.0,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "notes": f"Flywheel auto-train {time.strftime('%Y-%m-%d')}",
    }
    reg["history"].append(reg["ready"])

    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)

    print(f"    {version} registered as 'ready' in registry.json")
    print(f"    run 'yicenet_switch' or wait for auto-switch")


if __name__ == "__main__":
    flywheel_run()
