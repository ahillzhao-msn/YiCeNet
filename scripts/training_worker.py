"""
YiCeNet Training Worker — standalone CPU-based PPO fine-tuning.

Reads trajectories from SQLite, runs PPO updates, evaluates new model,
and writes to registry.json for A/B switching.

Run as:
    python3 scripts/training_worker.py --loop        # continuous loop
    python3 scripts/training_worker.py --once        # single training batch
    python3 scripts/training_worker.py --evaluate    # eval only

Called by Hermes cron: every 2 hours or every 500 new trajectories.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np

from src.model import YiCeNet
from src.config import YiCeNetConfig
from src.metrics import MetricsLogger


REGISTRY_PATH = os.path.join(Path(__file__).parent.parent, "checkpoints", "registry.json")
CHECKPOINT_DIR = os.path.join(Path(__file__).parent.parent, "checkpoints")
MAX_NEW_TRAJS = 500        # trigger training when this many new trajectories
PPO_EPOCHS = 4
PPO_CLIP = 0.2
GAMMA = 0.99
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
LR = 3e-4
BATCH_SIZE = 64


def load_registry() -> dict:
    """Load model registry."""
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {
        "active": None,
        "ready": None,
        "fallback": None,
        "history": [],
    }


def save_registry(reg: dict):
    """Save model registry."""
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)


def count_new_trajectories(since_version: str = None) -> int:
    """Count trajectories since last training."""
    ml = MetricsLogger()
    stats = ml.get_stats()
    return stats.get("total_trajectories", 0)


def train_once(device: str = "cpu") -> dict:
    """
    Execute one PPO training batch on CPU.

    In production, this reads real trajectories from SQLite.
    For now, uses synthetic data to validate the pipeline.
    """
    print(f"[Training Worker] Starting PPO training on {device}...")
    start_time = time.time()

    # Load current best model as starting point
    config = YiCeNetConfig()
    model = YiCeNet(config).to(device).train()

    # Try loading from registry
    reg = load_registry()
    if reg.get("active") and os.path.exists(reg["active"]["path"]):
        saved = torch.load(reg["active"]["path"], map_location=device, weights_only=False)
        # Filter: only load compatible keys (skip decoder if shapes mismatch)
        model_state = model.state_dict()
        compatible = {k: v for k, v in saved["model_state_dict"].items()
                      if k in model_state and model_state[k].shape == v.shape}
        model_state.update(compatible)
        model.load_state_dict(model_state)
        print(f"  Loaded from {reg['active']['path']} ({len(compatible)}/{len(saved['model_state_dict'])} params compatible)")
    else:
        # Fall back to default checkpoint
        default_ckpt = os.path.join(CHECKPOINT_DIR, "yicenet_rl_best.pt")
        if os.path.exists(default_ckpt):
            saved = torch.load(default_ckpt, map_location=device, weights_only=False)
            # Only load encoder, router, hexagram embed, value net — skip decoder
            exclude_prefixes = ("action_decoder.",)
            model_state = model.state_dict()
            compatible = {
                k: v for k, v in saved["model_state_dict"].items()
                if k in model_state and model_state[k].shape == v.shape
                and not any(k.startswith(p) for p in exclude_prefixes)
            }
            model_state.update(compatible)
            model.load_state_dict(model_state)
            print(f"  Loaded encoder+router from default ({len(compatible)}/{len(saved['model_state_dict'])} params)")
        else:
            print("  No checkpoint found, starting from scratch")

    # Trainable params: router + value network + decoder
    trainable = list(model.router.parameters()) + \
                list(model.value_net.parameters()) + \
                list(model.action_decoder.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=LR)

    # Simulate a few PPO steps (in production, read from SQLite)
    n_steps = 100
    total_loss = 0.0
    model.train()

    for step in range(n_steps):
        # Synthetic batch (in production: real trajectories)
        B = BATCH_SIZE
        seq_len = 16
        input_ids = torch.randint(1, config.vocab_size, (B, seq_len)).to(device)
        attention_mask = torch.ones(B, seq_len).to(device)

        # Forward
        output = model(input_ids, attention_mask, tau=model.tau, hard=False)

        # Compute surrogate loss (simplified PPO)
        action_logits = output["action_logits"]
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        # Simulated advantages (in production: GAE from trajectory rewards)
        advantages = torch.randn(B, device=device) * 0.1
        returns = advantages + 0.5

        # PPO clipped surrogate
        ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        values = output["candidate_values"].mean(dim=(1, 2))
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus
        entropy = dist.entropy().mean()

        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        total_loss += loss.item()

    # Done training
    duration = time.time() - start_time
    avg_loss = total_loss / n_steps
    model.eval()

    # Generate new version
    version = f"v{len(reg['history']) + 1}"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"yicenet_{version}.pt")
    torch.save({"model_state_dict": model.state_dict(), "tau": model.tau}, checkpoint_path)

    # Quick evaluation on synthetic data
    with torch.no_grad():
        eval_rewards = []
        for _ in range(10):
            dummy_ids = torch.randint(1, config.vocab_size, (1, 16)).to(device)
            dummy_mask = torch.ones(1, 16).to(device)
            out = model(dummy_ids, dummy_mask, tau=0.1, hard=True)
            # Simulate reward based on action distribution
            action_probs = F.softmax(out["action_logits"], dim=-1)
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum()
            reward = 0.5 - 0.1 * entropy.item() + max(out["candidate_values"].mean().item(), 0)
            eval_rewards.append(reward)

    avg_reward = np.mean(eval_rewards)
    win_rate = np.mean([r > 0 for r in eval_rewards])

    # Log evaluation
    MetricsLogger().log_evaluation(version, avg_reward, win_rate, n_steps, duration)

    # Update registry
    entry = {
        "version": version,
        "path": checkpoint_path,
        "avg_reward": round(avg_reward, 4),
        "win_rate": round(win_rate, 4),
        "created": datetime.now().isoformat(),
    }

    reg["ready"] = entry
    reg["history"].append(entry)
    # Keep last 10
    reg["history"] = reg["history"][-10:]
    save_registry(reg)

    result = {
        "version": version,
        "avg_reward": round(avg_reward, 4),
        "win_rate": round(win_rate, 4),
        "duration_sec": round(duration, 1),
        "steps": n_steps,
        "checkpoint": checkpoint_path,
    }
    print(f"  ✓ Training complete: {json.dumps(result, indent=2)}")
    return result


def evaluate_ready() -> dict | None:
    """
    Evaluate if ready model beats active model.
    Returns switch recommendation.
    """
    reg = load_registry()
    if not reg.get("ready") or not reg.get("active"):
        return None

    ready = reg["ready"]
    active = reg["active"]

    # Simple comparison: win rate threshold
    ready_wr = ready.get("win_rate", 0)
    active_wr = active.get("win_rate", 0)

    decision = {
        "ready_version": ready["version"],
        "active_version": active["version"],
        "ready_win_rate": ready_wr,
        "active_win_rate": active_wr,
        "should_switch": ready_wr > active_wr + 0.05,
        "reason": "",
    }

    if decision["should_switch"]:
        decision["reason"] = (
            f"Ready {ready['version']} (WR={ready_wr:.2f}) > "
            f"Active {active['version']} (WR={active_wr:.2f}) + 5% threshold"
        )
    else:
        decision["reason"] = (
            f"Ready {ready['version']} (WR={ready_wr:.2f}) not sufficiently "
            f"better than Active {active['version']} (WR={active_wr:.2f})"
        )

    return decision


def apply_switch():
    """
    Promote ready model to active, move old active to history.
    """
    reg = load_registry()
    if not reg.get("ready"):
        print("[AB Switch] No ready model to promote")
        return False

    # Current active becomes fallback
    if reg.get("active"):
        reg["fallback"] = reg["active"]

    # Ready becomes active
    reg["active"] = reg["ready"]
    reg["ready"] = None
    save_registry(reg)

    print(f"[AB Switch] Switched to {reg['active']['version']}")
    return True


def run_once():
    """Single training batch + evaluate + auto-switch."""
    print("=" * 50)
    print(f"YiCeNet Training Worker — {datetime.now().isoformat()}")
    print("=" * 50)

    # Step 1: Train
    result = train_once(device="cpu")

    # Step 2: Evaluate
    decision = evaluate_ready()
    if decision:
        print(f"\n  Evaluation: {decision['reason']}")
        if decision["should_switch"]:
            apply_switch()
            print("  ✓ A/B switch executed")
    else:
        print("\n  No active model yet — first training run")

    # Step 3: Log
    MetricsLogger().log_evaluation(
        result["version"], result["avg_reward"],
        result["win_rate"], result["steps"], result["duration_sec"]
    )

    print("=" * 50)
    return result


def run_loop(interval_minutes: int = 120):
    """Continuous training loop."""
    print(f"[Training Worker] Starting loop (interval={interval_minutes}min)")
    while True:
        try:
            run_once()
        except Exception as e:
            print(f"[Training Worker] Error: {e}")
        print(f"\nSleeping {interval_minutes} minutes...\n")
        time.sleep(interval_minutes * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YiCeNet Training Worker")
    parser.add_argument("--once", action="store_true", help="Single training batch")
    parser.add_argument("--loop", action="store_true", help="Continuous training loop")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate only, no training")
    parser.add_argument("--interval", type=int, default=120, help="Loop interval (minutes)")
    parser.add_argument("--switch", action="store_true", help="Force A/B switch now")
    args = parser.parse_args()

    if args.evaluate:
        decision = evaluate_ready()
        print(json.dumps(decision, indent=2) if decision else "No ready model")
    elif args.switch:
        apply_switch()
    elif args.loop:
        run_loop(args.interval)
    else:
        run_once()
