#!/usr/bin/env python3
"""
API-supervised YiCeNet RL training pipeline.
Reads buffer samples + API evaluations → trains World Model → RL fine-tune → v{N}
"""
import json, os, sys, time, math, random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

# Resolve project root relative to this script's location
YICENET_ROOT = Path(__file__).resolve().parent.parent

# Allow env var override (e.g., when installed as pip package)
YICENET_ROOT = Path(os.environ.get("YICENET_ROOT", str(YICENET_ROOT)))
sys.path.insert(0, str(YICENET_ROOT))
os.chdir(str(YICENET_ROOT))

from yicenet.config import YiCeNetConfig
from yicenet.model import YiCeNet
from yicenet.world_model import WorldModelV2, power_law_weight
from yicenet.rl_train import project_to_hexagram_space, compute_hexagram_reward
from yicenet.tokenizer import encode as yicenet_encode

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = YICENET_ROOT / "checkpoints"

WM_SLOW_TAU = 30.0
WM_FAST_TAU = 3.0
WM_ALPHA = 1.5
WM_BETA = 0.3
RL_EPISODES = 200
RL_LR = 3e-4


def load_eval_results(path: Path) -> dict[int, dict]:
    """Load evaluation results keyed by msg_id."""
    results = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            results[r["msg_id"]] = r
    return results


def load_buffer(path: Path) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def extract_probes_from_text(model: YiCeNet, text: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract probes and hexagram prediction from text."""
    ids, mask = yicenet_encode(text, max_len=128)
    ids, mask = ids.to(device), mask.to(device)
    
    with torch.no_grad():
        out = model(ids, mask, tau=0.01, hard=True)
        probes = out["probes"].cpu().squeeze(0)  # (1, 9) → (9,)
        hex_id = out["hexagram_idx"].cpu().squeeze(0)  # (1,) → scalar
    return probes, hex_id


def supervised_wm_training(
    wm: WorldModelV2,
    model: YiCeNet,
    samples: list[dict],
    eval_results: dict[int, dict],
    config: YiCeNetConfig,
    device: str,
    endogenous: bool = False,
) -> WorldModelV2:
    """Train World Model with Supervised targets."""
    wm.train()
    optimizer = torch.optim.AdamW(wm.parameters(), lr=1e-4)
    now = time.time()
    
    total_loss_a = 0.0
    total_loss_b = 0.0
    total_count = 0
    
    print(f"  Training WM on {len(samples)} Supervised samples...")
    
    for idx, s in enumerate(samples):
        msg_id = s.get("msg_id", 0)
        text = s.get("user_text", "")
        ts = s.get("timestamp", now)
        
        if msg_id not in eval_results:
            continue
        
        ds = eval_results[msg_id]
        signals = ds.get("signals", {})
        satisfaction = ds.get("satisfaction", 0.0)
        
        # Extract probes from YiCeNet
        try:
            probes, hex_id = extract_probes_from_text(model, text, device)
        except Exception as e:
            print(f"    [{idx}] msg_id={msg_id} probe extraction failed: {e}")
            continue
        
        probes, hex_id = probes.to(device), hex_id.to(device)
        
        # Target hexagram distribution from eval signals
        reward_sig = {
            "continued": signals.get("continued", False),
            "corrected": signals.get("corrected", False),
            "completed": signals.get("completed", False),
            "praised": signals.get("praised", False),
            "abandoned": signals.get("abandoned", False),
        }
        target_dist = project_to_hexagram_space(
            reward_sig,
            temperature=config.ext_projection_temperature,
            continuation_w=config.ext_continuation_weight,
            correction_w=config.ext_correction_weight,
            completion_w=config.ext_completion_weight,
            satisfaction=satisfaction,
        ).to(device)
        
        # Target external vector with eval satisfaction score
        target_ext = torch.tensor(
            [0.5, 0.5, max(0.0, min(1.0, (satisfaction + 1.0) / 2.0))],
            dtype=torch.float32, device=device
        )
        
        # Power-law weights
        w_slow = power_law_weight(ts, now, WM_SLOW_TAU, WM_ALPHA)
        w_fast = power_law_weight(ts, now, WM_FAST_TAU, WM_ALPHA)
        
        # 全內生噪聲感知：用 WM 預測驚訝度作為額外權重
        if endogenous:
            endo_weight = wm.compute_endogenous_weight(
                probes.unsqueeze(0).to(device),
                hex_id.unsqueeze(0).to(device),
                target_dist.unsqueeze(0),
            ).item()
            w_slow *= endo_weight
            w_fast *= endo_weight
        
        # Forward through WM
        pred_dist, pred_ext = wm(probes.unsqueeze(0), hex_id.unsqueeze(0))
        
        # Weighted loss
        loss_a = w_slow * F.kl_div(
            pred_dist.clamp(min=1e-8).log(),
            target_dist.unsqueeze(0).clamp(min=1e-8),
            reduction="sum",
        )
        loss_b = w_fast * (pred_ext - target_ext.unsqueeze(0)).pow(2).mean()
        loss = loss_a + WM_BETA * loss_b
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss_a += loss_a.item()
        total_loss_b += loss_b.item()
        total_count += 1
        
        if (idx + 1) % 40 == 0:
            print(f"    [{idx+1}/{len(samples)}] loss_A={total_loss_a/max(total_count,1):.4f} loss_B={total_loss_b/max(total_count,1):.4f}")
    
    avg_loss_a = total_loss_a / max(total_count, 1)
    avg_loss_b = total_loss_b / max(total_count, 1)
    print(f"  Supervised WM training done: {total_count} samples, "
          f"avg loss_A={avg_loss_a:.4f} loss_B={avg_loss_b:.4f}")
    
    return wm


def rl_fine_tune_v14(
    model: YiCeNet,
    wm: WorldModelV2,
    samples: list[dict],
    eval_results: dict[int, dict],
    config: YiCeNetConfig,
    device: str,
    version: str,
    episodes: int = RL_EPISODES,
) -> YiCeNet:
    """RL fine-tuning with 64-dim projection reward using Supervised WM."""
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False
    
    # Build training pairs (sample + API result)
    train_pairs = []
    for s in samples:
        msg_id = s.get("msg_id", 0)
        if msg_id in eval_results:
            train_pairs.append((s, eval_results[msg_id]))
    
    random.shuffle(train_pairs)
    
    # Trainable params
    trainable = (
        list(model.router.parameters())
        + list(model.value_net.parameters())
        + list(model.action_decoder.parameters())
        + [model.trigram_prototypes]
        + list(model.trigram_cross_attn.parameters())
    )
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.encoder.state_proj.parameters():
        p.requires_grad = True
    trainable += list(model.encoder.state_proj.parameters())
    
    optimizer = torch.optim.AdamW(trainable, lr=RL_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, episodes // 50)
    )
    
    print(f"\n  RL fine-tuning: {episodes} episodes, {len(train_pairs)} training pairs")
    model.train()
    
    episode_rewards = []
    best_avg_reward = float("-inf")
    now = time.time()
    
    for ep in range(episodes):
        pair = train_pairs[ep % len(train_pairs)]
        s, ds = pair
        text = s.get("user_text", "")
        ts = s.get("timestamp", now)
        signals = ds.get("signals", {})
        satisfaction = ds.get("satisfaction", 0.0)
        
        try:
            ids, mask = yicenet_encode(text, max_len=128)
            ids, mask = ids.to(device), mask.to(device)
        except Exception:
            continue
        
        # Forward through YiCeNet
        out = model(ids, mask, tau=max(0.5, 1.0 - ep / episodes), hard=False)
        probes = out["probes"]  # (1, 9)
        hex_id = out["hexagram_idx"]  # (1,)
        probs = out["hexagram_probs"]  # (1, 64)
        
        # WM prediction
        with torch.no_grad():
            wm_dist, _ = wm(probes.to(device).unsqueeze(0), hex_id.to(device))
        
        # API target
        reward_sig = {
            "continued": signals.get("continued", False),
            "corrected": signals.get("corrected", False),
            "completed": signals.get("completed", False),
            "praised": signals.get("praised", False),
            "abandoned": signals.get("abandoned", False),
        }
        target_dist = project_to_hexagram_space(
            reward_sig,
            temperature=config.ext_projection_temperature,
            satisfaction=satisfaction,
        ).to(device)
        
        # Reward = similarity between WM prediction and API target
        reward = compute_hexagram_reward(wm_dist, target_dist.unsqueeze(0))
        
        # Power-law weight
        w = power_law_weight(ts, now, WM_SLOW_TAU, WM_ALPHA)
        weighted_reward = reward * w
        
        # REINFORCE loss
        log_prob = torch.log(probs.gather(1, hex_id.unsqueeze(1)) + 1e-8)
        loss = -(log_prob * weighted_reward).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        
        episode_rewards.append(reward.item())
        
        if (ep + 1) % 40 == 0:
            avg = sum(episode_rewards[-40:]) / 40
            best_avg_reward = max(best_avg_reward, avg)
            print(f"    Episode {ep+1}/{episodes}: avg_reward={avg:.4f} best={best_avg_reward:.4f}")
    
    avg_reward = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
    print(f"  RL fine-tuning done: avg_reward_last100={avg_reward:.4f} best={best_avg_reward:.4f}")
    
    # Save
    out_path = CHECKPOINT_DIR / f"yicenet_{version}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "version": version,
        "samples": len(train_pairs),
        "avg_reward": avg_reward,
        "best_avg_reward": best_avg_reward,
        "api_supervised": True,
        "episodes": episodes,
    }, out_path)
    print(f"  Saved: {out_path}")
    
    # Also save as latest
    latest_path = CHECKPOINT_DIR / f"yicenet_{version}_latest.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "version": version,
        "avg_reward": avg_reward,
    }, latest_path)
    print(f"  Saved: {latest_path}")
    
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="e.g. r1, r2, ...")
    parser.add_argument("--buffer", required=True, help="flywheel_buffer.jsonl path (relative to YICENET_ROOT)")
    parser.add_argument("--eval-results", required=True, help="Evaluation .jsonl path (relative to YICENET_ROOT)")
    parser.add_argument("--start", type=int, default=0, help="start sample index")
    parser.add_argument("--end", type=int, default=None, help="end sample index")
    parser.add_argument("--endogenous", action="store_true", help="use WM prediction surprise as noise filter")
    args = parser.parse_args()
    
    # Resolve paths relative to YICENET_ROOT
    buffer_path = YICENET_ROOT / args.buffer
    eval_path = YICENET_ROOT / args.eval_results
    
    print(f"\n{'='*60}")
    print(f"API-supervised Training: {args.version}")
    print(f"{'='*60}")
    
    # Load data
    samples = load_buffer(buffer_path)
    if args.end:
        samples = samples[args.start:args.end]
    elif args.start:
        samples = samples[args.start:]
    print(f"Buffer samples: {len(samples)}")
    
    eval_results = load_eval_results(eval_path)
    print(f"Eval results: {len(eval_results)}")
    
    # Filter samples with Eval results
    valid = [(i, s) for i, s in enumerate(samples) if s.get("msg_id", 0) in eval_results]
    print(f"Valid pairs: {len(valid)}/{len(samples)}")
    
    # Load YiCeNet v4 base
    config = YiCeNetConfig()
    model = YiCeNet(config).to(DEVICE)
    base_path = CHECKPOINT_DIR / "yicenet_v4.pt"
    if base_path.exists():
        saved = torch.load(str(base_path), map_location=DEVICE, weights_only=False)
        model.load_state_dict(saved["model_state_dict"], strict=False)
        print(f"Loaded base model: {base_path}")
    else:
        print(f"WARNING: No base model at {base_path}")
    
    model.eval()
    for p in model.encoder.parameters():
        p.requires_grad = False
    
    # Load World Model
    wm_path = CHECKPOINT_DIR / "world_model_best.pt"
    if wm_path.exists():
        try:
            wm = WorldModelV2.load(str(wm_path), DEVICE)
            print(f"Loaded WM: {wm_path}")
        except KeyError as e:
            print(f"  Old WM format ({e}), starting fresh")
            wm = WorldModelV2().to(DEVICE)
    else:
        wm = WorldModelV2().to(DEVICE)
        print("Fresh WM (no checkpoint)")
    
    # Step 1: Supervised WM training
    print(f"\n--- Step 1: Supervised World Model training ---")
    wm = supervised_wm_training(wm, model, samples, eval_results, config, DEVICE, endogenous=args.endogenous)
    
    # Save updated WM
    wm.save(str(CHECKPOINT_DIR / f"world_model_v14_{args.version}.pt"))
    # Also save as best (for next round to load)
    wm.save(str(wm_path))
    print(f"  WM saved: world_model_v14_{args.version}.pt")
    
    # Step 2: RL fine-tuning
    print(f"\n--- Step 2: RL fine-tuning ---")
    model = rl_fine_tune_v14(model, wm, samples, eval_results, config, DEVICE, args.version)
    
    # Record metrics
    metrics = {
        "version": f"v14_{args.version}",
        "samples": len(valid),
        "wm_loss_A": None,  # filled during training
        "avg_reward": None,  # filled during training
    }
    
    metrics_path = CHECKPOINT_DIR / f"v14_{args.version}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"v14_{args.version} complete! Metrics: {metrics_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
