"""
RL fine-tuning for YiCeNet using the trained world model.

The world model predicts reward = f(task_embedding, hexagram).
RL training adjusts the Gumbel router to pick hexagrams with higher predicted rewards.

This is the first time YiCeNet's actions genuinely depend on the input.
"""

import os, sys, json, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def rl_train_world_model(
    checkpoint_in: str = "checkpoints/yicenet_v3.pt",
    checkpoint_out: str = "checkpoints/yicenet_v4.pt",
    world_model_path: str = "checkpoints/world_model_best.pt",
    episodes: int = 1000,
    lr: float = 3e-4,
    epochs_per_step: int = 1,
):
    print("=" * 60)
    print("YiCeNet RL Fine-tuning (World Model Driven)")
    print("=" * 60)

    from src.model import YiCeNet, count_parameters
    from src.config import YiCeNetConfig
    from src.data.dataset import SessionDataset
    from src.world_model import WorldModel
    from src.tokenizer import encode as yicenet_encode

    # ── Load YiCeNet ──
    config = YiCeNetConfig()
    model = YiCeNet(config).to(DEVICE)
    saved = torch.load(checkpoint_in, map_location=DEVICE, weights_only=False)
    model.load_state_dict(saved["model_state_dict"], strict=False)
    if "tau" in saved:
        model.tau = saved["tau"]
    print(f"  Loaded YiCeNet from {checkpoint_in}")

    # ── Load world model ──
    wm = WorldModel.load(world_model_path, DEVICE)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False
    print(f"  Loaded world model from {world_model_path}")

    # ── Load session data ──
    ds = SessionDataset(max_seq_len=config.max_seq_len)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    print(f"  {len(ds)} session samples available for RL")

    # ── Trainable: router + value_net + action_decoder ──
    trainable = list(model.router.parameters()) + \
                list(model.value_net.parameters()) + \
                list(model.action_decoder.parameters()) + \
                [model.trigram_prototypes] + \
                list(model.trigram_cross_attn.parameters())
    
    # Also tune the encoder slightly (low LR)
    for p in model.encoder.parameters():
        p.requires_grad = False  # keep encoder frozen
    # But allow the state_proj (last layer) to adapt
    for p in model.encoder.state_proj.parameters():
        p.requires_grad = True
    trainable += list(model.encoder.state_proj.parameters())

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, episodes // 50))

    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")
    print(f"\n  Training for {episodes} episodes...\n")

    model.train()
    episode_rewards = []
    best_avg_reward = float("-inf")

    # Pre-compute all task embeddings for speed
    print("  Pre-computing task embeddings...")
    all_h = []
    all_texts = []
    wm.eval()
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            ids = sample["input_ids"].unsqueeze(0).to(DEVICE)
            mask = sample["attention_mask"].unsqueeze(0).to(DEVICE)
            h = model.encode_context(ids, mask)  # (1, 256)
            all_h.append(h.squeeze(0))
            all_texts.append(sample.get("text", ""))
    all_h = torch.stack(all_h)  # (N, 256)
    print(f"  Pre-computed {len(all_h)} embeddings")

    for episode in range(episodes):
        # Sample a random task
        idx = np.random.randint(0, len(all_h))
        h = all_h[idx].unsqueeze(0).to(DEVICE)  # (1, 256)

        # ── Forward pass ──
        # Generate candidates
        hexagram_idx, hex_probs = model.router(h, tau=max(model.tau, 0.05), hard=False)
        # hexagram_idx: (1,), hex_probs: (1, 64)

        # Get 8 candidates through structural reasoning
        _, cand_idxs, cand_values = model.evaluate_candidates(hexagram_idx, h)
        # cand_idxs: (1, 8), cand_values: (1, 8, 1)

        # World model scores ALL 64 hexagrams for this task
        # This gives us the TRUE expected reward for each hexagram
        with torch.no_grad():
            all_scores = wm.forward(h, torch.arange(64, device=DEVICE).unsqueeze(0).expand(1, -1))
            # all_scores: (1, 64)
            true_best_hex = all_scores.argmax(dim=-1).item()

        # The world model reward for the SAMPLED hexagram
        with torch.no_grad():
            sampled_reward = wm.forward(h, hexagram_idx).item()

        # Value targets: the world model's score for each candidate
        cand_scores = all_scores[0, cand_idxs.squeeze(0)]  # (8,)

        # ── Value loss: value_net should match world model scores ──
        value_loss = F.mse_loss(cand_values.squeeze(-1), cand_scores.unsqueeze(0))

        # ── Policy gradient: increase prob of hexagrams with high world model scores ──
        # Use the world model's prediction as the advantage signal
        hex_probs_sq = hex_probs.squeeze(0)  # (64,)
        # The advantage of each hexagram = (world_model_score - average_score)
        avg_score = all_scores.mean()
        advantages = all_scores - avg_score  # (1, 64)
        advantages = advantages.squeeze(0)  # (64,)

        # Policy gradient loss: -log_prob * advantage
        log_probs = torch.log(hex_probs_sq.clamp(min=1e-8))
        policy_loss = -(log_probs * advantages * 0.1).sum()  # scale 0.1 for stability

        # Entropy bonus for exploration
        entropy = -(hex_probs_sq * log_probs).sum()
        entropy_loss = -0.01 * entropy

        # Total loss
        loss = policy_loss + value_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        # Track progress
        episode_rewards.append(sampled_reward)

        # Decay temperature
        model.decay_temperature()

        # Logging
        if (episode + 1) % 100 == 0:
            recent = episode_rewards[-100:]
            avg_r = np.mean(recent)
            tau = model.get_temperature()

            # Evaluate: how often does the model's top choice match the world model's?
            model.eval()
            with torch.no_grad():
                # Model's top hexagram (via value net + Gumbel during training, or argmax here)
                eval_h = all_h[0:1].to(DEVICE)
                router_out, _ = model.router(eval_h, tau=0.01, hard=True)  # nearly deterministic
                model_hex = router_out.item()
                wm_best = all_scores[0].argmax().item()
                match = "✓" if model_hex == wm_best else "✗"
            model.train()

            print(f"  Episode {episode+1:>4}/{episodes} — "
                  f"avg_reward: {avg_r:+.3f} | "
                  f"tau: {tau:.3f} | "
                  f"model_hex: {model_hex:2d} | "
                  f"wm_best: {wm_best:2d} {match} | "
                  f"policy_loss: {policy_loss.item():.4f} | "
                  f"value_loss: {value_loss.item():.4f}")

            # Save best
            if avg_r > best_avg_reward:
                best_avg_reward = avg_r
                model.save_pretrained(checkpoint_out.replace(".pt", "_best.pt"))
                print(f"    ✓ New best (avg_reward={avg_r:.3f})")

        scheduler.step()

    # Save final
    model.save_pretrained(checkpoint_out)
    print(f"\n  ✓ Final model saved to {checkpoint_out}")
    print(f"  Best avg reward: {best_avg_reward:.3f}")

    return episode_rewards


if __name__ == "__main__":
    rl_train_world_model(episodes=1000)
