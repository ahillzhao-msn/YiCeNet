"""
YiCeNet training pipeline.

Two stages:
  1. Unsupervised Pre-train: K-means clustering on synthetic orchestration
     features to initialize hexagram embeddings (太极生两仪 → 八卦 → 六十四卦)
  2. RL Fine-tune: REINFORCE/PPO to optimize orchestration quality

Usage:
    python train.py --stage pretrain --num_samples 10000
    python train.py --stage rl --episodes 5000
    python train.py --stage all  # full pipeline
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yicenet.config import YiCeNetConfig
from yicenet.model import YiCeNet, count_parameters

# Dataset — session-based is default; synthetic fallback kept in function body
from yicenet.data.dataset import SessionDataset, DataDrivenEnv


def pretrain_stage(
    model: YiCeNet,
    config: YiCeNetConfig,
    dataset: Optional[Dataset] = None,
    num_samples: int = 10000,
    batch_size: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints",
):
    """
    Stage 1: Unsupervised pre-training.

    Uses a two-phase approach:
    1. Collect feature representations from synthetic orchestration traces
    2. K-means cluster → initialize hexagram embeddings
    3. Fine-tune embeddings with contrastive loss

    Philosophy: "类万物之情" — clustering gives each hexagram
    a natural affinity for a class of orchestration scenarios.
    """
    print("\n" + "=" * 60)
    print("STAGE 1: Unsupervised Pre-training")
    print("=" * 60)

    # Use provided dataset or fall back to synthetic
    if dataset is not None:
        print(f"  Using real session dataset ({len(dataset)} samples)")
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        total_features = len(dataset)
    else:
        # Synthetic fallback: generate random token sequences
        print(f"  Using synthetic data ({num_samples} random samples)")
        class _RandomDataset(Dataset):
            def __init__(self, n, vocab, seq_len):
                self.n = n; self.vocab = vocab; self.seq_len = seq_len
            def __len__(self): return self.n
            def __getitem__(self, i):
                import random as _r
                seq = [_r.randint(1, self.vocab - 1) for _ in range(self.seq_len)]
                m = [1] * self.seq_len
                return {"input_ids": torch.tensor(seq, dtype=torch.long),
                        "attention_mask": torch.tensor(m, dtype=torch.long),
                        "features": torch.zeros(8), "cluster_id": torch.tensor(0)}
        syn = _RandomDataset(num_samples, config.vocab_size, min(config.max_seq_len, 16))
        dataloader = DataLoader(
            syn, batch_size=batch_size, shuffle=True, num_workers=0
        )
        total_features = num_samples

    # Phase 1: Collect encoder features
    print(f"\nPhase 1: Collecting {total_features} feature representations...")
    model.eval()
    all_features = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Use encoder to get features
            h = model.encode_context(input_ids, attention_mask)
            all_features.append(h.cpu())

    all_features = torch.cat(all_features, dim=0)  # (N, D)
    print(f"  Collected features: {all_features.shape}")

    # Phase 2: K-means clustering → initialize hexagram embeddings
    print(f"\nPhase 2: K-means clustering ({config.num_hexagrams} centers)...")
    centroids = kmeans_clustering(
        all_features, config.num_hexagrams, n_iter=50
    )

    # Initialize hexagram embeddings with cluster centroids
    with torch.no_grad():
        model.hexagram_embed.embedding.weight.copy_(centroids.to(device))
    print("  Initialized hexagram embeddings with cluster centroids.")

    # Phase 3: Contrastive fine-tuning (trains encoder + hexagram embeddings)
    print(f"\nPhase 3: Contrastive embedding fine-tuning ({epochs} epochs)...")
    # Train both the encoder AND hexagram embeddings so the encoder learns
    # to produce meaningful representations of BPE-tokenized input
    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) +
        list(model.hexagram_embed.parameters()), lr=lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                h = model.encode_context(input_ids, attention_mask)

            # Compute similarity with all hexagram embeddings
            hex_emb = model.hexagram_embed.embedding.weight  # (64, D)
            # Normalize
            h_norm = F.normalize(h, dim=-1)
            hex_norm = F.normalize(hex_emb, dim=-1)

            # Similarity matrix (B, 64)
            sim = h_norm @ hex_norm.T * 10.0  # scaled

            # Contrastive loss: pull closest centroid, push others
            # "Soft" nearest-neighbor assignment
            target = F.softmax(sim, dim=-1)
            log_probs = F.log_softmax(sim / 0.5, dim=-1)
            loss = -(target * log_probs).sum(dim=-1).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.hexagram_embed.parameters()) +
                list(model.encoder.parameters()), 1.0
            )
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"  Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}")

    # Save checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(os.path.join(checkpoint_dir, "yicenet_pretrained.pt"))
    print(f"\n  Pretrained model saved to {checkpoint_dir}/yicenet_pretrained.pt")


def kmeans_clustering(
    features: torch.Tensor,
    k: int,
    n_iter: int = 50,
    batch_size: int = 512,
) -> torch.Tensor:
    """
    Simple K-means clustering on features.

    Returns (k, D) centroid vectors.
    """
    n = features.shape[0]
    d = features.shape[1]

    # Initialize: k-means++
    centroids = []
    centroids.append(features[torch.randint(0, n, (1,))].squeeze(0))

    for _ in range(1, k):
        dists = torch.zeros(n)
        for c in centroids:
            dists += torch.sum((features - c.unsqueeze(0)) ** 2, dim=1)
        probs = dists / dists.sum()
        centroids.append(features[torch.multinomial(probs, 1)].squeeze(0))

    centroids = torch.stack(centroids)  # (k, D)

    # Iterate
    for iteration in range(n_iter):
        # Assign each point to nearest centroid
        dists = torch.cdist(features, centroids)  # (n, k)
        assignments = dists.argmin(dim=1)  # (n,)

        # Update centroids
        new_centroids = []
        for i in range(k):
            mask = assignments == i
            if mask.sum() > 0:
                new_centroids.append(features[mask].mean(dim=0))
            else:
                new_centroids.append(centroids[i])
        centroids = torch.stack(new_centroids)

        if (iteration + 1) % 10 == 0:
            inertia = dists.min(dim=1).values.sum().item()
            print(f"    K-means iteration {iteration+1}/{n_iter} — inertia: {inertia:.2f}")

    return centroids


def rl_train_stage(
    model: YiCeNet,
    config: YiCeNetConfig,
    env: Optional[object] = None,
    episodes: int = 5000,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints",
):
    """
    Stage 2: RL fine-tuning with REINFORCE.

    The "fortune teller" training loop:
    - Agent divines a hexagram for each state
    - Simulated environment provides reward
    - REINFORCE gradient updates the router + value network

    Philosophy: "用户持续交互即奖励" — user's continued engagement
    is the ultimate reward signal, token consumption is the cost.
    """
    print("\n" + "=" * 60)
    print("STAGE 2: RL Fine-tuning (REINFORCE)")
    print("=" * 60)

    # Use provided data-driven env or fall back to random simulation
    from yicenet.data.dataset import _random_env_fallback
    env = env or _random_env_fallback()
    if isinstance(env, DataDrivenEnv):
        print(f"  Using real session environment ({len(env.samples)} samples)")

    # Optimizer: only train router + value network + action decoder
    # Keep encoder and hexagram embeddings frozen during initial RL
    trainable_params = list(model.router.parameters()) + \
                       list(model.value_net.parameters()) + \
                       list(model.action_decoder.parameters()) + \
                       [model.trigram_prototypes] + \
                       list(model.trigram_cross_attn.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params, lr=lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=episodes // 100
    )

    # Training stats
    episode_rewards = []
    episode_lengths = []
    best_avg_reward = float("-inf")

    print(f"\nTraining for {episodes} episodes...")
    print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print(f"  Device: {device}\n")

    model.train()

    for episode in range(episodes):
        state = env.reset()
        done = False
        log_probs = []
        rewards = []
        values = []
        entropies = []
        step = 0

        while not done:
            step += 1

            # Convert state dict to input tensor
            state_tensor = torch.tensor(
                [[state["success_rate"], state["latency"],
                  state["complexity"], state["parallel_degree"] / 5.0,
                  state["session_depth"], state["user_engagement"],
                  state["token_cost"], 0.5]],
                dtype=torch.float32, device=device,
            )

            # Simulate input_ids for the encoder (use state proxy)
            # In production, this would be the actual user input
            seq_len = min(8, config.max_seq_len)
            input_ids = torch.randint(
                1, config.vocab_size, (1, seq_len), device=device
            )
            attention_mask = torch.ones(1, seq_len, device=device)

            # Forward pass
            output = model(input_ids, attention_mask, tau=model.tau, hard=False)

            # Safety: skip step if output contains NaN
            has_nan = any(torch.isnan(v).any() for v in output.values()
                         if isinstance(v, torch.Tensor))
            if has_nan:
                log_probs.append(torch.tensor(0.0, device=device))
                rewards.append(0.0)
                entropies.append(torch.tensor(0.0, device=device))
                # Step env anyway to advance the simulation
                next_state, _, done, _ = env.step(output["hexagram_idx"].item())
                state = next_state
                if step > 50:
                    done = True
                continue

            # Get log probability of chosen action
            action_probs = F.softmax(output["action_logits"], dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()  # (1,)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            # Step environment
            next_state, reward, done, terminal_type = env.step(
                output["hexagram_idx"].item()
            )

            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(output["candidate_values"].mean().unsqueeze(0))
            entropies.append(entropy.unsqueeze(0))

            state = next_state

            # Safety: limit episode length
            if step > 50:
                done = True

        # Compute REINFORCE with baseline
        episode_reward = sum(rewards)
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)

        # Compute advantages (with simple baseline = moving average)
        if len(episode_rewards) > 10:
            baseline = np.mean(episode_rewards[-10:])
        else:
            baseline = 0.0

        advantages = []
        R = 0
        for r in reversed(rewards):
            R = r + config.gamma * R
            advantages.insert(0, R - baseline)

        advantages = torch.tensor(advantages, device=device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs_tensor = torch.stack(log_probs).squeeze()
        entropies_tensor = torch.stack(entropies).squeeze()

        # Policy gradient loss
        policy_loss = -(log_probs_tensor * advantages.detach()).mean()
        entropy_loss = -config.entropy_coef * entropies_tensor.mean()
        loss = policy_loss + entropy_loss

        if loss.requires_grad:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, config.clip_grad_norm)
            optimizer.step()

        # Decay temperature
        model.decay_temperature()

        # Logging
        if (episode + 1) % 100 == 0:
            recent_rewards = episode_rewards[-100:]
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(episode_lengths[-100:])
            current_tau = model.get_temperature()

            print(
                f"  Episode {episode+1:>5}/{episodes} — "
                f"avg_reward: {avg_reward:>6.2f} | "
                f"avg_length: {avg_length:>4.1f} | "
                f"tau: {current_tau:.3f} | "
                f"loss: {loss.item():.4f}"
            )

            # Save best checkpoint
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(
                    os.path.join(checkpoint_dir, "yicenet_rl_best.pt")
                )
                print(f"    ✓ New best model saved (avg reward: {avg_reward:.2f})")

        scheduler.step()

    # Save final model
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(os.path.join(checkpoint_dir, "yicenet_rl_final.pt"))

    print(f"\n  Final model saved to {checkpoint_dir}/yicenet_rl_final.pt")
    print(f"  Best average reward: {best_avg_reward:.2f}")

    return episode_rewards


def main():
    parser = argparse.ArgumentParser(
        description="YiCeNet Training Pipeline"
    )
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=["pretrain", "rl", "all", "dry-run"],
        help="Training stage to run"
    )
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Synthetic samples for pretrain")
    parser.add_argument("--dataset", type=str, default="session",
                        choices=["synthetic", "session"],
                        help="Training data source: synthetic (25 hardcoded scenarios) or session (real Hermes logs)")
    parser.add_argument("--episodes", type=int, default=5000,
                        help="RL episodes")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pretrain_epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export_onnx", action="store_true",
                        help="Export model to ONNX after training")

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Config
    config = YiCeNetConfig()

    # Model
    model = YiCeNet(config).to(device)
    counts = count_parameters(model, verbose=True)

    # Create dataset & env based on --dataset flag
    session_dataset = None
    rl_env = None
    if args.dataset == "session":
        print("\nLoading session dataset...")
        session_dataset = SessionDataset(max_seq_len=config.max_seq_len)
        rl_env = DataDrivenEnv(session_dataset, seed=args.seed)
        print(f"  RL env: {len(session_dataset)} samples available")
    else:
        print("\nUsing synthetic dataset (25 hardcoded scenarios)")

    # Dry-run: just verify model works end-to-end
    if args.stage == "dry-run":
        print("\n" + "=" * 60)
        print("DRY RUN: Testing model forward pass")
        print("=" * 60)
        batch_size = 4
        seq_len = 16
        input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len)).to(device)
        attention_mask = torch.ones(batch_size, seq_len).to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask, tau=1.0, hard=False)

        print(f"  Input shape:             {input_ids.shape}")
        print(f"  h (state vector):        {output['h'].shape}")
        print(f"  Hexagram sampled:        {output['hexagram_idx'].tolist()}")
        print(f"  Best candidate idx:      {output['best_candidate_idx'].tolist()}")
        print(f"  Candidate hexagrams:     {output['candidate_idxs'].tolist()}")
        print(f"  Candidate values:        {output['candidate_values'].squeeze(-1).tolist()}")
        print(f"  Action IDs:              {output['action_ids'].tolist()}")

        # Test with structured reasoning
        print("\n  Candidate breakdown:")
        for b in range(min(batch_size, 2)):
            print(f"  ── Sample {b} ──")
            print(f"    本卦 (Original):    {output['candidate_idxs'][b,0].item():6d}  "
                  f"Q={output['candidate_values'][b,0].item():.3f}")
            print(f"    错卦 (Opposite):    {output['candidate_idxs'][b,1].item():6d}  "
                  f"Q={output['candidate_values'][b,1].item():.3f}")
            print(f"    综卦 (Upside-down): {output['candidate_idxs'][b,2].item():6d}  "
                  f"Q={output['candidate_values'][b,2].item():.3f}")
            print(f"    互卦 (Inner):       {output['candidate_idxs'][b,3].item():6d}  "
                  f"Q={output['candidate_values'][b,3].item():.3f}")
            print(f"    之卦 (Change):      {output['candidate_idxs'][b,4:].tolist()}  "
                  f"Q={output['candidate_values'][b,4:].squeeze().tolist()}")
            print(f"    → Selected candidate {output['best_candidate_idx'][b].item()}: "
                  f"hexagram {output['candidate_idxs'][b, output['best_candidate_idx'][b]].item()}")
            print(f"    → Action: {output['action_ids'][b].item()}")

        print("\n  ✓ Forward pass successful!")
        return

    # Stage 1: Pre-train
    if args.stage in ("pretrain", "all"):
        pretrain_stage(
            model=model,
            config=config,
            dataset=session_dataset,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            epochs=args.pretrain_epochs,
            lr=args.lr,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
        )

    # Stage 2: RL
    if args.stage in ("rl", "all"):
        # Load pretrained if available
        pretrained_path = os.path.join(
            args.checkpoint_dir, "yicenet_pretrained.pt"
        )
        if os.path.exists(pretrained_path):
            saved = torch.load(pretrained_path, map_location=device, weights_only=False)
            model.load_state_dict(saved["model_state_dict"], strict=False)
            missing = set(model.state_dict().keys()) - set(saved["model_state_dict"].keys())
            if missing:
                print(f"  New params (random init): {', '.join(sorted(missing))}")
            if "tau" in saved:
                model.tau = saved["tau"]
            print(f"\nLoaded pretrained weights from {pretrained_path}")
        elif args.stage == "rl":
            print(f"\nNo pretrained checkpoint found at {pretrained_path}.")
            print("Starting RL from scratch (random init).")

        rl_train_stage(
            model=model,
            config=config,
            env=rl_env,
            episodes=args.episodes,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
        )

    # Export to ONNX
    if args.export_onnx:
        print("\nExporting to ONNX...")
        batch_size = 1
        seq_len = config.max_seq_len
        dummy_input = torch.randint(
            1, config.vocab_size, (batch_size, seq_len)
        ).to(device)
        dummy_mask = torch.ones(batch_size, seq_len).to(device)

        torch.onnx.export(
            model,
            (dummy_input, dummy_mask),
            os.path.join(args.checkpoint_dir, "yicenet.onnx"),
            input_names=["input_ids", "attention_mask"],
            output_names=[
                "hexagram_idx", "best_candidate_idx",
                "candidate_values", "action_ids"
            ],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
            },
            opset_version=17,
        )
        print(f"  ONNX model saved to {args.checkpoint_dir}/yicenet.onnx")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
