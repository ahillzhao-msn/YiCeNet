"""
Train YiCeNet value network with supervised regression on real session rewards.

After this, the value network can score candidate hexagrams based on
expected session outcome, and the router can learn to pick high-value hexagrams.

Usage: python src/train_value_net.py [--epochs 50]
"""

import os, sys, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from yicenet.model import YiCeNet
from yicenet.config import YiCeNetConfig
from yicenet.data.dataset import SessionDataset
from yicenet.tokenizer import encode

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 50
LR = 3e-4
CKPT_PATH = "checkpoints/yicenet_v3.pt"
OUTPUT_PATH = "checkpoints/yicenet_v3_trained.pt"

def train_value_net():
    print("=" * 60)
    print("YiCeNet Value Network Training (Supervised Regression)")
    print("=" * 60)

    # Load model
    config = YiCeNetConfig()
    model = YiCeNet(config).to(DEVICE)
    saved = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(saved["model_state_dict"], strict=False)
    print(f"  Loaded v3 from {CKPT_PATH}")

    # Freeze encoder + hexagram embed — only train value_net + router + decoder
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.hexagram_embed.parameters():
        p.requires_grad = False
    model.trigram_prototypes.requires_grad = False
    for p in model.trigram_cross_attn.parameters():
        p.requires_grad = False

    trainable = list(model.value_net.parameters()) + \
                list(model.router.parameters()) + \
                list(model.action_decoder.parameters())
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-5)

    # Load session dataset
    ds = SessionDataset(max_seq_len=config.max_seq_len)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print(f"\n  Training for {EPOCHS} epochs on {len(ds)} samples...")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_mse = 0.0
        n_batches = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            target_reward = batch["reward"].to(DEVICE)  # (B,)

            # Encoder forward (frozen)
            with torch.no_grad():
                h = model.encode_context(input_ids, attention_mask)  # (B, 256)

            # Gumbel router: sample hexagram
            hex_idx, _ = model.router(h, tau=0.5, hard=True)  # (B,)

            # Get hexagram embedding
            hex_emb = model.hexagram_embed(hex_idx)  # (B, 256)

            # Value network: predict Q-value for this hexagram
            q_pred = model.value_net(hex_emb.unsqueeze(1))  # (B, 1, 1)
            q_pred = q_pred.squeeze(-1).squeeze(-1)  # (B,)

            # MSE loss between predicted Q and actual reward
            mse_loss = F.mse_loss(q_pred, target_reward)

            # Router should also learn: pick hexagrams that get high value
            # Use the predicted Q as advantage signal for the Gumbel-Softmax
            # But for now, just train the value net

            optimizer.zero_grad()
            mse_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            total_loss += mse_loss.item()
            total_mse += mse_loss.item()
            n_batches += 1

        avg_mse = total_mse / max(n_batches, 1)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:>3}/{EPOCHS} — MSE: {avg_mse:.4f}  RMSE: {np.sqrt(avg_mse):.4f}")

    # Save trained model
    torch.save({
        "model_state_dict": model.state_dict(),
        "tau": model.tau,
        "config": model.config,
    }, OUTPUT_PATH)
    print(f"\n  ✓ Saved to {OUTPUT_PATH}")

    # Quick validation: check value range
    model.eval()
    with torch.no_grad():
        sample = ds[0]
        ids = sample["input_ids"].unsqueeze(0).to(DEVICE)
        mask = sample["attention_mask"].unsqueeze(0).to(DEVICE)
        h = model.encode_context(ids, mask)
        hex_idx, _ = model.router(h, tau=0.1, hard=True)
        hex_emb = model.hexagram_embed(hex_idx)
        q = model.value_net(hex_emb.unsqueeze(1))
        true_r = sample["reward"]
        print(f"  Sample 0: predicted Q={q.item():.4f}, actual reward={true_r:.4f}")

    print("\n  Done!")


if __name__ == "__main__":
    train_value_net()
