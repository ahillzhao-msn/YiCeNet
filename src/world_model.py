"""
World Model for YiCeNet — learns to predict reward from (task_embedding, hexagram).

Input:  encoder hidden h (256-dim) + candidate hexagram ID (0-63, one-hot 64-dim)
Output: predicted scalar reward

Architecture:
  [h; onehot(hex)] → LayerNorm → Linear(320→128) → GELU → Linear(128→64) → GELU → Linear(64→1)
  ~18K parameters, <1 MB, <0.1 ms inference

Trains on real session data: for each user message, generates 8 training examples
(one per candidate hexagram), all with the same target = actual session reward.
This lets the model learn: "for this type of task, hexagram X tends to yield reward Y."
"""

import os, sys, json, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class WorldModel(nn.Module):
    """
    Lightweight reward predictor.

    Encodes: "given this task context (h), how good is hexagram x?"
    """

    def __init__(self, hidden_dim: int = 256, num_hexagrams: int = 64):
        super().__init__()
        self.input_dim = hidden_dim + num_hexagrams  # 320
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.orthogonal_(p, gain=0.5)

    def forward(self, h: torch.Tensor, hex_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, D) task embeddings from YiCeNet encoder
            hex_id: (B,) or (B, K) hexagram indices

        Returns:
            reward: (B,) or (B, K) predicted rewards
        """
        if hex_id.dim() == 2:
            # (B, K) → unsqueeze and expand
            B, K = hex_id.shape
            h_exp = h.unsqueeze(1).expand(-1, K, -1)  # (B, K, D)
            onehot = F.one_hot(hex_id, num_classes=64).float()  # (B, K, 64)
            x = torch.cat([h_exp, onehot], dim=-1)  # (B, K, 320)
            return self.net(x).squeeze(-1)  # (B, K)
        else:
            # (B,)
            onehot = F.one_hot(hex_id, num_classes=64).float()  # (B, 64)
            x = torch.cat([h, onehot], dim=-1)  # (B, 320)
            return self.net(x).squeeze(-1)  # (B,)

    def predict_reward(self, h: torch.Tensor, hex_id: int) -> float:
        """Single prediction for inference."""
        self.eval()
        with torch.no_grad():
            h_t = h.unsqueeze(0) if h.dim() == 1 else h
            hex_t = torch.tensor([hex_id], device=h_t.device)
            return self.forward(h_t, hex_t).item()

    def evaluate_candidates(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Score all 64 hexagrams for a given task embedding.
        Returns:
            best_id: (B,) best hexagram index
            all_values: (B, 64) scores for all hexagrams
        """
        self.eval()
        with torch.no_grad():
            B = h.shape[0]
            all_hex = torch.arange(64, device=h.device).unsqueeze(0).expand(B, -1)  # (B, 64)
            scores = self.forward(h, all_hex)  # (B, 64)
            best_id = scores.argmax(dim=-1)  # (B,)
        return best_id, scores

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "WorldModel":
        m = cls()
        saved = torch.load(path, map_location=device, weights_only=True)
        m.load_state_dict(saved["state_dict"])
        return m.to(device)


class WorldModelTrainingData(Dataset):
    """
    Generates training pairs: (h, hex_id, target_reward).

    For each of 663 session samples, generates 8 examples:
    one for each candidate hexagram position.

    Total: 663 × 8 = 5,304 training pairs.
    """

    def __init__(self, session_dataset, model, device=DEVICE):
        self.samples = []
        model.eval()
        ds = session_dataset

        for i in range(len(ds)):
            sample = ds[i]
            ids = sample["input_ids"].unsqueeze(0).to(device)
            mask = sample["attention_mask"].unsqueeze(0).to(device)
            reward = sample["reward"].item()

            with torch.no_grad():
                h = model.encode_context(ids, mask)  # (1, 256)

            # Generate candidates from the model's perspective
            _, cand_idxs, _ = model.evaluate_candidates(
                torch.zeros(1, dtype=torch.long, device=device), h
            )
            # cand_idxs: (1, 8) — 8 candidate hexagrams

            for k in range(8):
                hex_id = cand_idxs[0, k].item()
                self.samples.append({
                    "h": h.squeeze(0).cpu(),  # (256,)
                    "hex_id": hex_id,
                    "reward": reward,
                })

        print(f"[WorldModelData] Generated {len(self.samples)} training pairs "
              f"from {len(ds)} session samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "h": s["h"],
            "hex_id": torch.tensor(s["hex_id"], dtype=torch.long),
            "reward": torch.tensor(s["reward"], dtype=torch.float32),
        }


def train_world_model(
    model_path: str = "checkpoints/world_model.pt",
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 3e-4,
) -> WorldModel:
    """Train world model from session data."""
    print("=" * 60)
    print("Training World Model")
    print("=" * 60)

    from src.model import YiCeNet
    from src.config import YiCeNetConfig
    from src.data.dataset import SessionDataset

    # Load YiCeNet encoder (frozen)
    config = YiCeNetConfig()
    yicenet = YiCeNet(config).to(DEVICE)
    ckpt_path = "checkpoints/yicenet_v3.pt"
    if os.path.exists(ckpt_path):
        saved = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        yicenet.load_state_dict(saved["model_state_dict"], strict=False)
        print(f"  Loaded YiCeNet encoder from {ckpt_path}")
    yicenet.eval()
    for p in yicenet.parameters():
        p.requires_grad = False

    # Load session dataset
    ds = SessionDataset(max_seq_len=config.max_seq_len)
    train_data = WorldModelTrainingData(ds, yicenet, DEVICE)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

    # World model
    wm = WorldModel().to(DEVICE)
    optimizer = torch.optim.AdamW(wm.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n  Training for {epochs} epochs ({len(train_data)} pairs)...")
    wm.train()

    best_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            h = batch["h"].to(DEVICE)
            hex_id = batch["hex_id"].to(DEVICE)
            target = batch["reward"].to(DEVICE)

            pred = wm.forward(h, hex_id)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:>3}/{epochs} — MSE: {avg_loss:.4f}  "
                  f"RMSE: {np.sqrt(avg_loss):.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            wm.save(model_path.replace(".pt", "_best.pt"))

    wm.save(model_path)
    print(f"\n  ✓ World model saved to {model_path}")
    print(f"  Best MSE: {best_loss:.4f}, Best RMSE: {np.sqrt(best_loss):.4f}")

    # Quick sanity check
    wm.eval()
    with torch.no_grad():
        sample = ds[0]
        ids = sample["input_ids"].unsqueeze(0).to(DEVICE)
        mask = sample["attention_mask"].unsqueeze(0).to(DEVICE)
        h = yicenet.encode_context(ids, mask)
        best_id, scores = wm.evaluate_candidates(h)
        true_reward = sample["reward"].item()
        predicted = scores[0, best_id[0]].item()
        print(f"\n  Sanity check: best hex={best_id[0].item()}, "
              f"predicted Q={predicted:.3f}, true reward={true_reward:.3f}")
        print(f"  Score range: [{scores.min().item():.3f}, {scores.max().item():.3f}]")
        print(f"  Score spread: {(scores.max() - scores.min()).item():.3f}")

    return wm


if __name__ == "__main__":
    wm = train_world_model()
