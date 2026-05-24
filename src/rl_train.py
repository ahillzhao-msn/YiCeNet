"""
RL fine-tuning for YiCeNet v5 — 64-dim projection reward + probe-based training.

Key changes from v4:
  - Reward = 1 - CDist(WM_predicted_dist, projected_actual_dist) ∈ [0, 1]
  - Value Network learns to match WM's 64-dim prediction, not scalar reward
  - Training uses probe vectors (ℝ⁹) as WM input
  - Power-law weighting for sample importance
"""

import os, sys, json, time, math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── 64卦投影簇 ──
# 將 64 卦按功能性粗分為三簇（用於 projection target 生成）
CLUSTER_CONTINUE = [0, 4, 10, 14, 28, 34, 46, 57]      # 乾·需·泰·大有·坎·晉·升·兌
CLUSTER_CORRECTION = [5, 42, 48, 49, 55]                # 訟·睽·革·鼎·旅
CLUSTER_COMPLETION = [24, 25, 34, 47, 62, 63]            # 復·无妄·晉·困·既濟·未濟
CLUSTER_ABANDON = [2, 3, 38, 38, 51, 52]                # 屯·蒙·蹇·解·震·艮
CLUSTER_PRAISE = [10, 12, 16, 30, 44, 57]               # 泰·否·豫·離·姤·兌


def project_to_hexagram_space(
    reward_signals: dict,
    num_hexagrams: int = 64,
    temperature: float = 0.5,
    continuation_w: float = 0.4,
    correction_w: float = 0.4,
    completion_w: float = 0.4,
    satisfaction: Optional[float] = None,
) -> torch.Tensor:
    """
    將外部獎勵信號投影到 64 卦空間。

    Args:
        reward_signals: {
            "continued": bool,
            "corrected": bool,
            "completed": bool,
            "praised": bool,
        }
        num_hexagrams: 64
        temperature: softmax 溫度（被 satisfaction 覆蓋時忽略）
        continuation_w: 續航投影權重
        correction_w: 校正投影權重
        completion_w: 完成投影權重
        satisfaction: DS 滿意度 [-1,1]。越高→target越尖銳(高置信)，
                      越低→target越平坦(低置信≈噪聲忽略)。
                      為 None 時使用固定 temperature。

    Returns:
        target: (64,) 結果卦象分布
    """
    target = torch.zeros(num_hexagrams)

    if reward_signals.get("continued", False):
        for h in CLUSTER_CONTINUE:
            target[h] += continuation_w

    if reward_signals.get("praised", False):
        for h in CLUSTER_PRAISE:
            target[h] += completion_w * 1.5  # 讚賞比完成更強

    if reward_signals.get("completed", False):
        for h in CLUSTER_COMPLETION:
            target[h] += completion_w

    if reward_signals.get("corrected", False):
        for h in CLUSTER_CORRECTION:
            target[h] += correction_w

    if reward_signals.get("abandoned", False):
        for h in CLUSTER_ABANDON:
            target[h] += correction_w  # 放棄=負面信號也投影

    # 確保至少有一些質量
    if target.sum() == 0:
        target = torch.ones(num_hexagrams) / num_hexagrams  # 均勻分布
    else:
        # DS 置信度自適應溫度：|satisfaction|越低→溫度越高→target越平坦
        # DS 說「沒把握」(satisfaction≈0) → temperature≈1 → target≈均勻→loss≈0
        # DS 說「有把握」(|satisfaction|>0.5) → temperature<0.5 → target尖銳→強信號
        if satisfaction is not None:
            effective_temp = max(0.1, 1.0 - abs(satisfaction))
        else:
            effective_temp = temperature
        target = F.softmax(target / effective_temp, dim=-1)

    return target


def compute_hexagram_reward(
    wm_predicted_dist: torch.Tensor,
    actual_dist: torch.Tensor,
) -> torch.Tensor:
    """
    計算 64 卦空間中的獎勵。

    reward = 1 - Euclidean distance between predicted and actual distributions
    結果在 [0, 1] 之間，1 = 完美預測。

    Args:
        wm_predicted_dist: (B, 64) WM 預測結果分布
        actual_dist: (B, 64) 實際結果分布

    Returns:
        reward: (B,) 獎勵標量
    """
    # Cosine similarity (更適合分布比較)
    cos_sim = F.cosine_similarity(wm_predicted_dist, actual_dist, dim=-1)
    reward = (cos_sim + 1.0) / 2.0  # 映射到 [0, 1]
    return reward


def rl_train_v5(
    checkpoint_in: str = "checkpoints/yicenet_v4.pt",
    checkpoint_out: str = "checkpoints/yicenet_v5.pt",
    world_model_path: str = "checkpoints/world_model_best.pt",
    episodes: int = 500,
    lr: float = 3e-4,
    probe_weight: float = 0.1,
):
    """
    YiCeNet v5 RL 訓練。

    訓練流程：
      1. 載入 v4 基底模型（兼容）
      2. 載入世界模型 v2（雙頭）
      3. 對每條 session data：
         a. 前向傳播 → 獲取六探針
         b. WM 頭A預測結果卦象分布
         c. 從 follow-up 提取信號 → project_to_hexagram_space()
         d. compute_hexagram_reward() = 預測 vs 實際的分布距離
         e. 冪律加權梯度
      4. Value Network 學習 WM 頭A的預測
      5. Router 通過對比分布間距優化
    """
    print("=" * 60)
    print("YiCeNet v5 RL Fine-tuning (64-dim Projection Reward)")
    print("=" * 60)

    from src.model import YiCeNet
    from src.config import YiCeNetConfig
    from src.data.dataset import SessionDataset
    from src.world_model import WorldModelV2, power_law_weight_batch
    from src.tokenizer import encode as yicenet_encode
    from src.external_metrics import compute_satisfaction

    # ── Load YiCeNet v4 ──
    config = YiCeNetConfig()
    model = YiCeNet(config).to(DEVICE)
    saved = torch.load(checkpoint_in, map_location=DEVICE, weights_only=False)
    model.load_state_dict(saved["model_state_dict"], strict=False)
    if "tau" in saved:
        model.tau = saved["tau"]
    print(f"  Loaded base model from {checkpoint_in}")

    # ── Load World Model v2 ──
    wm = WorldModelV2.load(world_model_path, DEVICE)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False
    print(f"  Loaded World Model v2 from {world_model_path}")

    # ── Load session data ──
    ds = SessionDataset(max_seq_len=config.max_seq_len)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    print(f"  {len(ds)} session samples available for RL")

    # ── Trainable params ──
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

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, episodes // 50)
    )

    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")
    print(f"\n  Training for {episodes} episodes...\n")

    model.train()
    episode_rewards = []
    best_avg_reward = float("-inf")

    # Pre-compute all task embeddings + probes
    print("  Pre-computing task embeddings + probes...")
    all_probes = []
    all_hex_ids = []
    all_texts = []
    all_timestamps = []
    all_target_dists = []

    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            ids = sample["input_ids"].unsqueeze(0).to(DEVICE)
            mask = sample["attention_mask"].unsqueeze(0).to(DEVICE)
            out = model(ids, mask, tau=0.01, hard=True)

            probes_t = out["probes"].to(DEVICE)  # (9,)
            all_probes.append(probes_t)
            all_hex_ids.append(out["hexagram_idx"])
            all_texts.append(sample.get("text", ""))

            # Extract reward signals from sample
            reward_sig = {
                "continued": sample.get("continued", True),
                "corrected": sample.get("corrected", False),
                "completed": sample.get("completed", False),
                "praised": sample.get("praised", False),
                "abandoned": sample.get("abandoned", False),
            }
            target_dist = project_to_hexagram_space(
                reward_sig,
                temperature=config.ext_projection_temperature,
                continuation_w=config.ext_continuation_weight,
                correction_w=config.ext_correction_weight,
                completion_w=config.ext_completion_weight,
            )
            all_target_dists.append(target_dist.to(DEVICE))
            all_timestamps.append(sample.get("timestamp", time.time()))

    all_probes = torch.stack(all_probes)  # (N, 9)
    all_hex_ids = torch.stack(all_hex_ids).squeeze(-1)  # (N,)
    all_target_dists = torch.stack(all_target_dists)  # (N, 64)
    all_timestamps = torch.tensor(all_timestamps, device=DEVICE)
    print(f"  Pre-computed {len(all_probes)} samples")

    now = time.time()

    for episode in range(episodes):
        # Sample a random training case
        idx = np.random.randint(0, len(all_probes))
        probes = all_probes[idx].unsqueeze(0).to(DEVICE)  # (1, 9)
        hex_id = all_hex_ids[idx].unsqueeze(0).to(DEVICE)  # (1,)
        target_dist = all_target_dists[idx].unsqueeze(0).to(DEVICE)  # (1, 64)
        ts = all_timestamps[idx]

        # ── Forward through model ──
        # Router picks hexagram
        h = model.encoder(
            ds[idx]["input_ids"].unsqueeze(0).to(DEVICE),
            ds[idx]["attention_mask"].unsqueeze(0).to(DEVICE),
        )
        hex_idx, hex_probs = model.router(h, tau=max(model.tau, 0.05), hard=False)

        # Get candidates
        _, cand_idxs, cand_values = model.evaluate_candidates(hex_idx, h)

        # ── WM prediction (frozen) ──
        with torch.no_grad():
            wm_pred_dist, wm_ext_vec = wm(probes, hex_id)  # (1, 64), (1, N)

        # ── Reward = distribution similarity ──
        reward = compute_hexagram_reward(wm_pred_dist, target_dist)  # (1,)

        # ── Value loss: value_net should produce similar distribution to WM ──
        # First get value net scores for all candidates
        cand_scores = wm_pred_dist[0, cand_idxs.squeeze(0)]  # (8,)
        value_loss = F.mse_loss(cand_values.squeeze(-1), cand_scores.unsqueeze(0))

        # ── Policy gradient ──
        hex_probs_sq = hex_probs.squeeze(0)  # (64,)
        avg_score = wm_pred_dist.mean()
        advantages = wm_pred_dist - avg_score  # (1, 64)
        advantages = advantages.squeeze(0)  # (64,)

        log_probs = torch.log(hex_probs_sq.clamp(min=1e-8))
        policy_loss = -(log_probs * advantages * 0.1).sum()

        # Entropy bonus
        entropy = -(hex_probs_sq * log_probs).sum()
        entropy_loss = -0.01 * entropy

        # Total loss
        loss = policy_loss + value_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        episode_rewards.append(reward.item())
        model.decay_temperature()

        # Logging
        if (episode + 1) % 100 == 0:
            recent = episode_rewards[-100:]
            avg_r = np.mean(recent)
            tau = model.get_temperature()

            # Evaluate: correlation between model's top choice and WM top
            model.eval()
            with torch.no_grad():
                # Model's top hexagram from value net
                best_idx = cand_values.squeeze(-1).argmax(dim=-1)
                model_hex = cand_idxs[0, best_idx[0]].item()
                # WM's top prediction
                wm_top = wm_pred_dist[0].argmax().item()
                match = "✓" if model_hex == wm_top else "✗"
                # Overall reward trend
                rwd_trend = sum(episode_rewards[-100:]) / 100
            model.train()

            print(f"  Episode {episode+1:>4}/{episodes} — "
                  f"avg_reward: {avg_r:.3f} | "
                  f"rwd_trend: {rwd_trend:.3f} | "
                  f"tau: {tau:.3f} | "
                  f"model_hex: {model_hex:2d} | "
                  f"wm_top: {wm_top:2d} {match} | "
                  f"policy: {policy_loss.item():.4f} | "
                  f"value: {value_loss.item():.4f}")

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
    rl_train_v5(episodes=500)
