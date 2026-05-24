"""
Session dataset for YiCeNet — extracts real conversation data from Hermes session DB.

Each sample = one user message → corresponding assistant response.
Features extracted for world model:
  - text: user message (input to YiCeNet encoder)
  - token_cost: session-level avg tokens per message
  - success: 1 if conversation continued (user sent another message after assistant reply)
  - abandoned: 1 if user stopped talking after this exchange
  - corrected: 1 if user sent a correction/rebuke within next 2 messages
  - completion: 1 if the session naturally ended with a task done signal
"""

import json
import os
import sqlite3
import random
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from yicenet.tokenizer import encode

_YICENET_ROOT = Path(__file__).parent.parent
DB_PATH = str(Path.home() / ".hermes" / "state.db")

# ── Correction keywords (user signal detection) ──
# Full-word/substring patterns. Avoid single Chinese chars that
# appear in normal text ("不", "错" are too broad).
CORRECTION_PATTERNS = [
    # Chinese — two+ character patterns only
    "不对", "不是的", "错了", "不对的",
    "重新来", "再来", "搞错了",
    "你理解错了", "你错了", "我说的是",
    "不是这样", "不是那个",
    # English
    " that's not", "that is not", "that isn't",
    "not what i", "wrong.", "incorrect",
    "stop", "no!", "nope",
    "i didn't mean", "i meant ",
    # Mixed signals
    "听错了",
]

# Short messages (≤15 chars) that are likely corrections
SHORT_CORRECTION_TRIGGERS = {
    "no", "不", "不对", "不是", "错了", "stop", "no!", "wrong",
    "重新", "再来", "重来", "不是的",
}


def _detect_correction(text: str) -> bool:
    """Heuristic: check if user message is a correction."""
    text_lower = text.lower().strip()

    # Super-short messages (1-3 chars) — likely corrections
    if len(text_lower) <= 3:
        return text_lower in {"no", "不", "不对", "不是", "错了", "stop"}

    # Pattern matches
    for pat in CORRECTION_PATTERNS:
        if pat in text_lower:
            return True

    return False


def _detect_completion(text: str) -> bool:
    """Heuristic: check if session end sounds like completion."""
    text_lower = text.lower().strip()
    endings = [
        "完毕", "完成", "好", "好的", "谢谢", "thanks",
        "done", "complete", "finished", "that's all",
        "就这样", "没有别的了", "ok", "okay", "行了",
    ]
    return any(text_lower.startswith(e) or text_lower == e for e in endings)


class SessionDataset(Dataset):
    """
    Extract labeled (user_message, features) pairs from Hermes session DB.

    Each item:
      - text: user message string
      - input_ids: YiCeNet token IDs (Qwen BPE → rebucketed)
      - attention_mask: padding mask
      - features: 8-dim vector [success_rate, latency, complexity, ...]
      - reward: calibrated reward = success_bonus - token_penalty - correction_penalty
      - terminal_type: 'active', 'success', 'abandoned', 'corrected'

    Length: number of user messages (excluding the last one of each session,
    which has no follow-up for reward calculation)
    """

    def __init__(
        self,
        db_path: str = DB_PATH,
        max_seq_len: int = 128,
        min_session_length: int = 2,
        include_corrections: bool = True,
        reward_coefficients: Optional[dict] = None,
    ):
        """
        Args:
            db_path: path to Hermes state.db
            max_seq_len: max tokenization length
            min_session_length: skip sessions with fewer user messages than this
            include_corrections: if True, include samples where user corrects
            reward_coefficients: override default reward weights
        """
        self.max_seq_len = max_seq_len
        self.reward_coef = reward_coefficients or {
            "success_bonus": 0.8,          # +0.8 when user continues
            "correction_penalty": -1.0,    # -1.0 when user corrects
            "abandon_penalty": -2.0,        # -2.0 when user abandons
            "completion_bonus": 0.5,        # +0.5 when task completes
            "token_cost_weight": -0.0003,    # ~0.5 avg penalty at 1815 tok/sample
        }

        # Extract from DB
        self.samples = self._extract(
            db_path, min_session_length, include_corrections
        )
        print(f"[SessionDataset] Loaded {len(self.samples)} samples from {db_path}")

    def _extract(
        self, db_path: str, min_len: int, include_corrections: bool
    ) -> list[dict]:
        """Extract labeled sample pairs from session DB."""
        conn = sqlite3.connect(db_path)
        samples = []

        # Get all session IDs with user+assistant pairs
        sessions = conn.execute("""
            SELECT id, input_tokens, output_tokens, message_count,
                   title
            FROM sessions
            WHERE input_tokens IS NOT NULL
            ORDER BY started_at
        """).fetchall()

        for session_id, inp_tok, out_tok, msg_count, title in sessions:
            # Get all messages in order
            msgs = conn.execute("""
                SELECT id, role, content, timestamp
                FROM messages
                WHERE session_id = ? AND role IN ('user', 'assistant')
                ORDER BY timestamp
            """, (session_id,)).fetchall()

            if len(msgs) < min_len * 2:  # at least min_len user messages
                continue

            # Calculate per-message token cost
            total_tokens = (inp_tok or 0) + (out_tok or 0)
            per_msg_cost = total_tokens / max(len(msgs), 1)

            # Walk through user→assistant pairs
            i = 0
            while i < len(msgs) - 1:
                if msgs[i][1] != "user":
                    i += 1
                    continue
                if msgs[i + 1][1] != "assistant":
                    i += 1
                    continue

                user_msg = msgs[i]
                asst_msg = msgs[i + 1]
                user_text = user_msg[2] or ""
                asst_text = asst_msg[2] or ""

                if len(user_text.strip()) < 3:
                    i += 1
                    continue

                # Skip system-generated user messages (context compaction, etc.)
                if user_text.startswith(("[CONTEXT", "[SYSTEM", "SYSTEM:", "[Note:")):
                    i += 1
                    continue

                # Determine what happens next
                # Look at the NEXT user message (if any) for reward signal
                next_user_idx = None
                for j in range(i + 2, len(msgs)):
                    if msgs[j][1] == "user":
                        next_user_idx = j
                        break

                terminal_type = "active"
                reward = 0.0
                corrected = False
                abandoned = False
                completed = False

                if next_user_idx is not None:
                    next_text = msgs[next_user_idx][2] or ""
                    if _detect_correction(next_text):
                        corrected = True
                        terminal_type = "corrected"
                    elif _detect_completion(next_text):
                        completed = True
                        terminal_type = "success"
                    else:
                        # User continues normally
                        terminal_type = "active"
                else:
                    # No follow-up — abandoned or completed
                    if user_text and _detect_completion(user_text):
                        completed = True
                        terminal_type = "success"
                    else:
                        abandoned = True
                        terminal_type = "abandoned"

                # Compute reward
                token_penalty = self.reward_coef["token_cost_weight"] * per_msg_cost
                reward = (
                    (self.reward_coef["success_bonus"] if terminal_type == "active" else 0.0)
                    + (self.reward_coef["correction_penalty"] if corrected else 0.0)
                    + (self.reward_coef["abandon_penalty"] if abandoned else 0.0)
                    + (self.reward_coef["completion_bonus"] if completed else 0.0)
                    + token_penalty
                )

                # Build feature vector (similar to original 8-dim but data-driven)
                complexity = min(1.0, len(user_text) / 500.0)  # text length as proxy
                success_rate = 0.0 if corrected or abandoned else 0.8
                if completed:
                    success_rate = 0.95
                elif terminal_type == "active":
                    success_rate = 0.7 + 0.3 * random.random()  # slight variance

                features = torch.tensor([
                    success_rate,                                          # 0: success_rate
                    min(1.0, per_msg_cost / 5000.0),                       # 1: latency proxy
                    complexity,                                            # 2: complexity
                    min(1.0, len(msgs) / 100.0),                           # 3: parallel_degree
                    min(1.0, (i // 2) / 50.0),                             # 4: sequential_depth
                    0.0 if corrected else (0.3 + 0.3 * random.random()),   # 5: cache_hit
                    0.2 if corrected or abandoned else 0.05,               # 6: error_rate
                    0.5 + 0.3 * random.random(),                           # 7: diversity
                ], dtype=torch.float32)

                samples.append({
                    "text": user_text,
                    "user_msg_id": user_msg[0],
                    "session_id": session_id,
                    "features": features,
                    "reward": reward,
                    "terminal_type": terminal_type,
                    "corrected": corrected,
                    "abandoned": abandoned,
                    "completed": completed,
                    "token_cost": per_msg_cost,
                })

                i += 1  # Skip the assistant message we just consumed
                if not include_corrections and corrected:
                    # Remove the last added sample
                    samples.pop()
                    # But we already appended... skip in next iteration

        conn.close()
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Tokenize with Qwen BPE → YiCeNet rebucket
        input_ids, attention_mask = encode(
            sample["text"], max_len=self.max_seq_len
        )

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "features": sample["features"],
            "reward": torch.tensor(sample["reward"], dtype=torch.float32),
            "terminal_type": sample["terminal_type"],
            "text": sample["text"],
            "token_cost": sample["token_cost"],
        }

    def get_session_stats(self) -> dict:
        """Return summary statistics across all samples."""
        rewards = [s["reward"] for s in self.samples]
        types = [s["terminal_type"] for s in self.samples]
        costs = [s["token_cost"] for s in self.samples]

        return {
            "total_samples": len(self.samples),
            "avg_reward": sum(rewards) / max(len(rewards), 1),
            "active_pct": types.count("active") / max(len(types), 1) * 100,
            "corrected_pct": types.count("corrected") / max(len(types), 1) * 100,
            "abandoned_pct": types.count("abandoned") / max(len(types), 1) * 100,
            "success_pct": types.count("success") / max(len(types), 1) * 100,
            "avg_token_cost": sum(costs) / max(len(costs), 1),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
        }


class DataDrivenEnv:
    """
    Replacement for RLSimulationEnv.

    Serves pre-computed (state, reward) transitions from real session data.
    The agent samples from real conversation traces instead of random noise.

    This is a "world model" of the simplest form: empirical replay.
    """

    def __init__(
        self,
        dataset: SessionDataset,
        seed: int = 42,
    ):
        self.rng = random.Random(seed)
        self.samples = dataset.samples
        self.rng.shuffle(self.samples)
        self._position = 0
        self.current_sample = None
        self.step_count = 0
        self.max_steps = 10

    def reset(self) -> dict:
        """Reset and return initial state from a random sample."""
        self._position = (self._position + 1) % len(self.samples)
        self.current_sample = self.samples[self._position]
        self.step_count = 0
        self.max_steps = self.rng.randint(3, 8)

        state = {
            "success_rate": self.current_sample["features"][0].item(),
            "latency": self.current_sample["features"][1].item(),
            "complexity": self.current_sample["features"][2].item(),
            "parallel_degree": int(self.current_sample["features"][3].item() * 4 + 1),
            "session_depth": 0.0,
            "user_engagement": 1.0,
            "token_cost": self.current_sample["token_cost"],
            "text": self.current_sample["text"],
        }
        return state

    def step(self, hexagram_id: int) -> tuple[dict, float, bool, str]:
        """
        Advance one step.

        Returns real reward from session data + next state.
        """
        self.step_count += 1
        sample = self.current_sample
        reward = sample["reward"]
        terminal_type = sample["terminal_type"]

        # Session end conditions
        done = (
            self.step_count >= self.max_steps
            or terminal_type == "success"
            or terminal_type == "abandoned"
        )

        # Next state (advance to next sample in same session, or reset)
        next_idx = (self._position + 1) % len(self.samples)
        next_sample = self.samples[next_idx]
        next_state = {
            "success_rate": next_sample["features"][0].item(),
            "latency": next_sample["features"][1].item(),
            "complexity": next_sample["features"][2].item(),
            "parallel_degree": int(next_sample["features"][3].item() * 4 + 1),
            "session_depth": self.step_count / max(self.max_steps, 1),
            "user_engagement": max(0.0, 1.0 - 0.1 * self.step_count),
            "token_cost": next_sample["token_cost"],
            "text": next_sample["text"],
        }

        self.current_sample = next_sample
        self._position = next_idx

        return next_state, reward, done, terminal_type


# ── Backward-compatible fallback for --dataset synthetic ──

class _RandomEnv:
    """Minimal random env for backward compat with --dataset synthetic."""
    def __init__(self, seed=42):
        import random
        self.rng = random.Random(seed)
        self.step_count = 0
    def reset(self):
        self.step_count = 0
        return {"success_rate": 0.7, "latency": 0.3, "complexity": 0.5,
                "parallel_degree": 2, "session_depth": 0.0, "user_engagement": 1.0,
                "token_cost": 100.0, "text": "fallback"}
    def step(self, hexagram_id):
        self.step_count += 1
        done = self.step_count >= 5
        return (self.reset(), -0.5 if done else 0.0, done, "timeout" if done else "active")

def _random_env_fallback():
    return _RandomEnv()


# ── Quick test ──
if __name__ == "__main__":
    ds = SessionDataset()
    stats = ds.get_session_stats()
    print(f"\nSession stats:")
    for k, v in stats.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nSample 0: {ds[0]['text'][:60]}...")
    print(f"  reward={ds[0]['reward']:.3f}, type={ds[0]['terminal_type']}")
