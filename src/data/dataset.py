"""
Synthetic data generator for YiCeNet pre-training and RL training.

Generates realistic orchestration traces simulating Hermes behavior:
- User intents (diverse orchestration scenarios)
- Execution outcomes (success/timing)
- Session patterns (continuation/abandonment)

This enables offline pre-training before real data accumulation.
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader


# ── Pre-tokenized orchestration scenarios ──
# Each is a (tokens, success_rate, latency_score) tuple
# tokens are random ints in [1, vocab_size-1] representing BPE tokens

ORCHESTRATION_SCENARIOS = [
    # Intent description, typical success rate, avg latency impact
    ("search knowledge base", 0.95, 0.3),
    ("route to multiple APIs and merge", 0.85, 0.7),
    ("sequential API call chain", 0.90, 0.5),
    ("conditional branching based on query type", 0.88, 0.4),
    ("parallel data fetch then aggregate", 0.82, 0.6),
    ("user intent classification first", 0.93, 0.2),
    ("retrieve context then generate response", 0.91, 0.5),
    ("caching layer lookup with fallback", 0.97, 0.1),
    ("multi-step form wizard", 0.78, 0.8),
    ("monitor and poll for completion", 0.75, 0.9),
    ("fan-out to 3 services then merge", 0.80, 0.7),
    ("select best model based on query", 0.89, 0.3),
    ("extract entities then query database", 0.87, 0.5),
    ("summarize long document", 0.86, 0.6),
    ("translate then format output", 0.94, 0.4),
    ("validate input then process", 0.92, 0.2),
    ("stream response with progress updates", 0.83, 0.7),
    ("load balance across workers", 0.96, 0.2),
    ("retry with backoff on failure", 0.71, 0.9),
    ("orchestrate sub-agent delegation", 0.79, 0.8),
    ("tool selection from registry", 0.90, 0.3),
    ("schedule recurring task", 0.88, 0.5),
    ("chain of thought before answering", 0.84, 0.6),
    ("context window management", 0.93, 0.2),
    ("error recovery with graceful degradation", 0.72, 0.8),
]


class SyntheticOrchestrationDataset(Dataset):
    """
    Generate synthetic orchestration traces for pre-training.

    Each sample:
      - input_ids: tokenized intent description
      - attention_mask: padding mask
      - features: 8-dim feature vector for clustering
          [success_rate, latency, complexity, parallel_degree,
           sequential_depth, cache_hit, error_rate, diversity]
      - cluster_target: (optional) ground-truth hexagram cluster assignment
    """

    def __init__(
        self,
        num_samples: int = 10000,
        vocab_size: int = 8000,
        max_seq_len: int = 32,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.rng = random.Random(seed)

        # Generate scenarios for each sample
        self.scenarios = []
        for _ in range(num_samples):
            scenario = self.rng.choice(ORCHESTRATION_SCENARIOS)
            self.scenarios.append(scenario)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        text, success_rate, latency = self.scenarios[idx]

        # Simple tokenization: hash-based pseudo-token IDs
        # In production, use a real tokenizer
        token_ids = [
            (hash(f"{text}_{i}") % (self.vocab_size - 1)) + 1
            for i in range(min(len(text), self.max_seq_len))
        ]
        # Ensure minimum length of 4
        while len(token_ids) < 4:
            token_ids.append(self.rng.randint(1, self.vocab_size - 1))

        seq_len = len(token_ids)
        attention_mask = [1] * seq_len

        # Pad
        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            token_ids = token_ids + [0] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        # Feature vector for clustering (8-dim)
        complexity = self.rng.uniform(0.2, 1.0)
        parallel_degree = self.rng.randint(1, 5)
        sequential_depth = self.rng.randint(1, 6)
        cache_hit = self.rng.random()
        error_rate = 1.0 - success_rate
        diversity = self.rng.uniform(0.1, 0.9)

        features = torch.tensor([
            success_rate,
            latency,
            complexity,
            parallel_degree / 5.0,
            sequential_depth / 5.0,
            cache_hit,
            error_rate,
            diversity,
        ], dtype=torch.float32)

        # Ground-truth cluster: simplified — group by latency buckets
        # In production this comes from real traces
        cluster_id = min(int(latency * 64), 63)

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "features": features,
            "cluster_id": torch.tensor(cluster_id, dtype=torch.long),
            "text": text,
        }


class RLSimulationEnv:
    """
    Lightweight simulation environment for RL training.

    Simulates the "fortune teller—customer" interaction:
    - Agent generates a hexagram (decision skeleton)
    - Environment evaluates it based on simplified success/latency model
    - Returns reward mimicking real user behavior

    This is a simplified world model for offline RL bootstrapping.
    In production, this is replaced by a learned world model
    trained on real Hermes execution logs.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.session_active = True
        self.step_count = 0
        self.max_steps = self.rng.randint(3, 10)
        self.current_reward = 0.0

    def reset(self) -> dict:
        """Reset environment for a new episode. Returns initial state."""
        self.session_active = True
        self.step_count = 0
        self.max_steps = self.rng.randint(3, 10)
        self.current_reward = 0.0

        # Generate random initial state features
        state = {
            "success_rate": self.rng.uniform(0.6, 0.95),
            "latency": self.rng.uniform(0.1, 0.9),
            "complexity": self.rng.uniform(0.2, 0.8),
            "parallel_degree": self.rng.randint(1, 4),
            "session_depth": self.step_count / self.max_steps,
            "user_engagement": 1.0,
            "token_cost": self.rng.uniform(0.0, 0.3),
        }
        return state

    def step(self, hexagram_id: int) -> tuple[dict, float, bool, str]:
        """
        Take a step with the chosen hexagram.

        Args:
            hexagram_id: hexagram index 0-63

        Returns:
            next_state: dict of new state features
            reward: scalar reward
            done: whether session is done
            terminal_type: why the episode ended (when done=True)
                          "active" (not done), "success" (completed),
                          "abandoned" (user left), "timeout" (max steps)
        """
        self.step_count += 1

        # Simulate execution outcome based on hexagram quality
        yangness = bin(hexagram_id).count("1") / 6.0

        # Success probability: moderate yangness is often best (中庸之道)
        success_prob = 0.5 + 0.5 * (1.0 - abs(yangness - 0.5) * 2.0)
        success = self.rng.random() < success_prob

        # Latency: bolder (more yang) = faster but riskier
        latency = 0.3 + 0.6 * (1.0 - yangness)
        token_cost = 0.1 + 0.3 * (1.0 - yangness)

        # User engagement: drops with failures
        engagement_decay = 0.2 if not success else 0.02

        # Determine termination type and reward
        timeout = self.step_count >= self.max_steps
        abandon = not success and self.rng.random() < 0.3
        done = timeout or abandon

        # ── Disambiguated reward ──
        # Natural end (success): neutral-to-positive reward
        # Abandonment: strong negative penalty
        # Mid-session: based on success + continuation

        terminal_type = "active"

        if done and timeout and success:
            # Task completed naturally — slight positive for efficiency
            reward = 2.0 - 0.5 * token_cost - 0.3 * latency
            terminal_type = "success"
        elif done and timeout and not success:
            # Timeout with failures — negative but not abandonment
            reward = -0.5 - 0.5 * token_cost - 0.3 * latency
            terminal_type = "timeout"
        elif done and abandon:
            # User explicitly left after failure — strongest penalty
            reward = -2.0 - 0.5 * token_cost - 0.3 * latency
            terminal_type = "abandoned"
        else:
            # Mid-session step
            continue_bonus = 1.0 if success else -0.5
            reward = (
                + 1.0 * int(success)
                + continue_bonus
                - 0.5 * token_cost
                - 0.3 * latency
            )

        # Build next state
        next_state = {
            "success_rate": 0.7 + 0.3 * success,
            "latency": latency,
            "complexity": self.rng.uniform(0.2, 0.8),
            "parallel_degree": self.rng.randint(1, 4),
            "session_depth": self.step_count / self.max_steps,
            "user_engagement": max(0.0, 1.0 - engagement_decay * self.step_count),
            "token_cost": token_cost,
            "terminal_type": terminal_type if done else "active",
        }

        return next_state, reward, done, terminal_type
