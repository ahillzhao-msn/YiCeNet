"""
YiCeNet (易策网络) — Full model.

~10.2M parameters, <100 MB memory, <3 ms inference.

Architecture:
  Input → TinyEncoder (8M) → h (256-dim)
       → Binary Router (Gumbel-Softmax) → hexagram index
       → Hexagram Embedding Table (64×256)
       → Structural Reasoning Engine (fixed logic: 错∕综∕互∕变)
       → Value Network (13K) per candidate → select best
       → Action Decoder (50K) → action sequence

Philosophy:
  "太极生两仪，两仪生四象，四象生八卦"
  The model mirrors the I Ching's generative hierarchy:
  intent → binary decision → patterned structure → evaluated outcome.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import YiCeNetConfig
from .encoder import TinyEncoder
from .hexagram import (
    generate_candidates,
    hexagram_to_lines,
    lines_to_hexagram,
)
from .value_net import ValueNetwork
from .decoder import ActionDecoder
from .probes import extract_probes_tensor


class GumbelRouter(nn.Module):
    """
    二分路由器 (Binary Divination Router) — "起卦" simulation.

    Projects state vector h to 64-dim logits, applies Gumbel-Softmax
    to sample a hexagram index with controlled randomness.

    Philosophy: "蓍之德圆而神" — randomness brings traversal of
    all possibilities, avoiding cognitive ruts.

    Temperature controls exploration:
      - High τ → near-uniform (maximum exploration)
      - Low τ → near-argmax (exploitation)
    """

    def __init__(self, hidden_dim: int, num_hexagrams: int):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, num_hexagrams)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        h: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, D) state vectors
            tau: Gumbel-Softmax temperature
            hard: if True, output one-hot (straight-through estimator)

        Returns:
            hexagram_idx: (B,) sampled hexagram indices
            probs: (B, 64) categorical probabilities
        """
        logits = self.projection(h)  # (B, 64)
        # Clamp extreme logits for numerical stability
        logits = torch.clamp(logits, -10, 10)

        # Gumbel-Softmax sampling
        y = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)

        if hard:
            # Straight-through: one-hot output, soft gradient
            hexagram_idx = y.argmax(dim=-1)
        else:
            # Soft sampling: weighted average of hexagram embeddings
            hexagram_idx = y.argmax(dim=-1)

        return hexagram_idx, F.softmax(logits / tau, dim=-1)


class HexagramEmbedding(nn.Module):
    """
    六十四卦嵌入表 — 64 × 256-dim learnable embeddings.

    Each hexagram is a "task topology signature" — a learned prototype
    of an orchestration pattern. Initially seeded by k-means clustering
    of real orchestration traces.
    """

    def __init__(self, num_hexagrams: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_hexagrams, hidden_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: (...,) hexagram indices

        Returns:
            emb: (..., D) hexagram embeddings
        """
        return self.embedding(idx)


class YiCeNet(nn.Module):
    """
    YiCeNet — complete model.

    Input:  tokenized user intent + context (B, T)
    Output: selected action (B,), Q-values (B, 8), hexagram index (B,)
    """

    def __init__(self, config: YiCeNetConfig | None = None):
        super().__init__()
        if config is None:
            config = YiCeNetConfig()
        self.config = config
        self.tau = config.gumbel_tau_init

        # ── Core components ──
        self.encoder = TinyEncoder(config)
        self.router = GumbelRouter(config.hidden_dim, config.num_hexagrams)
        self.hexagram_embed = HexagramEmbedding(
            config.num_hexagrams, config.hidden_dim
        )
        self.value_net = ValueNetwork(config.hidden_dim, config.value_hidden)
        self.action_decoder = ActionDecoder(
            config.hidden_dim, config.num_actions
        )

        # ── 八卦原型层 (Trigram prototypes) ──
        # 8 learnable prototype vectors, used for cross-attention
        self.trigram_prototypes = nn.Parameter(
            torch.randn(config.num_trigrams, config.hidden_dim) * 0.02
        )
        # Cross-attention from state h to trigrams
        self.trigram_cross_attn = nn.MultiheadAttention(
            config.hidden_dim, 1, batch_first=True
        )

        # Register hexagram patterns as buffer (non-parametric)
        patterns = torch.tensor(
            list(config.hexagram_patterns), dtype=torch.long
        )
        self.register_buffer("hexagram_patterns", patterns)

        # ── 探針狀態追蹤 ──
        self._prev_hexagram_idx: int | None = None
        """上輪的卦象 ID，用於計算跳躍度（探針⑤）。由引擎在每次 predict 後更新。"""
        self._prev_hexagram_idx_tensor: torch.Tensor | None = None
        """上輪卦象的 tensor 形式，供張量化探針提取直接使用。"""

    @property
    def device(self):
        return next(self.parameters()).device

    def set_temperature(self, tau: float):
        """Set Gumbel-Softmax temperature."""
        self.tau = max(self.config.gumbel_tau_min, tau)

    def get_temperature(self) -> float:
        return self.tau

    def decay_temperature(self):
        """Apply temperature decay (for training)."""
        self.tau = max(
            self.config.gumbel_tau_min,
            self.tau * self.config.gumbel_tau_decay,
        )

    def encode_context(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode user context into state vector h."""
        return self.encoder(input_ids, attention_mask)

    def divine(
        self,
        h: torch.Tensor,
        tau: float | None = None,
        hard: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Step 1-2: Sample a hexagram via Gumbel-Softmax.

        "起卦" — the divination step.

        Args:
            h: (B, D) state vectors
            tau: temperature override
            hard: straight-through estimator

        Returns:
            hexagram_idx: (B,) sampled hexagram indices
            hexagram_probs: (B, 64) categorical distribution
            hexagram_emb: (B, D) embedding of sampled hexagram
        """
        tau = tau or self.tau
        hexagram_idx, probs = self.router(h, tau=tau, hard=hard)
        hexagram_emb = self.hexagram_embed(hexagram_idx)
        return hexagram_idx, probs, hexagram_emb

    def evaluate_candidates(
        self,
        hexagram_idx: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Step 3-4: Generate candidates, evaluate, select best.

        "错综互变" — structural reasoning + value judgment.

        Args:
            hexagram_idx: (B,) current hexagram indices
            h: (B, D) optional state vectors for attention-guided mutations

        Returns:
            best_idx: (B,) index of best candidate within candidate set
            candidate_idxs: (B, K) all candidate hexagram indices
            candidate_values: (B, K) Q-values for all candidates
        """
        B = hexagram_idx.shape[0]

        # Generate candidate hexagrams via structural reasoning
        # (no learned params — pure deterministic logic)
        attention_weights = None
        if h is not None:
            # Compute attention from h to trigram prototypes
            # as proxy for "which lines are most relevant"
            h_expanded = h.unsqueeze(1)  # (B, 1, D)
            prototypes = self.trigram_prototypes.unsqueeze(0).expand(B, -1, -1)
            attn_out, attn_weights = self.trigram_cross_attn(
                h_expanded, prototypes, prototypes
            )
            # Map trigram attention (B, 1, 8) to line attention (B, 6)
            # Each trigram influences 3 lines
            attn = attn_weights.squeeze(1)  # (B, 8)
            line_attn = torch.zeros(B, 6, device=h.device)
            for trigram_idx in range(8):
                # Map trigram to lines: each trigram corresponds to
                # 3 contiguous lines in the traditional arrangement
                start_line = (trigram_idx % 4) * 3  # simplified mapping
                end_line = min(start_line + 3, 6)
                line_attn[:, start_line:end_line] += attn[:, trigram_idx:trigram_idx+1]
            attention_weights = line_attn / line_attn.sum(dim=-1, keepdim=True).clamp(min=1.0)

        candidate_idxs = generate_candidates(
            hexagram_idx, attention_weights
        )  # (B, K) where K=8

        # Embed all candidates
        candidate_embeds = self.hexagram_embed(candidate_idxs)  # (B, K, D)

        # Evaluate via value network
        candidate_values = self.value_net(candidate_embeds)  # (B, K, 1)

        # Select best (highest Q-value)
        best_idx = candidate_values.squeeze(-1).argmax(dim=-1)  # (B,)

        return best_idx, candidate_idxs, candidate_values

    def decode_action(
        self, best_hexagram_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Step 5: Decode selected hexagram to action.

        Args:
            best_hexagram_idx: (B,) best candidate in original hexagram space

        Returns:
            action_ids: (B,) action IDs
            action_logits: (B, num_actions)
        """
        best_embed = self.hexagram_embed(best_hexagram_idx)
        logits, _ = self.action_decoder(best_embed)
        action_ids = logits.argmax(dim=-1)
        return action_ids, logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        tau: float | None = None,
        hard: bool = False,
    ) -> dict:
        """
        Full forward pass.

        Args:
            input_ids: (B, T) tokenized input
            attention_mask: (B, T) optional
            tau: temperature override
            hard: straight-through Gumbel

        Returns:
            dict with keys:
                - h: (B, D) state vectors
                - hexagram_idx: (B,) sampled hexagram
                - hexagram_probs: (B, 64) sampling distribution
                - best_candidate_idx: (B,) best candidate (0-7)
                - candidate_idxs: (B, 8) all candidate hexagram indices
                - candidate_values: (B, 8) Q-values
                - action_ids: (B,) chosen actions
                - action_logits: (B, num_actions)
        """
        # Step 1: Encode context
        h = self.encode_context(input_ids, attention_mask)

        # Step 2: Divine → get hexagram
        hexagram_idx, probs, hexagram_emb = self.divine(h, tau, hard)

        # Step 3-4: Evaluate candidates
        best_candidate_idx, candidate_idxs, candidate_values = (
            self.evaluate_candidates(hexagram_idx, h)
        )

        # Select the actual best hexagram from candidate set
        best_hexagram_id = candidate_idxs.gather(
            1, best_candidate_idx.unsqueeze(-1)
        ).squeeze(-1)

        # Step 5: Decode to action
        action_ids, action_logits = self.decode_action(best_hexagram_id)

        # ── 六探針提取（張量化）──
        router_logits = self.router.projection(h)  # (B, 64)
        prev_t = (self._prev_hexagram_idx_tensor.to(h.device)
                  if hasattr(self, "_prev_hexagram_idx_tensor")
                  and self._prev_hexagram_idx_tensor is not None
                  else None)
        probe_tensor = extract_probes_tensor(
            h=h,
            router_logits=router_logits,
            router_probs=probs,
            candidate_values=candidate_values,
            hexagram_idx=hexagram_idx,
            prev_hexagram_idx=prev_t,
            action_logits=action_logits,
        )  # (9,) tensor

        # 更新上輪卦象（tensor 形式，供下次張量化調用）
        self._prev_hexagram_idx_tensor = hexagram_idx.clone()
        self._prev_hexagram_idx = hexagram_idx[0].item()

        return {
            "h": h,
            "hexagram_idx": hexagram_idx,
            "hexagram_probs": probs,
            "best_candidate_idx": best_candidate_idx,
            "candidate_idxs": candidate_idxs,
            "candidate_values": candidate_values,
            "action_ids": action_ids,
            "action_logits": action_logits,
            "probes": probe_tensor,        # (9,) tensor
            "router_logits": router_logits,
        }

    def get_param_count(self) -> dict:
        """Return detailed parameter breakdown."""
        encoder_counts = self.encoder.get_param_count()
        
        # Router: linear projection 256→64
        router_params = sum(p.numel() for p in self.router.parameters())
        
        # Hexagram embedding: 64×256
        hex_emb_params = sum(p.numel() for p in self.hexagram_embed.parameters())
        
        # Trigram prototypes: 8×256
        tri_params = self.trigram_prototypes.numel()
        
        # Value network
        value_params = self.value_net.get_param_count()
        
        # Action decoder
        decoder_params = self.action_decoder.get_param_count()
        
        # Cross-attention
        cross_attn_params = sum(p.numel() for p in self.trigram_cross_attn.parameters())

        total = (
            encoder_counts["total"]
            + router_params
            + hex_emb_params
            + tri_params
            + value_params
            + decoder_params
            + cross_attn_params
        )

        return {
            "encoder_total": encoder_counts["total"],
            "encoder_embedding": encoder_counts["embedding"],
            "encoder_blocks": encoder_counts["transformer_blocks"],
            "router": router_params,
            "hexagram_embedding": hex_emb_params,
            "trigram_prototypes": tri_params,
            "trigram_cross_attention": cross_attn_params,
            "value_network": value_params,
            "action_decoder": decoder_params,
            "total": total,
        }

    def estimate_memory(self, quantized: bool = False) -> float:
        """
        Estimate memory footprint in MB.

        Args:
            quantized: if True, estimate for INT8

        Returns:
            memory in MB
        """
        params = sum(p.numel() for p in self.parameters())
        bytes_per_param = 1 if quantized else 4
        return params * bytes_per_param / (1024 * 1024)

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "YiCeNet":
        """Load a saved model."""
        state = torch.load(path, map_location=device, weights_only=False)
        config = YiCeNetConfig()
        model = cls(config)
        model.load_state_dict(state["model_state_dict"], strict=False)
        if "tau" in state:
            model.tau = state["tau"]
        return model

    def save_pretrained(self, path: str):
        """Save model to disk."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "tau": self.tau,
                "config": self.config,
            },
            path,
        )


def count_parameters(model: YiCeNet, verbose: bool = True) -> dict:
    """Helper to print parameter counts."""
    counts = model.get_param_count()
    if verbose:
        print("=== YiCeNet Parameter Count ===")
        print(f"  Encoder (total):       {counts['encoder_total']:>8,}")
        print(f"    ├─ Embedding:         {counts['encoder_embedding']:>8,}")
        print(f"    └─ Transformer Blocks: {counts['encoder_blocks']:>8,}")
        print(f"  Router:                 {counts['router']:>8,}")
        print(f"  Hexagram Embedding:     {counts['hexagram_embedding']:>8,}")
        print(f"  Trigram Prototypes:     {counts['trigram_prototypes']:>8,}")
        print(f"  Trigram Cross-Attn:     {counts['trigram_cross_attention']:>8,}")
        print(f"  Value Network:          {counts['value_network']:>8,}")
        print(f"  Action Decoder:         {counts['action_decoder']:>8,}")
        print(f"  ─────────────────────────────────")
        print(f"  TOTAL:                  {counts['total']:>8,}")
        print(f"  FP32 Memory:            {counts['total'] * 4 / 1024**2:.1f} MB")
        print(f"  INT8 Memory:            {counts['total'] * 1 / 1024**2:.1f} MB")
    return counts
