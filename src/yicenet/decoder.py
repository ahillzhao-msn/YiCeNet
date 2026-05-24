"""
Action Decoder — maps selected hexagram embedding to orchestration action sequence.

~26K params: LayerNorm + Linear projection 256 → 50 (orchestration primitives).

The LayerNorm ensures stable logits regardless of input embedding scale,
preventing NaN from extreme value propagation.
"""

import math
import torch
import torch.nn as nn


class ActionDecoder(nn.Module):
    """
    Decodes the optimal hexagram embedding into an orchestration action ID.

    The 50 output classes represent orchestration primitives like:
    - Route to service A
    - Parallel invoke B and C
    - Sequential: D → E
    - Aggregate results
    - Wait / Poll
    - Notify user
    - Cache lookup
    - etc.

    Philosophy: "卦象 → 编排原语序列" — the hexagram's structural
    pattern is decoded into a concrete action plan.
    """

    def __init__(self, hidden_dim: int = 256, num_actions: int = 50):
        super().__init__()
        # LayerNorm to stabilize inputs regardless of embedding scale
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, num_actions)

        # Action embedding table for candidate evaluation
        self.action_embed = nn.Embedding(num_actions, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        # Scale initialization based on expected output range
        nn.init.normal_(self.decoder.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.decoder.bias)

    def forward(
        self, hexagram_embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hexagram_embed: (B, D) selected hexagram embedding

        Returns:
            action_logits: (B, num_actions) logits over action space
            action_embeds: (num_actions, D) action embeddings
        """
        # Normalize to prevent extreme values → NaN
        x = self.input_norm(hexagram_embed)
        logits = self.decoder(x)
        # Clamp to prevent extreme logits
        logits = torch.clamp(logits, -20, 20)
        action_embeds = self.action_embed.weight  # (num_actions, D)
        return logits, action_embeds

    def decode_to_action_ids(
        self, hexagram_embed: torch.Tensor
    ) -> torch.Tensor:
        """Get deterministic action IDs (argmax)."""
        logits, _ = self.forward(hexagram_embed)
        return logits.argmax(dim=-1)

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
