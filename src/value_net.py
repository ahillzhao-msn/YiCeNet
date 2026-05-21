"""
Value Network — evaluates candidate hexagram quality.

A small MLP (256→128→1, ~13K params) that scores each candidate
orchestration pattern based on expected utility.
"""

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """
    Three-layer MLP value head.

    Input:  candidate hexagram embedding (256-dim) + optional state context
    Output: scalar Q-value representing expected utility

    Philosophy: "卦爻辞逻辑" — like the I Ching judgment texts,
    this network learns to evaluate which pattern is auspicious (吉)
    vs inauspicious (凶) based on actual execution outcomes.
    """

    def __init__(self, hidden_dim: int = 256, value_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, value_hidden),
            nn.LayerNorm(value_hidden),
            nn.GELU(),
            nn.Linear(value_hidden, value_hidden // 2),
            nn.GELU(),
            nn.Linear(value_hidden // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.orthogonal_(p, gain=0.5)
            if isinstance(p, nn.LayerNorm):
                nn.init.constant_(p.weight, 1.0)
                nn.init.constant_(p.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) or (B, K, D) candidate hexagram embeddings

        Returns:
            values: (B, 1) or (B, K, 1) scalar Q-values
        """
        return self.net(x)

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
