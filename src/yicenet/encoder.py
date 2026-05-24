"""
Tiny Transformer encoder — 4 layers, hidden_dim=256, 4 heads, ~8M params.

Designed for YiCeNet: encodes user intent + orchestration context
into a 256-dim state vector for hexagram-based decision making.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding — no learned params."""

    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """Pre-LN Transformer encoder block."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-LN: normalize before attention
        attn_out, _ = self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            key_padding_mask=mask, need_weights=False,
        )
        x = x + self.dropout(attn_out)

        # FFN with pre-LN
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x)))))
        x = x + self.dropout(ffn_out)
        return x


class TinyEncoder(nn.Module):
    """
    Tiny Transformer encoder: 4 layers, 256-dim, 4 heads, vocab=8000.

    Input:  token ids (B, T)
    Output: pooled 256-dim state vector h

    Params: ~5.7M total (2.05M token embedding + ~3.3M transformer blocks + ~0.3M heads)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding — this dominates param count
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_dim, padding_idx=0
        )

        # Positional encoding (fixed sinusoidal, no params)
        self.pos_encoding = PositionalEncoding(
            config.hidden_dim, config.max_seq_len, config.dropout
        )

        # 4 transformer blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.intermediate_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_encoder_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_dim)

        # Projection to state vector h
        self.state_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) token indices
            attention_mask: (B, T) 1=keep, 0=mask

        Returns:
            h: (B, D) pooled state vector
        """
        B, T = input_ids.shape

        # (B, T, D)
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)

        # Invert mask for Transformer (True = masked)
        padding_mask = None
        if attention_mask is not None:
            padding_mask = ~attention_mask.bool()

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask=padding_mask)

        # Final norm
        x = self.norm(x)

        # Mean-pool over sequence dimension (masked)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            h = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            h = x.mean(dim=1)

        # Project to state vector
        h = self.state_proj(h)
        return h

    def get_param_count(self) -> dict:
        """Return parameter breakdown."""
        emb = sum(p.numel() for p in self.token_embedding.parameters())
        blocks = sum(p.numel() for p in self.blocks.parameters())
        other = sum(p.numel() for p in [
            *self.norm.parameters(),
            *self.state_proj.parameters(),
        ])
        return {"embedding": emb, "transformer_blocks": blocks, "other": other, "total": emb + blocks + other}
