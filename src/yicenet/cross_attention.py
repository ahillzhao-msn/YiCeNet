"""
YiCeNet Cross-Attention — pure numpy, no training required.

Core computation: q · Kᵀ / √d → softmax → threshold → prescription.

This is NOT machine learning. It's deterministic cosine-based similarity
with interpretable attention weights. Every step can be inspected.

Usage:
    attn = CrossAttention()
    weights = attn.compute(query_384d, key_matrix_Nx384)
    rx = ContextPrescription(weights, metadata)
    rx.generate(threshold_high=0.15, threshold_low=0.05)
    # → {"retain_turns": [...], "summarize_turns": [...], "discard_turns": [...]}

See docs/cross-attention-memory-cortex.md for architecture details.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Prescription:
    """Context compression prescription for LOOM to execute.
    
    Attributes:
        retain_turns: turn IDs to keep full context
        summarize_turns: turn IDs to compress to 1-3 fact lines
        discard_turns: turn IDs to remove entirely
        mode: "compress" | "expand" | "full"
        attention_entropy: entropy of attention distribution (0=concentrated, >3=diffuse)
        compression_ratio: estimated fraction of context that can be compressed
        key_insight: human-readable insight about attention pattern
    """
    retain_turns: list[int] = field(default_factory=list)
    summarize_turns: list[int] = field(default_factory=list)
    discard_turns: list[int] = field(default_factory=list)
    mode: str = "compress"
    attention_entropy: float = 0.0
    compression_ratio: float = 0.0
    key_insight: str = ""
    
    def to_dict(self) -> dict:
        """Serialize for JSON return to LOOM."""
        return {
            "mode": self.mode,
            "retain_turns": self.retain_turns,
            "summarize_turns": self.summarize_turns,
            "discard_turns": self.discard_turns,
            "attention_entropy": round(self.attention_entropy, 4),
            "compression_ratio": round(self.compression_ratio, 4),
            "key_insight": self.key_insight,
        }


class CrossAttention:
    """Pure numpy cross-attention over historical turn encoder outputs.
    
    No training, no gradients, no state. Deterministic given same inputs.
    """
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def compute(
        self,
        query: np.ndarray,
        keys: np.ndarray,
    ) -> np.ndarray:
        """Compute attention weights: softmax(q · Kᵀ / √d / τ)
        
        Args:
            query: (384,) current turn's encoder output
            keys: (N, 384) historical encoder outputs
            
        Returns:
            weights: (N,) attention distribution over historical turns
        """
        if keys.shape[0] == 0:
            return np.array([], dtype=np.float32)
        
        d = keys.shape[-1]  # 384
        # q · Kᵀ : cosine in encoder space
        logits = np.dot(keys, query) / np.sqrt(d) / self.temperature
        
        # Numerical stability: softmax
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        weights = exp_logits / (np.sum(exp_logits) + 1e-10)
        
        return weights.astype(np.float64)
    
    def entropy(self, weights: np.ndarray) -> float:
        """Shannon entropy of attention distribution.
        
        0.0 = all weight on one turn (maximally focused)
        >3.0 = nearly uniform (diffuse, no clear focus)
        """
        if weights.size == 0:
            return 0.0
        # Avoid log(0)
        p = np.clip(weights, 1e-10, 1.0)
        return float(-np.sum(p * np.log2(p)))


class ContextPrescription:
    """Generate context compression prescription from attention weights.
    
    Uses dynamic thresholding based on the weight distribution shape.
    """
    
    def __init__(
        self,
        weights: np.ndarray,
        metadata: list[dict],
        n_turns_total: int,
    ):
        self.weights = weights  # (N,)
        self.metadata = metadata  # [{turn_id, hexagram_id, summary}, ...]
        self.n_turns = len(weights)
        self.n_turns_total = n_turns_total
    
    def generate(
        self,
        threshold_high: float | None = None,
        threshold_low: float | None = None,
        min_retain: int | None = None,
        max_discard_ratio: float | None = None,
    ) -> Prescription:
        """Generate context prescription from attention weights.

        All thresholds are NATURAL — derived from the weight distribution
        itself (mean ± std). No external modulation, no hand-tuned constants.
        
        - threshold_high: mean + std (one sigma above mean)
        - threshold_low: max(0, mean - 0.5*std) 
        - min_retain: ceil(N/10), at least 1
        - max_discard_ratio: 70%

        Args:
            threshold_high: override (not recommended — let distribution decide)
            threshold_low: override (not recommended — let distribution decide)
            min_retain: override minimum retain count
            max_discard_ratio: override max discard fraction
            
        Returns:
            Prescription with retain/summarize/discard lists
        """
        rx = Prescription()
        
        if self.n_turns == 0:
            rx.mode = "full"
            return rx
        
        # ── Natural thresholds from weight distribution ──
        mean_w = float(np.mean(self.weights))
        std_w = float(np.std(self.weights))
        
        if threshold_high is None:
            threshold_high = mean_w + std_w  # one sigma above mean
        if threshold_low is None:
            threshold_low = max(0.0, mean_w - std_w * 0.5)  # half sigma below
        if min_retain is None:
            min_retain = max(1, self.n_turns // 10)  # at least 1, up to 10%
        if max_discard_ratio is None:
            max_discard_ratio = 0.7
        
        # Sort turns by attention weight (descending)
        indices = np.argsort(-self.weights)
        sorted_weights = self.weights[indices]
        
        # --- Determine mode from weight distribution ---
        attn_entropy = CrossAttention().entropy(self.weights)
        rx.attention_entropy = attn_entropy
        
        # High entropy = diffuse attention = need more context
        if attn_entropy > 2.5:
            rx.mode = "expand"
        elif attn_entropy > 1.8:
            rx.mode = "compress"  # moderate compression
        else:
            rx.mode = "compress"  # aggressive compression
        
        # Determine max discard count
        max_discard = int(self.n_turns * max_discard_ratio)
        
        # --- Classify each turn ---
        # Iterate from highest to lowest weight
        n_retained = 0
        n_summarized = 0
        n_discarded = 0
        
        for idx in range(self.n_turns):
            turn_idx = indices[idx]
            weight = sorted_weights[idx]
            meta = self.metadata[turn_idx]
            turn_id = meta["turn_id"]
            
            if n_retained < min_retain or weight >= threshold_high:
                rx.retain_turns.append(turn_id)
                n_retained += 1
            elif weight < threshold_low and n_discarded < max_discard:
                rx.discard_turns.append(turn_id)
                n_discarded += 1
            else:
                rx.summarize_turns.append(turn_id)
                n_summarized += 1
        
        # --- Compression ratio ---
        # Assumes: retain = full (1), summarize = 0.1, discard = 0
        total = self.n_turns
        compressed = n_summarized * 0.9 + n_discarded * 1.0
        rx.compression_ratio = compressed / total if total > 0 else 0.0
        
        # --- Key insight (公交站台风格: 简洁、聚焦) ---
        # Skill doc 约定：焦點: #N, 壓縮比 X%，不出现「輪次」前缀（# 已表意）
        top_turns = rx.retain_turns[:3]
        turns_str = ", ".join(f"#{t}" for t in top_turns)
        if attn_entropy < 0.5:
            rx.key_insight = f"焦點: {turns_str}, 壓縮比 {rx.compression_ratio:.0%}"
        elif attn_entropy > 2.5:
            rx.key_insight = f"發散 ({turns_str} 略高), 壓縮比 {rx.compression_ratio:.0%}"
        else:
            rx.key_insight = f"焦點: {turns_str}, 壓縮比 {rx.compression_ratio:.0%}"
        
        return rx
