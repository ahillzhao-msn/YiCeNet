"""
World Model V4 — 察言观色: how the customer will react (DESIGN-phase4.md §3.2).

    P(outcome | question, 本卦, candidate 之卦, context known at question time)

Input:  frozen-encoder state of the question (256, L2-normalised) → 16-dim projection
        ⊕ onehot(本卦) ⊕ onehot(之卦)·has_之卦 ⊕ context_pre·has_context ⊕ the two flags
Output: sigmoid probabilities for OUTCOMES, trained with BCE against observed outcomes.

Nothing observed after the question is an input: the features are built only from
FEATURE_FIELDS (see sample_features), and a test pins that.

Legacy samples (no 之卦, no context) train the reduced part "question + 本卦 →
outcome".  Only samples that carry the 之卦 can tell the candidates apart, so the
re-rank skill (which gates λ at inference) is measured on those alone.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUTCOMES = ("corrected", "completed", "abandoned", "continued")

# DaemonContextCollector.build_vector keys, in order: the previous turn's context.
CONTEXT_KEYS = (
    "tok_user_input_len", "tok_is_first_turn",
    "tok_prompt_tokens", "tok_completion_tokens", "tok_api_duration",
    "tok_tool_count", "tok_tool_success_rate", "tok_tool_retry_count",
    "tok_tool_duration", "tok_tool_output_size", "tok_tool_diversity",
    "tok_response_len", "tok_has_code", "tok_code_block_count",
    "tok_hex_conf", "tok_hex_q_gap", "tok_hex_entropy",
    "tok_user_speed", "tok_user_speed_ratio", "tok_mood_trend",
    "tok_drift_trend", "tok_is_prev_correction", "tok_is_prev_praise",
    "tok_user_satisfaction", "tok_is_correction", "tok_is_praise", "tok_is_abandon",
)
# Known at question time besides the previous turn: session position and the question itself
# (a question can be the customer's correction of the previous answer).
QUESTION_CONTEXT = ("turn_position", "q_is_correction", "q_is_praise", "q_len")
CONTEXT_DIM = len(CONTEXT_KEYS) + len(QUESTION_CONTEXT)
QUESTION_DIM = 256
NUM_HEXAGRAMS = 64
MAX_TURN = 50

# The only sample fields the features may read.  Outcomes and the reaction text are
# labels; they must never be here.
FEATURE_FIELDS = ("user_text", "base_hexagram", "chosen_hexagram", "context_pre", "turn_id")

DEFAULT_UTILITY = {"completed": 1.0, "continued": 1.0, "corrected": -1.0, "abandoned": -1.0}


# ── Features ──────────────────────────────────────────────────────────────────

def context_features(context_pre: Optional[dict], turn_id: Optional[int], question: str) -> tuple[list[float], bool]:
    """(CONTEXT_DIM vector, has_context).  `context_pre` is the previous turn's context_vector."""
    from yicenet.external_metrics import _check_patterns, _CORRECTION_PATTERNS, _PRAISE_PATTERNS

    prev = context_pre if isinstance(context_pre, dict) else {}
    vec = [max(-2.0, min(2.0, float(prev.get(k, 0.0) or 0.0))) for k in CONTEXT_KEYS]
    vec += [
        min(max(int(turn_id or 0), 0), MAX_TURN) / MAX_TURN,
        float(_check_patterns(question, _CORRECTION_PATTERNS)),
        float(_check_patterns(question, _PRAISE_PATTERNS)),
        min(len(question) / 512.0, 1.0),
    ]
    return vec, bool(prev)


def sample_features(sample: dict) -> dict:
    """The question-time view of a trajectory: only FEATURE_FIELDS are read."""
    s = {k: sample.get(k) for k in FEATURE_FIELDS}
    question = s["user_text"] or ""
    ctx, has_ctx = context_features(s["context_pre"], s["turn_id"], question)
    chosen = s["chosen_hexagram"]
    has_chosen = isinstance(chosen, int) and 0 <= chosen < NUM_HEXAGRAMS
    base = s["base_hexagram"]
    return {
        "question": question,
        "base": base if isinstance(base, int) and 0 <= base < NUM_HEXAGRAMS else None,
        "chosen": chosen if has_chosen else 0,
        "has_chosen": float(has_chosen),
        "ctx": ctx,
        "has_ctx": float(has_ctx),
    }


def normalise_sample(sample: dict) -> dict:
    """Legacy buffer rows keep the 本卦 only in hexagram_evolution."""
    if sample.get("base_hexagram") is None:
        evo = sample.get("hexagram_evolution") or []
        if evo and isinstance(evo[0], int):
            sample = {**sample, "base_hexagram": evo[0]}
    return sample


class QuestionEncoder:
    """Frozen prior encoder: question → (L2-normalised h0, argmax 本卦)."""

    def __init__(self, prior):
        self.prior = prior.eval()
        self.device = next(prior.parameters()).device
        self.raw: Optional[torch.Tensor] = None  # last call's un-normalised h0

    @torch.no_grad()
    def __call__(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        from yicenet.tokenizer import encode
        hs = []
        for t in texts:
            ids, mask = encode(t, max_len=self.prior.config.max_seq_len)
            hs.append(self.prior.encoder(ids.to(self.device), mask.to(self.device)))
        h0 = torch.cat(hs).float()
        base = self.prior.router.projection(h0).argmax(dim=-1)
        self.raw = h0
        return F.normalize(h0, dim=-1).cpu(), base.cpu()


@dataclass
class WMData:
    h: torch.Tensor           # (N, 256)
    base: torch.Tensor        # (N,) long
    chosen: torch.Tensor      # (N,) long (0 when absent)
    has_chosen: torch.Tensor  # (N,)
    ctx: torch.Tensor         # (N, CONTEXT_DIM)
    has_ctx: torch.Tensor     # (N,)
    y: torch.Tensor           # (N, len(OUTCOMES))
    w: torch.Tensor           # (N,) training weights, mean 1
    ts: np.ndarray            # (N,) timestamps
    producers: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ts)

    def subset(self, idx) -> "WMData":
        idx = torch.as_tensor(np.asarray(idx, dtype=np.int64))
        w = self.w[idx]
        return WMData(self.h[idx], self.base[idx], self.chosen[idx], self.has_chosen[idx],
                      self.ctx[idx], self.has_ctx[idx], self.y[idx],
                      w / w.mean().clamp(min=1e-8) if len(idx) else w,
                      self.ts[idx.numpy()], [self.producers[i] for i in idx.tolist()])

    def inputs(self) -> tuple:
        return self.h, self.base, self.chosen, self.has_chosen, self.ctx, self.has_ctx


def sample_weights(samples: Sequence[dict], now: float, tau_days: float = 30.0,
                   alpha: float = 1.5) -> np.ndarray:
    """Time decay × producer balance × inverse propensity, normalised to mean 1.

    Producer balance is n_p^-½: a platform with 100× the samples weighs 10× as much
    in total, not 100×.  IPS: only the chosen 之卦's outcome is ever observed.
    """
    from collections import Counter
    from yicenet.world_model import power_law_weight

    counts = Counter(s.get("producer", "unknown") for s in samples)
    w = np.empty(len(samples), dtype=np.float64)
    for i, s in enumerate(samples):
        decay = power_law_weight(float(s.get("timestamp", now)), now, tau_days, alpha)
        balance = counts[s.get("producer", "unknown")] ** -0.5
        p = s.get("chosen_prob")
        ips = 1.0 / min(max(float(p), 0.05), 1.0) if isinstance(p, (int, float)) and p > 0 else 1.0
        w[i] = decay * balance * ips
    return w / max(w.mean(), 1e-12) if len(w) else w


def build_dataset(samples: Sequence[dict], encoder: Callable, now: Optional[float] = None) -> WMData:
    """Featurise trajectories.  `encoder(texts) -> (h0 (N,256), argmax 本卦 (N,))`."""
    now = now or time.time()
    samples = [normalise_sample(s) for s in samples]
    feats = [sample_features(s) for s in samples]
    h, argmax_base = encoder([f["question"] for f in feats])
    base = [f["base"] if f["base"] is not None else int(argmax_base[i]) for i, f in enumerate(feats)]
    return WMData(
        h=h.float(),
        base=torch.tensor(base, dtype=torch.long),
        chosen=torch.tensor([f["chosen"] for f in feats], dtype=torch.long),
        has_chosen=torch.tensor([f["has_chosen"] for f in feats], dtype=torch.float32),
        ctx=torch.tensor([f["ctx"] for f in feats], dtype=torch.float32).reshape(len(feats), CONTEXT_DIM),
        has_ctx=torch.tensor([f["has_ctx"] for f in feats], dtype=torch.float32),
        y=torch.tensor([[float(bool(s.get(o))) for o in OUTCOMES] for s in samples],
                       dtype=torch.float32).reshape(len(samples), len(OUTCOMES)),
        w=torch.tensor(sample_weights(samples, now), dtype=torch.float32),
        ts=np.array([float(s.get("timestamp", now)) for s in samples]),
        producers=[s.get("producer", "unknown") for s in samples],
    )


def time_split(ts: np.ndarray, test_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """(train idx, test idx): the newest `test_fraction` of samples is the test set."""
    order = np.argsort(ts, kind="stable")
    n_test = int(round(len(ts) * test_fraction))
    n_test = min(max(n_test, 1), len(ts) - 1) if len(ts) > 1 else 0
    return order[: len(ts) - n_test], order[len(ts) - n_test:]


# ── Model ─────────────────────────────────────────────────────────────────────

class WorldModelV4(nn.Module):
    """~16k parameters; trains on CPU in seconds."""

    def __init__(self, question_dim: int = QUESTION_DIM, q_proj: int = 16,
                 context_dim: int = CONTEXT_DIM, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(question_dim, q_proj)
        in_dim = q_proj + 2 * NUM_HEXAGRAMS + context_dim + 2
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden, len(OUTCOMES))
        self.meta: dict = {}

    def forward(self, h, base, chosen, has_chosen, ctx, has_ctx) -> torch.Tensor:
        """Outcome logits (N, len(OUTCOMES))."""
        q = F.gelu(self.q_proj(h))
        base_1h = F.one_hot(base, NUM_HEXAGRAMS).float()
        chosen_1h = F.one_hot(chosen, NUM_HEXAGRAMS).float() * has_chosen.unsqueeze(-1)
        x = torch.cat([q, base_1h, chosen_1h, ctx * has_ctx.unsqueeze(-1),
                       has_chosen.unsqueeze(-1), has_ctx.unsqueeze(-1)], dim=-1)
        return self.head(self.body(x))

    @torch.no_grad()
    def proba(self, *inputs) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self(*inputs))

    @torch.no_grad()
    def utility(self, *inputs, weights: Optional[dict] = None) -> torch.Tensor:
        return utility_of(self.proba(*inputs), weights)

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(), "meta": self.meta,
                    "arch": {"context_dim": CONTEXT_DIM, "outcomes": OUTCOMES}}, path)

    @classmethod
    def load(cls, path: str) -> "WorldModelV4":
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if tuple(saved.get("arch", {}).get("outcomes", ())) != OUTCOMES:
            raise ValueError(f"{path}: not a WorldModelV4 checkpoint")
        wm = cls()
        wm.load_state_dict(saved["state_dict"])
        wm.meta = saved.get("meta", {})
        return wm.eval()


def utility_of(p: torch.Tensor, weights: Optional[dict] = None) -> torch.Tensor:
    """u = w·p over OUTCOMES."""
    weights = weights or DEFAULT_UTILITY
    w = torch.tensor([float(weights.get(o, 0.0)) for o in OUTCOMES], dtype=p.dtype)
    return p @ w


def train_world_model(train: WMData, seed: int = 0, max_epochs: int = 400, patience: int = 40,
                      lr: float = 3e-3, weight_decay: float = 1e-3) -> WorldModelV4:
    """Full-batch weighted BCE; early stopping on the newest 15% of the training set."""
    torch.manual_seed(seed)
    fit_idx, val_idx = time_split(train.ts, 0.15)
    fit, val = train.subset(fit_idx), train.subset(val_idx)
    wm = WorldModelV4()
    opt = torch.optim.AdamW(wm.parameters(), lr=lr, weight_decay=weight_decay)

    def loss_on(d: WMData) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(wm(*d.inputs()), d.y, reduction="none").mean(-1)
        return (bce * d.w).mean()

    best, best_state, bad = math.inf, None, 0
    for _ in range(max_epochs):
        wm.train()
        opt.zero_grad()
        loss_on(fit).backward()
        opt.step()
        wm.eval()
        with torch.no_grad():
            v = loss_on(val).item() if len(val) else 0.0
        if v < best - 1e-5:
            best, best_state, bad = v, {k: t.clone() for k, t in wm.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        wm.load_state_dict(best_state)
    return wm.eval()


# ── Honest evaluation (§3.4) ──────────────────────────────────────────────────

def base_rates(train: WMData, recent_fraction: float = 0.15, min_n: int = 30) -> torch.Tensor:
    """The base-rate predictor: outcome prevalence over the newest training samples.

    Outcome rates drift (abandonment doubled within weeks), so a model can beat an
    all-time average just by tracking the current prevalence.  The baseline gets
    the same recency the WM's early stopping sees, so skill measures only what the
    question, the 卦 and the context add.
    """
    order = np.argsort(train.ts, kind="stable")
    n = min(len(order), max(min_n, int(round(len(order) * recent_fraction))))
    y = train.y[torch.as_tensor(order[len(order) - n:])]
    return y.mean(0).clamp(0.01, 0.99)


def _log_loss(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-outcome mean log-loss, (len(OUTCOMES),)."""
    return F.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), y, reduction="none").mean(0)


def auc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Mann–Whitney AUC; None when only one class is present."""
    pos, neg = labels > 0.5, labels <= 0.5
    if not pos.any() or not neg.any():
        return None
    ranks = np.empty(len(scores))
    order = np.argsort(scores, kind="stable")
    ranks[order] = np.arange(1, len(scores) + 1)
    for v in np.unique(scores):  # average ranks on ties
        tie = scores == v
        ranks[tie] = ranks[tie].mean()
    n_pos, n_neg = pos.sum(), neg.sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_ci(x: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple:
    """Percentile bootstrap CI of the mean."""
    if len(x) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _metrics(p: torch.Tensor, y: torch.Tensor, rates: torch.Tensor) -> dict:
    ll = _log_loss(p, y)
    ll_base = _log_loss(rates.expand_as(y), y)
    # Per-sample gain over the base rate, normalised like `skill`: its CI says
    # whether the gain is real or seed noise.
    per_sample = F.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), y, reduction="none")
    per_base = F.binary_cross_entropy(rates.expand_as(y), y, reduction="none")
    scale = ll_base.mean().clamp(min=1e-6)
    gain = ((per_base - per_sample).mean(-1) / scale).numpy().astype(np.float64)
    per = {}
    for j, o in enumerate(OUTCOMES):
        per[o] = {"log_loss": round(ll[j].item(), 4), "base_log_loss": round(ll_base[j].item(), 4),
                  "auc": auc(p[:, j].numpy(), y[:, j].numpy())}
    # Normalised total gain; per-outcome ratios blow up on near-constant outcomes.
    skill = ((ll_base.mean() - ll.mean()) / scale).item()
    lo, hi = bootstrap_ci(gain)
    return {"n": len(y), "log_loss": round(ll.mean().item(), 4),
            "base_log_loss": round(ll_base.mean().item(), 4), "skill": round(skill, 4),
            "skill_ci": [round(lo, 4), round(hi, 4)], "beats_base": lo > 0, "per_outcome": per}


def evaluate(wm: Optional[WorldModelV4], test: WMData, rates: torch.Tensor, min_full: int = 30) -> dict:
    """Held-out metrics vs the base rate.

    `skill` = 1 − log-loss / base-rate log-loss, over all outcomes (> 0: better);
    `beats_base` = the bootstrap CI of that gain excludes 0.
    `rerank_skill` = the skill on samples that carry the 之卦 — the only ones that
    say anything about choosing between candidates; 0 when there are < min_full
    or its CI includes 0.
    """
    p = wm.proba(*test.inputs()) if wm is not None else rates.expand_as(test.y)
    out = _metrics(p, test.y, rates)
    full = (test.has_chosen > 0.5).nonzero().squeeze(-1)
    out["n_full"] = int(len(full))
    if len(full) >= min_full:
        out["full"] = _metrics(p[full], test.y[full], rates)
        out["rerank_skill"] = out["full"]["skill"] if out["full"]["beats_base"] else 0.0
    else:
        out["rerank_skill"] = 0.0
    return out


# ── Loading for inference ─────────────────────────────────────────────────────

def active_world_model_entry(checkpoint_dir: Path) -> Optional[dict]:
    try:
        reg = json.loads((Path(checkpoint_dir) / "registry.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return (reg.get("world_model") or {}).get("active")


def rerank_lambda(entry: Optional[dict], lambda_max: float) -> float:
    """λ = λ_max · clamp(rerank_skill, 0, 1); an unproven WM gets 0."""
    if not entry:
        return 0.0
    skill = float(entry.get("rerank_skill", 0.0) or 0.0)
    return float(lambda_max) * min(max(skill, 0.0), 1.0)


def zscore(x: torch.Tensor, floor: float) -> torch.Tensor:
    """Standardise within a candidate set; `floor` keeps near-ties from being blown up."""
    return (x - x.mean()) / max(float(x.std(unbiased=False)), floor)


# Q values move on a ~0.1 scale, utility on ±2: they are mixed as z-scores.
Q_STD_FLOOR = 1e-4
U_STD_FLOOR = 0.05


def rerank_scores(q: torch.Tensor, u: torch.Tensor, lam: float) -> torch.Tensor:
    """score_k = z(Q_prior)_k + λ · z(u_WM)_k.  λ = 0 ranks exactly as Q does."""
    if lam <= 0:
        return q
    return zscore(q, Q_STD_FLOOR) + lam * zscore(u, U_STD_FLOOR)
