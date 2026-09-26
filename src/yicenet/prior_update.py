"""
Slow 卦理 update channel (DESIGN-phase4.md §3.5).

The prior changes only when every gate passes:

  1. Distillation, not raw RL: the target is the validated WM's utility with the
     context marginalised, E_ctx[u(question, 本卦, 之卦)] — what stays true once
     察言观色 is averaged out.
  2. Universal: the candidate must not regress on any slice (platform, time
     window, question type = upper trigram of the 本卦), judged with each
     question's own context.
  3. High-confidence: the bootstrap CI of the held-out gain excludes 0, and the
     gates pass K cycles in a row.
  4. Small steps: trust region KL(new ‖ old) ≤ ε on the candidate-choice
     distribution; only the value head learns (encoder, router, embeddings frozen).
  5. Shadow period: the candidate only logs what it would choose; it is promoted
     once the shadow-period data confirms the gain.  `fallback` stays the rollback.
  6. Cadence: at most one gate attempt per `prior_update_days`.

The gate logic takes the utility as a function, so it can be checked on synthetic
patterns (tests/test_prior_update.py) without a trained WM.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from yicenet.world_model_v4 import bootstrap_ci

# u(items, chosen (N,), ctx (N, C), has_ctx (N,)) -> (N,)
UtilityFn = Callable[["Items", torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class Items:
    """Questions with the frozen prior's candidate sets and question-time context."""
    h: torch.Tensor          # (N, 256) question state (WM input)
    base: torch.Tensor       # (N,) 本卦
    cands: torch.Tensor      # (N, K) candidate 之卦
    ctx: torch.Tensor        # (N, C) own context
    has_ctx: torch.Tensor    # (N,)
    producers: list
    ts: np.ndarray

    def __len__(self) -> int:
        return len(self.ts)

    def subset(self, idx) -> "Items":
        idx = np.asarray(idx, dtype=np.int64)
        t = torch.as_tensor(idx)
        return Items(self.h[t], self.base[t], self.cands[t], self.ctx[t], self.has_ctx[t],
                     [self.producers[i] for i in idx], self.ts[idx])

    def slices(self) -> dict[str, list]:
        """Slice labels per item: platform, time window (older/newer half), question type."""
        order = np.argsort(self.ts, kind="stable")
        half = np.empty(len(self.ts), dtype=object)
        half[order[: len(order) // 2]] = "older"
        half[order[len(order) // 2:]] = "newer"
        return {
            "platform": list(self.producers),
            "time": list(half),
            "question_type": [f"upper{int(b) >> 3}" for b in self.base.tolist()],
        }


@dataclass
class GateResult:
    passed: bool
    reason: str
    gain: float = 0.0
    ci: tuple = (0.0, 0.0)
    slices: dict = field(default_factory=dict)
    kl: float = 0.0
    step: float = 0.0
    n: int = 0

    def to_dict(self) -> dict:
        return {"passed": self.passed, "reason": self.reason, "gain": round(self.gain, 4),
                "ci": [round(c, 4) for c in self.ci], "kl": round(self.kl, 5),
                "step": self.step, "n": self.n, "slices": self.slices}


# ── Prior Q over candidates ───────────────────────────────────────────────────

def prior_q(value_net, embed, cands: torch.Tensor) -> torch.Tensor:
    """(N, K) Q-values of the candidates under a value head."""
    return value_net(embed(cands)).squeeze(-1)


def q_temperature(q: torch.Tensor) -> float:
    """Scale of the old prior's Q spread: turns Q into a choice distribution."""
    return max(float(q.std(dim=-1, unbiased=False).mean()), 1e-3)


def choice_kl(q_new: torch.Tensor, q_old: torch.Tensor, temp: float) -> float:
    """Mean KL(new ‖ old) between softmax(Q/temp) candidate-choice distributions."""
    p_new = F.log_softmax(q_new / temp, dim=-1)
    p_old = F.log_softmax(q_old / temp, dim=-1)
    return float((p_new.exp() * (p_new - p_old)).sum(-1).mean())


# ── 1. Distillation ───────────────────────────────────────────────────────────

def marginal_utility(items: Items, util: UtilityFn, ctx_pool: torch.Tensor,
                     has_pool: torch.Tensor) -> torch.Tensor:
    """(N, K) E_ctx[u(question, 本卦, candidate)] over a pool of question-time contexts."""
    n, k = items.cands.shape
    total = torch.zeros(n, k)
    for m in range(len(ctx_pool)):
        ctx = ctx_pool[m].expand(n, -1)
        has = has_pool[m].expand(n)
        for j in range(k):
            total[:, j] += util(items, items.cands[:, j], ctx, has)
    return total / max(len(ctx_pool), 1)


def distill_value_head(prior, items: Items, target_u: torch.Tensor, trust_kl: float,
                       steps: int = 300, lr: float = 1e-3, seed: int = 0):
    """Fit a copy of the value head to the marginal-utility ranking, inside the trust region.

    Listwise: softmax(Q_new/temp) is pulled towards softmax(z(target)).  The step
    from the old head is then shrunk (1, ½, ¼, …) until KL(new ‖ old) ≤ trust_kl.
    Returns (new value head, kl, step); step 0 means no admissible change.
    """
    torch.manual_seed(seed)
    old = prior.value_net
    with torch.no_grad():
        embeds = prior.hexagram_embed(items.cands).detach()
        q_old = old(embeds).squeeze(-1)
    temp = q_temperature(q_old)
    z = (target_u - target_u.mean(-1, keepdim=True)) / target_u.std(-1, unbiased=False, keepdim=True).clamp(min=0.05)
    target_p = F.softmax(z, dim=-1)

    new = copy.deepcopy(old)
    new.train()
    opt = torch.optim.Adam(new.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        logp = F.log_softmax(new(embeds).squeeze(-1) / temp, dim=-1)
        (-(target_p * logp).sum(-1).mean()).backward()
        opt.step()
    new.eval()

    old_params = [p.detach().clone() for p in old.parameters()]
    new_params = [p.detach().clone() for p in new.parameters()]
    step = 1.0
    while step >= 1 / 64:
        with torch.no_grad():
            for p, a, b in zip(new.parameters(), old_params, new_params):
                p.copy_(a + step * (b - a))
            kl = choice_kl(new(embeds).squeeze(-1), q_old, temp)
        if kl <= trust_kl:
            return new, kl, step
        step /= 2
    return copy.deepcopy(old).eval(), 0.0, 0.0


# ── 2–3. Universality and confidence ──────────────────────────────────────────

def choice_gain(items: Items, util: UtilityFn, q_old: torch.Tensor, q_new: torch.Tensor) -> torch.Tensor:
    """(N,) u(new prior's 之卦) − u(old prior's 之卦), each judged in its own context."""
    rows = torch.arange(len(items))
    old_c = items.cands[rows, q_old.argmax(-1)]
    new_c = items.cands[rows, q_new.argmax(-1)]
    return util(items, new_c, items.ctx, items.has_ctx) - util(items, old_c, items.ctx, items.has_ctx)


def gate(gains: torch.Tensor, slices: dict, tolerance: float, min_slice: int = 5,
         n_boot: int = 1000) -> GateResult:
    """Universal (no slice regresses) and high-confidence (bootstrap CI > 0)."""
    g = gains.detach().numpy().astype(np.float64)
    if len(g) == 0:
        return GateResult(False, "no held-out questions")
    lo, hi = bootstrap_ci(g, n_boot)
    report, regressed = {}, []
    for name, labels in slices.items():
        labels = np.asarray(labels, dtype=object)
        for value in sorted(set(labels.tolist()), key=str):
            mask = labels == value
            n = int(mask.sum())
            mean = float(g[mask].mean())
            report[f"{name}={value}"] = {"n": n, "gain": round(mean, 4)}
            if n >= min_slice and mean < -tolerance:
                regressed.append(f"{name}={value}")
    result = GateResult(False, "", gain=float(g.mean()), ci=(lo, hi), slices=report, n=len(g))
    if regressed:
        result.reason = "not universal: regresses on " + ", ".join(regressed)
    elif lo <= 0:
        result.reason = f"not confident: CI [{lo:.4f}, {hi:.4f}] includes 0"
    else:
        result.passed, result.reason = True, "universal and confident"
    return result


def propose(prior, train: Items, held_out: Items, util: UtilityFn, ctx_pool: torch.Tensor,
            has_pool: torch.Tensor, cfg: dict):
    """One gate attempt: distil a candidate head, then judge it on held-out questions.

    Returns (candidate value head, GateResult).
    """
    target = marginal_utility(train, util, ctx_pool, has_pool)
    head, kl, step = distill_value_head(prior, train, target, float(cfg["prior_trust_kl"]))
    if step == 0:
        return head, GateResult(False, "no step fits the trust region")
    with torch.no_grad():
        q_old = prior_q(prior.value_net, prior.hexagram_embed, held_out.cands)
        q_new = prior_q(head, prior.hexagram_embed, held_out.cands)
    result = gate(choice_gain(held_out, util, q_old, q_new), held_out.slices(),
                  float(cfg["slice_tolerance"]))
    result.kl, result.step = kl, step
    return head, result


# ── Value-head files (shadow) ─────────────────────────────────────────────────

def save_value_head(head, path: str, meta: dict) -> None:
    torch.save({"value_net_state_dict": head.state_dict(), "meta": meta}, path)


def load_value_head(path: str, prior):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    head = copy.deepcopy(prior.value_net).cpu()
    head.load_state_dict(saved["value_net_state_dict"])
    return head.eval()


# ── 3–6. The cycle: streak → shadow → promotion ───────────────────────────────

@dataclass
class CycleContext:
    """Everything one cycle needs; built by the flywheel, synthetic in tests."""
    prior: object                  # active YiCeNet (value_net, hexagram_embed)
    util: UtilityFn                # validated WM utility
    train: Items
    held_out: Items
    ctx_pool: torch.Tensor
    has_pool: torch.Tensor
    shadow_items: Optional[Items] = None       # questions answered since the shadow started
    shadow_logged: Optional[torch.Tensor] = None  # (M, 2): [chosen, shadow_chosen] 之卦


def run_cycle(cc: CycleContext, registry_path: Path, checkpoint_dir: Path, cfg: dict,
              next_version: Callable[[], str], now: Optional[float] = None,
              log: Callable[[str], None] = print) -> str:
    """Advance the slow channel by at most one step; returns a short status."""
    now = now or time.time()
    reg = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    pu = reg.setdefault("prior_update", {"streak": 0, "last_attempt": 0.0, "history": []})

    def save():
        pu["history"] = pu["history"][-20:]
        registry_path.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")

    shadow = reg.get("shadow")
    if shadow:
        status = _judge_shadow(cc, reg, shadow, checkpoint_dir, cfg, next_version, now, log)
        pu["history"].append({"time": now, "event": status})
        save()
        return status

    if now - float(pu.get("last_attempt", 0.0)) < float(cfg["prior_update_days"]) * 86400:
        return "waiting: at most one 卦理 attempt per %s days" % cfg["prior_update_days"]
    pu["last_attempt"] = now

    head, result = propose(cc.prior, cc.train, cc.held_out, cc.util, cc.ctx_pool, cc.has_pool, cfg)
    log(f"    卦理 gate: {result.reason} (gain={result.gain:+.4f}, "
        f"CI=[{result.ci[0]:+.4f}, {result.ci[1]:+.4f}], KL={result.kl:.4f}, step={result.step})")
    if not result.passed:
        pu["streak"] = 0
        pu["history"].append({"time": now, "event": "rejected", **result.to_dict()})
        save()
        return "rejected: " + result.reason

    pu["streak"] = int(pu.get("streak", 0)) + 1
    k = int(cfg["prior_stable_cycles"])
    if pu["streak"] < k:
        pu["history"].append({"time": now, "event": f"passed {pu['streak']}/{k}", **result.to_dict()})
        save()
        return f"passed {pu['streak']}/{k}: needs {k} cycles in a row"

    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    path = checkpoint_dir / f"prior_shadow_{stamp}.pt"
    save_value_head(head, str(path), {"gate": result.to_dict()})
    reg["shadow"] = {"version": f"shadow-{stamp}", "path": path.name, "started": now,
                     "base": (reg.get("active") or {}).get("version"), "gate": result.to_dict()}
    pu["streak"] = 0
    pu["history"].append({"time": now, "event": "shadow started", **result.to_dict()})
    save()
    return "shadow started"


def _judge_shadow(cc: CycleContext, reg: dict, shadow: dict, checkpoint_dir: Path, cfg: dict,
                  next_version: Callable[[], str], now: float, log) -> str:
    age_days = (now - float(shadow.get("started", now))) / 86400
    n = 0 if cc.shadow_items is None else len(cc.shadow_items)
    if age_days < float(cfg["shadow_min_days"]) or n < int(cfg["shadow_min_samples"]):
        return f"shadow running: {age_days:.1f}/{cfg['shadow_min_days']} days, {n}/{cfg['shadow_min_samples']} samples"

    items, logged = cc.shadow_items, cc.shadow_logged
    gains = (cc.util(items, logged[:, 1], items.ctx, items.has_ctx)
             - cc.util(items, logged[:, 0], items.ctx, items.has_ctx))
    result = gate(gains, items.slices(), float(cfg["slice_tolerance"]))
    log(f"    shadow {shadow['version']}: {result.reason} (gain={result.gain:+.4f}, "
        f"CI=[{result.ci[0]:+.4f}, {result.ci[1]:+.4f}], n={result.n})")
    reg["shadow"] = None
    if not result.passed:
        return "shadow rejected: " + result.reason

    version = next_version()
    prior = copy.deepcopy(cc.prior)
    prior.value_net.load_state_dict(load_value_head(str(checkpoint_dir / shadow["path"]), prior).state_dict())
    path = checkpoint_dir / f"yicenet_{version}.pt"
    prior.save_pretrained(str(path))
    old_active = reg.get("active")
    if old_active:
        reg.setdefault("history", []).append(dict(old_active))
    reg["fallback"] = old_active
    reg["active"] = {"version": version, "path": path.name,
                     "created": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
                     "notes": f"卦理 update from {shadow['version']} (distilled from WM)",
                     "gate": shadow.get("gate"), "shadow_gate": result.to_dict()}
    return f"promoted {version}"
