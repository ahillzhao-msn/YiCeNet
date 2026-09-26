"""Slow 卦理 update channel — DESIGN-phase4 Phase 3 acceptance.

A universal pattern (holds in every context) is promoted; a context-specific one
(good in one slice, bad in another) is rejected.
"""
import copy
import json

import numpy as np
import pytest
import torch

from yicenet.config import LEARNING_DEFAULTS
from yicenet.prior_update import (
    CycleContext, Items, choice_kl, gate, prior_q, propose, q_temperature, run_cycle,
)

GOOD = torch.arange(0, 64, 8)  # the planted auspicious 之卦


def make_items(n, seed, frac_a=0.7, t0=0.0):
    g = torch.Generator().manual_seed(seed)
    cands = torch.stack([torch.randperm(64, generator=g)[:8] for _ in range(n)])
    producers = ["A" if r < frac_a else "B" for r in torch.rand(n, generator=g).tolist()]
    ctx = torch.tensor([[1.0 if p == "A" else 0.0] for p in producers])
    return Items(torch.zeros(n, 4), cands[:, 0].clone(), cands, ctx, torch.ones(n),
                 producers, t0 + np.arange(n, dtype=float))


def universal(items, chosen, ctx, has_ctx):
    return torch.isin(chosen, GOOD).float()


def context_specific(items, chosen, ctx, has_ctx):
    """Good for platform A's customers, bad for platform B's."""
    return torch.isin(chosen, GOOD).float() * (2 * ctx[:, 0] - 1)


@pytest.fixture(scope="module")
def prior():
    from yicenet.config import YiCeNetConfig
    from yicenet.model import YiCeNet
    torch.manual_seed(0)
    return YiCeNet(YiCeNetConfig()).eval()


def pool(items):
    """A representative context pool (~70% platform A)."""
    return items.ctx[::10], items.has_ctx[::10]


def test_universal_pattern_passes_the_gates(prior):
    train, held = make_items(600, 1), make_items(300, 2, t0=1000)
    head, result = propose(prior, train, held, universal, *pool(train), LEARNING_DEFAULTS)
    assert result.passed, result.reason
    assert result.ci[0] > 0 and result.kl <= LEARNING_DEFAULTS["prior_trust_kl"] + 1e-9
    with torch.no_grad():
        q_new = prior_q(head, prior.hexagram_embed, held.cands)
    rows = torch.arange(len(held))
    picked = held.cands[rows, q_new.argmax(-1)]
    has_good = torch.isin(held.cands, GOOD).any(-1)
    old = held.cands[rows, prior_q(prior.value_net, prior.hexagram_embed, held.cands).argmax(-1)]
    assert torch.isin(picked[has_good], GOOD).float().mean() > torch.isin(old[has_good], GOOD).float().mean()


def test_context_specific_pattern_is_rejected(prior):
    train, held = make_items(600, 1), make_items(300, 2, t0=1000)
    _, result = propose(prior, train, held, context_specific, *pool(train), LEARNING_DEFAULTS)
    assert not result.passed
    assert result.gain > 0  # it even helps on average (A dominates) …
    assert "not universal" in result.reason and "platform=B" in result.reason  # … but hurts B


def test_trust_region_bounds_the_step(prior):
    train, held = make_items(600, 1), make_items(300, 2, t0=1000)
    tight = dict(LEARNING_DEFAULTS, prior_trust_kl=1e-4)
    head, result = propose(prior, train, held, universal, *pool(train), tight)
    with torch.no_grad():
        q_old = prior_q(prior.value_net, prior.hexagram_embed, train.cands)
        kl = choice_kl(prior_q(head, prior.hexagram_embed, train.cands), q_old, q_temperature(q_old))
    assert kl <= 1e-4 + 1e-9 and result.step < 1.0


def test_no_change_is_not_confident():
    r = gate(torch.zeros(100), {"platform": ["A"] * 100}, 0.005)
    assert not r.passed and "not confident" in r.reason


def test_cycle_streak_shadow_then_promotion(prior, tmp_path):
    """K passing cycles (one per cadence window) → shadow → promoted after the shadow period."""
    cfg = dict(LEARNING_DEFAULTS, prior_stable_cycles=2, shadow_min_days=7, shadow_min_samples=50)
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(json.dumps({"active": {"version": "v18", "path": "yicenet_v18.pt"}}))
    train, held = make_items(600, 1), make_items(300, 2, t0=1000)
    cc = CycleContext(prior=prior, util=universal, train=train, held_out=held,
                      ctx_pool=pool(train)[0], has_pool=pool(train)[1])
    versions = iter(["v43"])
    day = 86400.0
    t = 100 * day

    assert run_cycle(cc, reg_path, tmp_path, cfg, lambda: next(versions), now=t).startswith("passed 1/2")
    assert run_cycle(cc, reg_path, tmp_path, cfg, lambda: next(versions), now=t + day).startswith("waiting")
    assert run_cycle(cc, reg_path, tmp_path, cfg, lambda: next(versions), now=t + 7 * day) == "shadow started"
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    assert reg["shadow"] and (tmp_path / reg["shadow"]["path"]).exists()
    assert reg["active"]["version"] == "v18"  # the shadow only watches

    # Shadow period: the engine logged [chosen, shadow_chosen] for real questions.
    from yicenet.prior_update import load_value_head
    head = load_value_head(str(tmp_path / reg["shadow"]["path"]), prior)
    live = make_items(120, 3, t0=5000)
    rows = torch.arange(len(live))
    with torch.no_grad():
        chosen = live.cands[rows, prior_q(prior.value_net, prior.hexagram_embed, live.cands).argmax(-1)]
        shadow = live.cands[rows, prior_q(head, prior.hexagram_embed, live.cands).argmax(-1)]
    cc.shadow_items, cc.shadow_logged = live, torch.stack([chosen, shadow], dim=1)

    assert run_cycle(cc, reg_path, tmp_path, cfg, lambda: next(versions), now=t + 10 * day).startswith("shadow running")
    assert run_cycle(cc, reg_path, tmp_path, cfg, lambda: next(versions), now=t + 15 * day) == "promoted v43"
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    assert reg["active"]["version"] == "v43" and reg["fallback"]["version"] == "v18"
    assert reg["shadow"] is None and (tmp_path / "yicenet_v43.pt").exists()
    saved = torch.load(tmp_path / "yicenet_v43.pt", weights_only=False)["model_state_dict"]
    for k, v in head.state_dict().items():
        assert torch.equal(saved[f"value_net.{k}"], v)
    enc = {k: v for k, v in saved.items() if k.startswith("encoder.")}
    assert all(torch.equal(v, prior.state_dict()[k]) for k, v in enc.items())  # encoder frozen


def test_context_specific_shadow_is_not_promoted(prior, tmp_path):
    cfg = dict(LEARNING_DEFAULTS, shadow_min_days=0, shadow_min_samples=10)
    reg_path = tmp_path / "registry.json"
    live = make_items(200, 4)
    rows = torch.arange(len(live))
    chosen = live.cands[:, 1]
    shadow = torch.where(torch.isin(live.cands, GOOD).any(-1),
                         live.cands[rows, torch.isin(live.cands, GOOD).float().argmax(-1)], chosen)
    (tmp_path / "s.pt").write_bytes(b"")
    reg_path.write_text(json.dumps({"active": {"version": "v18"},
                                    "shadow": {"version": "shadow-x", "path": "s.pt", "started": 0.0}}))
    cc = CycleContext(prior=prior, util=context_specific, train=live, held_out=live,
                      ctx_pool=live.ctx[:16], has_pool=live.has_ctx[:16],
                      shadow_items=live, shadow_logged=torch.stack([chosen, shadow], dim=1))
    status = run_cycle(cc, reg_path, tmp_path, cfg, lambda: "v99", now=10 * 86400.0)
    assert status.startswith("shadow rejected: not universal")
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    assert reg["shadow"] is None and reg["active"]["version"] == "v18"
