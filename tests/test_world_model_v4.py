"""World model V4 (察言观色) — DESIGN-phase4 Phase 1 and 2."""
import json
import zlib
from unittest.mock import patch

import numpy as np
import pytest
import torch

from yicenet import world_model_v4 as wm4
from yicenet.config import LEARNING_DEFAULTS
from yicenet.world_model_v4 import (
    CONTEXT_DIM, FEATURE_FIELDS, OUTCOMES, WorldModelV4, base_rates, build_dataset,
    evaluate, rerank_lambda, rerank_scores, sample_features, sample_weights, time_split,
    train_world_model,
)


def fake_encoder(texts):
    """Deterministic stand-in for the frozen prior: a hash of the text → h0, 本卦."""
    seeds = [zlib.crc32(t.encode("utf-8")) for t in texts]
    h = torch.stack([torch.tensor(np.random.default_rng(k).normal(size=256), dtype=torch.float32)
                     for k in seeds])
    return torch.nn.functional.normalize(h, dim=-1), torch.tensor([k % 64 for k in seeds])


def full_sample(i, **kw):
    s = {"user_text": f"问题 {i}", "producer": "claude-code", "timestamp": 1_000_000.0 + i,
         "turn_id": i % 7, "base_hexagram": i % 64, "chosen_hexagram": (i * 7) % 64,
         "candidates": [(i * 7 + k) % 64 for k in range(8)], "chosen_prob": 1.0,
         "context_pre": {"tok_tool_count": 0.3, "tok_is_correction": float(i % 2)},
         "next_user_text": "好的", "corrected": False, "completed": True,
         "abandoned": False, "continued": True, "praised": False}
    s.update(kw)
    return s


class TestNoLabelLeakage:
    """Outcomes are what the WM predicts: they must never be an input."""

    def test_feature_fields_exclude_labels(self):
        labels = set(OUTCOMES) | {"praised", "satisfaction", "next_user_text", "token_cost"}
        assert not labels & set(FEATURE_FIELDS)

    def test_outcomes_do_not_change_features(self):
        base = full_sample(3)
        flipped = dict(base, next_user_text="不对，重新来", corrected=True, completed=False,
                       abandoned=True, continued=False, praised=True, satisfaction=-1.0)
        assert sample_features(base) == sample_features(flipped)
        a, b = build_dataset([base], fake_encoder), build_dataset([flipped], fake_encoder)
        for x, y in zip(a.inputs(), b.inputs()):
            assert torch.equal(x, y)
        assert not torch.equal(a.y, b.y)

    def test_context_is_previous_turn_plus_question(self):
        ctx, has = wm4.context_features({"tok_tool_count": 0.5}, 3, "不对，重新来")
        assert has and len(ctx) == CONTEXT_DIM
        assert ctx[wm4.CONTEXT_KEYS.index("tok_tool_count")] == 0.5
        assert ctx[len(wm4.CONTEXT_KEYS) + 1] == 1.0  # the question itself is a correction
        _, has = wm4.context_features(None, 0, "hi")
        assert not has

    def test_legacy_row_takes_base_from_evolution(self):
        f = build_dataset([{"user_text": "q", "hexagram_evolution": [12], "timestamp": 1.0}], fake_encoder)
        assert f.base.tolist() == [12] and f.has_chosen.tolist() == [0.0] and f.has_ctx.tolist() == [0.0]


class TestWeightsAndSplit:
    def test_time_split_tests_on_newest(self):
        ts = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
        tr, te = time_split(ts, 0.4)
        assert sorted(ts[te].tolist()) == [4.0, 5.0] and sorted(ts[tr].tolist()) == [1.0, 2.0, 3.0]

    def test_producer_balance_and_ips(self):
        now = 2_000_000.0
        rows = [{"producer": "hermes", "timestamp": now}] * 100 + [{"producer": "kimi", "timestamp": now}]
        w = sample_weights(rows, now)
        assert w[:100].sum() / w[100] == pytest.approx(10.0)  # n^-½: 100× samples → 10× weight
        w = sample_weights([{"timestamp": now, "chosen_prob": 0.25}, {"timestamp": now, "chosen_prob": 1.0}], now)
        assert w[0] / w[1] == pytest.approx(4.0)

    def test_older_samples_weigh_less(self):
        now = 100 * 86400.0
        w = sample_weights([{"timestamp": now}, {"timestamp": now - 60 * 86400}], now)
        assert w[0] > w[1] > 0


class TestLearnsAndGates:
    def _data(self, n=600, full=True):
        rng = np.random.default_rng(0)
        rows = []
        for i in range(n):
            chosen = int(rng.integers(64))
            p_correct = 0.8 if chosen < 32 else 0.1  # planted: the 之卦 matters
            corrected = bool(rng.random() < p_correct)
            kw = {"chosen_hexagram": chosen, "corrected": corrected, "completed": not corrected,
                  "abandoned": False, "continued": True}
            if not full:
                kw.update(chosen_hexagram=None, candidates=None, context_pre=None)
            rows.append(full_sample(i, **kw))
        return build_dataset(rows, fake_encoder, now=1_000_000.0 + n)

    def test_beats_base_rate_on_held_out(self):
        data = self._data()
        tr, te = time_split(data.ts, 0.2)
        train, test = data.subset(tr), data.subset(te)
        rates = base_rates(train)
        m = evaluate(train_world_model(train), test, rates, min_full=30)
        base = evaluate(None, test, rates, min_full=30)
        assert m["log_loss"] < base["log_loss"] and m["skill"] > 0
        assert base["skill"] == pytest.approx(0.0)
        assert m["rerank_skill"] > 0 and m["per_outcome"]["corrected"]["auc"] > 0.7

    def test_legacy_only_never_gets_rerank_skill(self):
        data = self._data(full=False)
        tr, te = time_split(data.ts, 0.2)
        m = evaluate(train_world_model(data.subset(tr)), data.subset(te), base_rates(data.subset(tr)))
        assert m["n_full"] == 0 and m["rerank_skill"] == 0.0

    def test_save_load_round_trip(self, tmp_path):
        data = self._data(n=50)
        wm = WorldModelV4()
        wm.meta = {"skill": 0.1}
        wm.save(str(tmp_path / "wm.pt"))
        wm2 = WorldModelV4.load(str(tmp_path / "wm.pt"))
        assert wm2.meta == {"skill": 0.1}
        assert torch.allclose(wm.proba(*data.inputs()), wm2.proba(*data.inputs()))
        assert sum(p.numel() for p in wm.parameters()) < 25_000


class TestRerank:
    def test_lambda_gated_by_skill(self):
        assert rerank_lambda(None, 0.3) == 0.0
        assert rerank_lambda({"rerank_skill": 0.0}, 0.3) == 0.0
        assert rerank_lambda({"rerank_skill": 0.5}, 0.3) == pytest.approx(0.15)
        assert rerank_lambda({"rerank_skill": 4.0}, 0.3) == pytest.approx(0.3)

    def test_zero_lambda_keeps_prior_order(self):
        q = torch.tensor([0.1, 0.3, -0.2, 0.05])
        u = torch.tensor([2.0, -2.0, 1.0, 0.0])
        assert torch.equal(rerank_scores(q, u, 0.0), q)
        assert int(rerank_scores(q, u, 5.0).argmax()) == 0  # a strong WM can move the choice
        # a flat utility cannot move it, however large λ
        assert int(rerank_scores(q, torch.full((4,), 0.7), 5.0).argmax()) == 1


@pytest.fixture
def engine(tmp_path):
    from yicenet.config import YiCeNetConfig
    from yicenet.model import YiCeNet
    from yicenet.yicenet_engine import YiCeNetEngine
    torch.manual_seed(0)
    ckpt = tmp_path / "prior.pt"
    YiCeNet(YiCeNetConfig()).save_pretrained(str(ckpt))
    with patch("yicenet.config.yicenet_checkpoint_dir", return_value=tmp_path), \
         patch("yicenet.config.get_learning_config", return_value=dict(LEARNING_DEFAULTS)), \
         patch("yicenet.yicenet_engine._ensure_vocab"):
        yield YiCeNetEngine(checkpoint=str(ckpt), device="cpu", project_root=str(tmp_path)), tmp_path


class TestEngineRerank:
    CTX = {"context_pre": {"tok_tool_count": 0.2}, "turn_id": 2}

    def test_without_proven_wm_the_prior_decides(self, engine):
        eng, tmp = engine
        plain = eng.predict("重构登录模块", deterministic=True)
        d = plain["decision"]
        assert d["chosen_hexagram"] == d["candidates"][int(np.argmax(d["candidate_q"]))]
        assert d["chosen_prob"] == 1.0 and d["wm_lambda"] == 0.0

        # An unproven WM (re-rank skill 0) is loaded but gets λ = 0: identical choice.
        WorldModelV4().save(str(tmp / "wm.pt"))
        (tmp / "registry.json").write_text(json.dumps({"world_model": {"active": {
            "version": "wm-x", "path": "wm.pt", "rerank_skill": 0.0}}}))
        gated = eng.predict("重构登录模块", deterministic=True, wm_context=self.CTX)
        assert eng._wm is not None
        for k in ("selected_hexagram_id", "action_id", "candidates"):
            assert gated[k] == plain[k]
        assert gated["decision"]["wm_lambda"] == 0.0

    def test_proven_wm_reranks_candidates(self, engine):
        eng, tmp = engine
        plain = eng.predict("重构登录模块", deterministic=True)
        cands, q = plain["decision"]["candidates"], torch.tensor(plain["decision"]["candidate_q"])
        target = cands[int(q.argsort()[0])]  # the prior's worst candidate

        class StubWM:
            def utility(self, h, base, chosen, *rest, weights=None):
                return (chosen == target).float()

        WorldModelV4().save(str(tmp / "wm.pt"))
        (tmp / "registry.json").write_text(json.dumps({"world_model": {"active": {
            "version": "wm-x", "path": "wm.pt", "rerank_skill": 1.0}}}))
        eng.predict("warm", deterministic=True, wm_context=self.CTX)  # loads the registry
        eng._wm, eng._wm_lambda = StubWM(), 50.0
        r = eng.predict("重构登录模块", deterministic=True, wm_context=self.CTX)
        assert r["selected_hexagram_id"] == target and r["decision"]["chosen_hexagram"] == target
        assert r["decision"]["candidates"] == cands  # only re-ranks the 本卦's candidates
        from yicenet.yicenet_engine import ACTION_NAMES
        with torch.no_grad():
            a, _ = eng._model.decode_action(torch.tensor([target]))
        assert r["action_name"] == ACTION_NAMES[int(a)]


class TestDecisionReachesTrajectory:
    def test_pre_decision_rides_to_buffer(self, tmp_path):
        from yicenet import flywheel
        from yicenet.hook_engine import HookOrchestrator
        from yicenet.memory_bank import MemoryBank
        from yicenet.tools.claude_hook import ClaudeCodeAdapter

        bank = MemoryBank()
        adapter = ClaudeCodeAdapter(process_model="daemon")
        sid = adapter.session_id({"session_id": "d1"})
        decision = {"base_hexagram": 5, "chosen_hexagram": 12, "candidates": list(range(8)),
                    "candidate_q": [0.1] * 8, "chosen_prob": 1.0, "action": "wait_poll",
                    "context_pre": {"tok_tool_count": 0.4}, "wm_lambda": 0.0}
        bank.store_turn(sid, 3, np.zeros(384, dtype=np.float32), 5, summary="帮我重构登录模块",
                        metadata={"response_snippet": "已重构", **decision})
        submitted = []
        with patch("yicenet.memory_bank.get_memory_bank", return_value=bank), \
             patch("yicenet.flywheel.submit_trajectory", side_effect=submitted.append):
            HookOrchestrator(adapter).before_prediction({"session_id": "d1", "prompt": "好的，继续"})
        (t,) = submitted
        assert t["version"] == 2 and t["turn_id"] == 3
        for k, v in decision.items():
            assert t[k] == v

        with patch.object(flywheel, "yicenet_data_dir", return_value=tmp_path):
            flywheel.submit_trajectory(t)
        row = json.loads((tmp_path / "flywheel_buffer.jsonl").read_text(encoding="utf-8"))
        assert row["chosen_hexagram"] == 12 and row["context_pre"] == {"tok_tool_count": 0.4}
        assert row["turn_id"] == 3 and row["completed"] is True

    def test_adapter_records_decision_and_previous_context(self):
        from yicenet.memory_bank import MemoryBank
        from yicenet.tools.hooks_adapter import HooksAdapter

        bank = MemoryBank()
        bank.store_turn("s", 0, np.zeros(384, dtype=np.float32), 1, metadata={"context_vector": {"tok_has_code": 1.0}})
        bank.store_turn("s", 1, np.zeros(384, dtype=np.float32), 2)
        with patch("yicenet.memory_bank.get_memory_bank", return_value=bank):
            prev = HooksAdapter._previous_context("s", 1)
            HooksAdapter._record_decision("s", 1, {"decision": {"chosen_hexagram": 9}},
                                          {"context_pre": prev, "turn_id": 1})
        assert prev == {"tok_has_code": 1.0}
        assert bank.get_turn("s", 1).metadata == {"chosen_hexagram": 9, "context_pre": {"tok_has_code": 1.0}}
