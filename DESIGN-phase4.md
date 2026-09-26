# Phase 4 — Two-speed learning: 察言观色 (world model) and 卦理 (prior)

> Goal: make the flywheel learn the right thing at the right speed.
> - The **world model (WM)** learns 察言观色: how the customer will react to a given
>   answer in a given situation. It updates every flywheel run.
> - The **卦理 prior** (the YiCeNet model) evolves only slowly, and only on universal,
>   high-confidence evidence distilled from a validated WM.
>
> Status: design agreed on 2026-09-25; Phases 1–3 implemented on 2026-09-25 (§7 records
> the implementation decisions and the first result on real data). Each phase is
> shippable on its own.

---

## 1. Principles

A diviner (卦师) always casts and reads a hexagram the same way; the theory is the I Ching.
The reading of the same hexagram differs from case to case because of 察言观色, which is
matching the situation and the person. That part is experience.

| | 卦理: the prior | 察言观色: the world model |
|---|---|---|
| What | question → 本卦 → 8 变卦 candidates → prior Q → 之卦 + action | P(customer reaction \| question, 本卦, candidate 之卦, context) |
| Nature | distilled human prior; *can* evolve | empirical pattern learning (not a log) |
| Update speed | slow: rare, gated | fast: every flywheel run |
| Update bar | universal (holds in every slice) **and** high-confidence (CI excludes 0, stable over cycles) | beats the base rate on held-out outcome prediction |

Other agreed principles:

- **本卦 = the question, 之卦 = the answer.** The session chain records 本卦, which follows
  the customer's line of thought and not the diviner's. Trajectories carry the question
  text (`user_text`) and the customer's reaction (`next_user_text`).
- **The hexagram is a direction, not advice.** 殊途同归: when paths converge, the value lies
  in choosing and committing, not in computing the one precise answer. The WM must never
  turn the hexagram into step-by-step instructions. It only re-ranks among the candidates
  the 本卦 produces.
- **Session memory is working memory; the flywheel is what persists.** Sessions are kept
  under the retention policy (48 h, 200 turns). With LOOM, `memory.session_manager: external`.

## 2. Current state: what is wrong (as of commit 6642cd8)

1. **The WM is not used at inference.** `yicenet_engine.py` and `model.py` never reference
   it. Its only role is to score RL rewards in `flywheel.py`.
2. **Experience is written into the prior, hastily.** Step 4 (`_rl_fine_tune_v5`,
   `flywheel.py` ~L655) fine-tunes `router`, `value_net` and `encoder.state_proj` every 6 h
   on ~100 samples against a saturating reward. No candidate since v18 has been validated.
   Until 29930d9, each run also chained on the newest checkpoint, so v19→v41 compounded
   training on placeholder data. Those versions are retired to
   `~/.yicenet/backup/flywheel-cleanup-20260925/`.
3. **The WM input leaks the labels.** `_flywheel_entry_to_context_vector` (`flywheel.py`
   ~L456) builds a mostly-constant "context" whose last three dims are `corrected`,
   `praised` and `abandoned`, which are the outcomes it is supposed to predict. The loss
   drops to ~1e-4, every reward is ~0.95, and the evaluation saturates at a 100% win rate,
   so nothing is ever promoted.
4. **The real context is thrown away.** `DaemonContextCollector.build_vector`
   (`hook_engine/collector/daemon.py`) produces a genuine 27-dim context per turn: tools,
   response, user rhythm, correction/praise, hexagram confidence. It is stored in
   MemoryBank metadata (`context_vector`) but never reaches the flywheel.
5. **The WM target is synthetic.** `project_to_hexagram_space` (`rl_train.py`) maps outcome
   booleans onto hand-picked hexagram clusters (继续→乾需泰…) and trains a 64-dim KL
   against that. It does not model observed outcomes.
6. **The evaluation cannot discriminate.** Steps 6–7 compare win rates of `reward > 0.5`,
   which is saturated because of item 3.

Already fixed today (for context): real turn counting, reply taken from the Stop payload,
tool outcome, trajectories carrying the question, `extract_question` filtering, cleaned
buffer (3686 → 1085 samples), RL base = registry active, windowless flywheel task.

## 3. Target architecture

### 3.1 Trajectory schema (additions)

Everything the WM needs must be known **at question time** (no outcomes as input):

```jsonc
{
  "user_text": "...",            // the question (本卦 is cast from it; re-encoded in training)
  "next_user_text": "...",       // the customer's reaction
  "base_hexagram": 12,           // 本卦 (already in hexagram_evolution)
  "chosen_hexagram": 44,         // NEW: the 之卦 actually injected
  "candidates": [12, 7, ...],    // NEW: the 8 candidate ids
  "candidate_q": [0.41, ...],    // NEW: prior Q-values (the prior's opinion)
  "chosen_prob": 0.83,           // NEW: selection probability (propensity, for IPS)
  "context_pre": [...27],        // NEW: context known before answering: previous turn's
                                 //      context_vector + session position + chain signals
  "action": "wait_poll",         // NEW
  "outcome": {"corrected": false, "completed": true, "abandoned": false, "continued": true, "praised": false}
}
```

Source: store `chosen_hexagram`, `candidates`, `candidate_q`, `chosen_prob` and `action` in
the TurnRecord metadata at `pre` (`HooksAdapter.predict_for_turn_payload`). Take
`context_pre` from the previous turn's `context_vector` (`get_turn(sid, turn_id-1)`). Pass
all of it through `build_trajectory` → `submit_trajectory`.

### 3.2 World model V4 (察言观色)

- **Input:** frozen-encoder probes(question) ⊕ onehot(本卦) ⊕ onehot(candidate 之卦) ⊕ `context_pre`.
- **Output:** outcome probabilities `[corrected, completed, abandoned, continued]` (sigmoid),
  with loss = BCE against the observed outcome.
  - Time-decayed sample weights (keep the power-law weighting).
  - Per-producer balancing, so one platform does not dominate.
  - Inverse-propensity weighting with `chosen_prob`: only the chosen candidate's outcome is
    ever observed.
- **Utility:** `u = w·p`, for example `completed + continued − corrected − abandoned`,
  with the weights kept in config.
- **Size:** small (~20k params), CPU training in seconds.

### 3.3 Inference (Phase 2)

For each of the 8 candidates, `score_k = Q_prior_k + λ · u_WM(k)`; the 之卦 is chosen from
`score`.
- **Gate:** `λ = λ_max · clamp(skill, 0, 1)`. `skill` is the WM's held-out improvement over
  the base rate (e.g. normalized log-loss gain), stored with the WM checkpoint.
- An unproven WM gets λ = 0, and the prior alone decides.
- **Exploration:** keep the existing temperature sampling and log `chosen_prob`.
- **Constraint:** the WM only re-ranks candidates generated from the 本卦. It never
  generates hexagrams or actions of its own.

### 3.4 Honest evaluation (replaces Steps 6–7)

- **Time-based split:** train on older samples, test on the newest N% (N configurable,
  e.g. 20%).
- **Metrics:** log-loss and AUC per outcome, versus the base-rate predictor and versus the
  current WM.
- **Promote a WM** only if it beats both on held-out data. Record `skill`, metrics and
  sample counts in `registry.json` under a `world_model` section (active / ready /
  fallback, like the prior).

### 3.5 Slow 卦理 update (Phase 3)

The prior may change only when every gate passes:

1. **Distillation, not raw RL.** Target = the validated WM's utility with context
   marginalized, `E_context[u_WM(question, 本卦, 之卦)]`. What remains true once 察言观色 is
   averaged out is a candidate for 卦理.
2. **Universal.** Evaluate per slice: platform (claude-code / kimi-code / hermes), time
   window, question type. The candidate prior must **not regress on any slice**.
3. **High-confidence.** The bootstrap CI of the held-out improvement excludes 0, and the
   gain holds over K consecutive flywheel cycles.
4. **Small steps.** Enforce a trust region `KL(new ‖ old) ≤ ε` and use a low learning rate.
   The encoder (text → 象) stays frozen; the value head and router are the main learners.
5. **Shadow period.** The candidate runs in shadow: it logs what it would choose while the
   active prior still decides. It is promoted after the shadow outcomes confirm it.
   `fallback` remains the rollback.
6. **Cadence.** At most weekly; the gates decide whether an update happens at all.

## 4. Flywheel after Phase 1

```
Step 1  scan sources (extract_question filter)                     [exists]
Step 2  build the training set: trajectories with the §3.1 fields  [new]
Step 3  train WM V4 on the time split                              [new]
Step 4  evaluate WM V4 vs base rate and active WM; promote if better [new]
Step 5  (Phase 3) prior distillation candidate + gates + shadow    [later]
```

The current Step 4 (RL on the prior every run) is **removed** in Phase 1.

## 5. Phases and acceptance criteria

### Phase 1 — data + WM V4 + honest evaluation (stop hasty prior training)
- [x] TurnRecord metadata at `pre`: `chosen_hexagram`, `candidates`, `candidate_q`,
      `chosen_prob`, `action`.
- [x] Trajectories carry the §3.1 fields; `context_pre` comes from the previous turn's
      `context_vector`.
- [x] `WorldModelV4` with BCE outcome heads, IPS and producer weights; no outcome in input.
- [x] Time-split evaluation; `registry.json["world_model"]` holds active / ready / fallback
      with metrics and `skill`.
- [x] Flywheel: remove the RL fine-tune of the prior; train and gate the WM only.
- [x] Legacy samples (no §3.1 fields) train a reduced baseline, "question + 本卦 →
      outcome", which is marked as such.
- **Accept:**
  - With real data, WM V4 held-out log-loss is lower than the base rate.
  - No label leakage: a unit test asserts that the outcome fields never appear in the
    input features.
  - The flywheel run log reports honest metrics.

### Phase 2 — WM re-ranks 之卦 at inference
- [x] The engine loads the active WM; `score = Q_prior + λ·u_WM`, λ gated by `skill`.
- [x] `chosen_prob` is logged; `pre` latency stays within +2 ms.
- **Accept:**
  - With λ = 0, behaviour is identical to today.
  - With a proven WM, the offline replay utility is at least the prior's.

### Phase 3 — slow 卦理 update channel
- [x] Distillation target, slice evaluation, bootstrap CI, K-cycle stability, trust
      region, shadow mode.
- **Accept:**
  - On synthetic data with a planted universal pattern, the gates promote the pattern.
  - A context-specific pattern (one slice only) is **rejected**.

## 6. Open questions
- Outcome utility weights: fixed in config, or learned per user?
- Question-type slices: derive them from the 本卦 upper/lower trigram, or from a text
  classifier?
- λ_max and K (stability cycles): start conservative, λ_max = 0.3 and K = 3?
- Until Phase 1 lands, the scheduled flywheel still RL-trains a candidate from v18 every
  6 h. Nothing gets promoted, because the evaluation is saturated. Should that step be
  disabled right away?

## 7. Implementation notes (2026-09-25)

Code: `world_model_v4.py` (WM V4, features, weights, evaluation), `prior_update.py`
(slow channel), `flywheel.py` (Steps 1–5), `yicenet_engine.py` (`_choose`),
`hooks_adapter.py` (`_previous_context`, `_record_decision`), settings under
`learning:` in config.yaml (`config.LEARNING_DEFAULTS`).
Tests: `tests/test_world_model_v4.py`, `tests/test_prior_update.py`.

Decisions taken while implementing:

- **Question features.** The WM sees the frozen encoder output `h0` of the question
  (before the env residual, L2-normalised, projected to 16 dims) rather than the
  9 probes, of which only 3 depend on the question. `h0` is returned by
  `YiCeNet.forward`, so re-ranking costs no second encode.
- **context_pre** = the previous turn's 27-dim `context_vector` plus four question-time
  signals (session position, whether the question is itself a correction or praise,
  its length) and an availability flag. Only `FEATURE_FIELDS` are read; a test flips
  every outcome and asserts identical inputs.
- **Legacy samples** train the same network with the 之卦 and context inputs masked
  (availability flags 0). The checkpoint records `mode: reduced` until samples with the
  之卦 prove re-ranking skill.
- **Baseline = recent base rate** (prevalence over the newest 15% of the training
  set). Outcome rates drift: abandonment went from 8–15% to 33% within weeks, and
  against the all-time rate a WM scored skill +0.07 by tracking the drift alone, with
  AUC below 0.5.
- **skill** = 1 − log-loss / base log-loss over all outcomes (per-outcome ratios blow
  up on near-constant outcomes). A WM is promoted only when the **bootstrap CI** of the
  per-sample gain excludes 0 and it beats the active WM. `rerank_skill` is the skill on
  samples that carry the 之卦 (≥ `min_full_test`) and is 0 unless its CI excludes 0.
- **Mixing Q and u.** Q moves on a ~0.1 scale, u on ±2, so
  `score = z(Q) + λ·z(u)` with z-scores within the candidate set (floors 1e-4 and
  0.05 so that near-ties are not amplified). λ = 0 ranks exactly as Q.
- **chosen_prob.** The 之卦 is argmax over candidates (the Gumbel sampling acts on the
  本卦), so `chosen_prob` = 1.0 and IPS is a no-op until `select_temperature` > 0
  turns on softmax exploration among candidates.
- **Slow channel.** The prior's Q depends only on the candidate hexagram
  (`value_net(hexagram_embed(k))`), so distillation fits the value head listwise to
  `softmax(z(E_ctx u))`. The context pool is 16 training contexts. The trust region
  shrinks the step (1, ½, … 1/64) until the mean KL of `softmax(Q/T)` ≤ ε, with T =
  the old Q spread. Slices: platform, time half, upper trigram of the 本卦. The shadow
  is a value-head file: the engine scores the same candidates with it and logs
  `shadow_chosen`, and promotion writes a full `yicenet_vN.pt` (active → fallback),
  which a running engine picks up from registry.json.
- A prior promoted by registry.json is hot-switched in the engine unless the caller
  pinned a checkpoint.

First run on real data (641 unique questions — 445 of 1086 buffered rows were exact
duplicates; all legacy):

| | log-loss | recent base rate | skill (95% CI) | AUC c/cp/a/ct |
|---|---|---|---|---|
| WM V4, reduced | 0.6607 | 0.6615 | +0.001 (−0.011, +0.015) | 0.46 / 0.65 / 0.39 / 0.40 |

Question + 本卦 alone do not predict the reaction beyond the current prevalence, so
nothing is promoted and λ stays 0. Only the "completed" outcome carries some signal.
The Phase 1 acceptance criterion "held-out log-loss lower than the base rate" is **not
met yet**. It needs the §3.1 samples, which the hooks record from now on. Re-ranking
adds +0.6 ms to `pre` (median 5.8 → 6.4 ms, CPU).

Open question 4 is settled: the RL step is gone.
