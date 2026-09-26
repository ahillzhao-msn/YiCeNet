"""
YiCeNet in-process inference engine — Plan B.

Loads the trained model directly in-process (no HTTP, no ONNX).
~4ms GPU inference, ~51MB memory.

Key design decisions:
  - deterministic=True bypasses Gumbel noise for rigid workflows
  - exploration_override lets callers force τ=0 for specific tasks
  - trajectory logging includes terminal_type for reward disambiguation

Usage:
    from yicenet_engine import YiCeNetEngine
    engine = YiCeNetEngine()
    result = engine.predict("search knowledge base")
    # → {hexagram: 35, hexagram_name: "晋", action: "route_to_api", q_values: [...]}
    engine.switch_model("checkpoints/yicenet_rl_final.pt")
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .types import PredictionResult, EnvAnalysis
    from .cross_attention import Prescription

import torch
import torch.nn.functional as F
import numpy as np

# ── Real Qwen BPE tokenizer ──
from .tokenizer import encode as yicenet_encode, build_vocab
from .config import yicenet_data_dir

# Cross-attention MemoryBank (Phase 1)
from .memory_bank import get_memory_bank
from .cross_attention import CrossAttention, ContextPrescription

# Check if vocab exists, build if not
_VOCAB_CHECKED = False
def _ensure_vocab():
    global _VOCAB_CHECKED
    if _VOCAB_CHECKED:
        return
    map_path = yicenet_data_dir() / "qwen_to_yicenet.json"
    if not map_path.exists():
        print("[YiCeNet] Building vocabulary from session DB...", file=sys.stderr)
        build_vocab()
    _VOCAB_CHECKED = True

from .display import HEXAGRAM_NAMES  # single source of truth — defined in display.py

ACTION_NAMES = [
    "route_to_service", "parallel_invoke", "sequential_chain",
    "aggregate_results", "wait_poll", "notify_user", "cache_lookup",
    "intent_classify", "context_retrieve", "response_generate",
    "fan_out_merge", "model_select", "entity_extract_query",
    "summarize_doc", "translate_format", "validate_process",
    "stream_progress", "load_balance", "retry_backoff", "subagent_delegate",
    "tool_registry_select", "schedule_recurring", "chain_of_thought",
    "context_window_mgmt", "error_recovery", "search_knowledge_base",
    "route_multi_api", "conditional_branch", "parallel_fetch_aggregate",
    "caching_fallback", "multi_step_form", "monitor_poll",
    "auth_check", "data_validation", "batch_process",
    "incremental_sync", "circuit_breaker", "health_check",
    "log_audit", "metric_collect", "alert_trigger", "rollback",
    "rate_limit_enforce", "extract_transform", "streaming_response",
    "dead_letter_handle", "format_conversion", "retry_failover",
    "binary_decision", "route_bypass",
]


class YiCeNetEngine:
    """
    In-process YiCeNet inference engine.

    Loads model lazily on first predict() call.
    Supports A/B weight switching without restart.
    Supports deterministic mode for rigid workflows.
    Thread-safe for single-process use.
    """

    def __init__(
        self,
        checkpoint: str = "",
        device: str = "auto",
        project_root: str = "",
    ):
        self._model = None
        self._device = device
        self._checkpoint = checkpoint
        self._config = None

        # Per-session env signal cache: stores signals computed from the
        # PREVIOUS turn that are useful for conditioning the CURRENT turn.
        # Currently tracks attention_entropy from CrossAttention.
        # Dict[session_id, Dict[signal_name, value]]
        self._session_env_cache: dict[str, dict] = {}

        # 察言观色 (DESIGN-phase4 §3.3): the active world model re-ranks the 之卦
        # candidates; a shadow value head (§3.5) logs what it would choose.
        # Both follow registry.json, re-read when the file changes.
        self._wm = None
        self._wm_entry: Optional[dict] = None
        self._wm_lambda = 0.0
        self._shadow_value_net = None
        self._shadow_version = ""
        self._registry_stamp = None
        self._learning: Optional[dict] = None

        # Resolve paths: YICENET_HOME env var > explicit > auto-detect
        if not project_root:
            from .config import yicenet_home
            project_root = str(yicenet_home())
        self._project_root = project_root
        if not checkpoint:
            # Don't set default here — let _lazy_load check registry.json first
            pass

    def _resolve_device(self) -> str:
        if self._device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _lazy_load(self):
        """Load model on first use with registry-aware fallback."""
        if self._model is not None:
            return

        device = self._resolve_device()
        ckpt = self._checkpoint

        # If no explicit checkpoint, try registry.json
        if not ckpt:
            reg_path = os.path.join(self._project_root, "checkpoints", "registry.json")
            if os.path.exists(reg_path):
                try:
                    with open(reg_path) as f:
                        reg = json.load(f)
                    ckpt = reg.get("active", {}).get("path", "")
                    # Resolve relative path (relative to checkpoints/)
                    if ckpt and not os.path.isabs(ckpt):
                        ckpt = os.path.join(self._project_root, "checkpoints", ckpt)
                except Exception:
                    pass

        # Default fallback
        if not ckpt:
            default_ckpt = os.path.join(self._project_root, "checkpoints", "yicenet_rl_best.pt")
            if os.path.exists(default_ckpt):
                ckpt = default_ckpt

        if not ckpt or not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"YiCeNet checkpoint not found. "
                "Set checkpoint path or run training first."
            )

        # Import here to avoid top-level dependency on heavy libs
        from yicenet.model import YiCeNet
        from yicenet.config import YiCeNetConfig

        self._config = YiCeNetConfig()
        self._model = YiCeNet(self._config).to(device).eval()

        # Backward compat shim: old checkpoints reference src.* module paths
        # which moved to yicenet.* in the restructure.
        import types
        if 'src' not in sys.modules:
            _src = types.ModuleType('src')
            _src.__path__ = []
            sys.modules['src'] = _src
        import importlib
        for _mod in ['model', 'config', 'tokenizer', 'encoder', 'decoder', 'hexagram', 'value_net']:
            if f'src.{_mod}' not in sys.modules:
                try:
                    sys.modules[f'src.{_mod}'] = importlib.import_module(f'yicenet.{_mod}')
                except ImportError:
                    pass

        saved = torch.load(ckpt, map_location=device, weights_only=False)
        self._model.load_state_dict(saved["model_state_dict"], strict=False)
        if "tau" in saved:
            self._model.tau = saved["tau"]

        self._active_checkpoint = ckpt
        _device_used = device
        _mem = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        print(f"[YiCeNet] Loaded {ckpt} on {_device_used} ({_mem:.0f}MB)", file=sys.stderr)

    # ── Hexagram chain signal computation ─────────────────────────────────

    def _compute_chain_signals(self, session_id: str) -> dict:
        """Compute hexagram chain dynamics + cached env signals for this session.

        Queries the global MemoryBank for the hexagram sequence and derives
        structural chain signals (stability, velocity, clan diversity, entropy).
        Also merges `_session_env_cache[session_id]` which holds signals from
        the previous turn (currently: attention_entropy from CrossAttention).

        Returns an empty dict when there is no history or session_id is empty.
        Caller-supplied values in environment always override auto-computed ones.
        """
        if not session_id:
            return {}

        import math as _math
        from collections import Counter

        bank = get_memory_bank()
        history = bank.get_hexagram_history(session_id)
        depth = bank.get_turn_count(session_id)

        result: dict = {"memory_bank_depth": depth}

        # Include cached signals from the PREVIOUS turn (e.g. attention_entropy)
        cached = self._session_env_cache.get(session_id, {})
        result.update(cached)

        if not history:
            return result

        # ── Hexagram stability: consecutive same hexagram at tail ──────────
        last_hx = history[-1]
        stability = 1
        for hx in reversed(history[:-1]):
            if hx == last_hx:
                stability += 1
            else:
                break
        result["hexagram_stability"] = stability  # raw int; build_env_vec /20

        # ── Hexagram velocity: EMA of |jump| between successive hexagrams ──
        if len(history) >= 2:
            jumps = [abs(history[i] - history[i - 1]) for i in range(1, len(history))]
            ema = float(jumps[0])
            alpha = 0.3
            for j in jumps[1:]:
                ema = (1 - alpha) * ema + alpha * float(j)
            result["hexagram_velocity"] = ema / 63.0  # normalized
        else:
            result["hexagram_velocity"] = 0.0

        # ── Clan diversity: fraction of 8 trigram clans visited (last 20) ──
        recent = history[-20:]
        clans_visited = len({hx // 8 for hx in recent})
        result["clan_diversity"] = clans_visited / 8.0

        # ── Hexagram entropy: Shannon entropy of recent hex distribution ───
        counts = Counter(recent)
        n = len(recent)
        if n > 1:
            entropy = -sum((c / n) * _math.log(c / n) for c in counts.values())
            max_e = _math.log(min(n, 64))
            result["hexagram_entropy"] = entropy / max_e if max_e > 0 else 0.0
        else:
            result["hexagram_entropy"] = 0.0

        return result

    # ──────────────────────────────────────────────────────────────────────

    def predict(
        self,
        text: str,
        temperature: float = 0.1,
        deterministic: bool = False,
        return_prescription: bool = False,
        session_id: str = "",
        turn_id: int = 0,
        turn_summary: str = "",
        environment: Optional[dict] = None,
        wm_context: Optional[dict] = None,
    ) -> "PredictionResult":
        """
        Run full inference: encode → divine → evaluate → act.

        Args:
            text: Natural language task description
            temperature: Gumbel sampling temperature (ignored when deterministic=True)
            deterministic: If True, bypass Gumbel noise entirely.
            return_prescription: If True, also run cross-attention against
                                 historical turns and return context_prescription.
            session_id: Session identifier (required for prescription)
            turn_id: Sequential turn number within session
            turn_summary: Optional short summary of this turn
            environment: Optional dict of structural signals that condition routing.
                         Allowed keys: hour_of_day, session_turn, last_hexagram_id,
                         correction_rate, satisfaction_ema, attention_entropy,
                         last_tool_success.  Unknown keys are silently ignored.
                         See env_context.py for full documentation.
            wm_context: What the world model may know at question time:
                        {"context_pre": previous turn's context_vector, "turn_id": n}.
                        None → the prior alone chooses the 之卦.

        Returns:
            dict with keys:
                hexagram_id, hexagram_name, hexagram_number, hexagram_pattern,
                best_candidate, selected_hexagram_id, selected_hexagram_name,
                candidates[{index, hexagram_id, hexagram_name, q_value}],
                action_id, action_name, q_values, temperature, deterministic,
                probes, env_confidence, context_status,
                context_hint (only when context_status != "sufficient"),
                context_prescription (only when return_prescription=True and session_id given),
                chosen_prob, decision (base/chosen 之卦, candidates, candidate_q, …)
        """
        from .env_context import build_env_vec, compute_env_confidence

        self._lazy_load()
        _ensure_vocab()
        self._refresh_learning_state()  # before inference: may hot-switch the prior

        config = self._config
        device = next(self._model.parameters()).device

        # ── REAL BPE tokenization ──
        input_ids, attention_mask = yicenet_encode(
            text, max_len=config.max_seq_len
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Auto-compute hexagram chain signals from MemoryBank when session_id
        # is provided.  Caller-supplied values take precedence (merge order:
        # chain_signals first, then caller's environment overrides).
        if session_id:
            get_memory_bank().init_session(session_id)
            chain = self._compute_chain_signals(session_id)
            if chain:
                environment = {**chain, **(environment or {})}

        # Build env vector once; None → no-op inside encode_context
        env_vec = build_env_vec(environment)
        if env_vec is not None:
            env_vec = env_vec.to(device)

        with torch.no_grad():
            if deterministic:
                # ── Deterministic path: no Gumbel noise ──
                # Direct argmax over router logits
                h0 = self._model.encoder(input_ids, attention_mask)
                h = self._model.add_env(h0, env_vec)
                router_logits = self._model.router.projection(h)
                hex_idx = router_logits.argmax(dim=-1)  # (1,)
                hex_probs = F.softmax(router_logits, dim=-1)

                # Evaluate candidates using the deterministic hexagram
                best_cand, cand_idxs, cand_values = (
                    self._model.evaluate_candidates(hex_idx, h)
                )

                # Select best hexagram
                best_hex_id = cand_idxs.gather(1, best_cand.unsqueeze(-1)).squeeze(-1)

                # Decode action
                action_ids, action_logits = self._model.decode_action(best_hex_id)

                # Extract probes (deterministic path — tensor form)
                from yicenet.probes import extract_probes_tensor
                prev_t = (self._model._prev_hexagram_idx_tensor.to(device)
                          if hasattr(self._model, "_prev_hexagram_idx_tensor")
                          and self._model._prev_hexagram_idx_tensor is not None
                          else None)
                probe_tensor = extract_probes_tensor(
                    h=h,
                    router_logits=router_logits,
                    router_probs=hex_probs,
                    candidate_values=cand_values,
                    hexagram_idx=hex_idx,
                    prev_hexagram_idx=prev_t,
                    action_logits=action_logits,
                )
                self._model._prev_hexagram_idx_tensor = hex_idx.clone()
                self._model._prev_hexagram_idx = hex_idx[0].item()

            else:
                # ── Stochastic path: Gumbel-Softmax sampling ──
                output = self._model(
                    input_ids, attention_mask,
                    tau=max(temperature, 0.01), hard=True,
                    env_vec=env_vec,
                )
                hex_idx = output["hexagram_idx"]
                hex_probs = output["hexagram_probs"]
                best_cand = output["best_candidate_idx"]
                cand_idxs = output["candidate_idxs"]
                cand_values = output["candidate_values"]
                action_ids = output["action_ids"]
                best_hex_id = cand_idxs.gather(1, best_cand.unsqueeze(-1)).squeeze(-1)
                h = output["h"]
                h0 = output["h0"]
                # probes already extracted and prev_hexagram already updated in model.forward()

            # ── Choose the 之卦 among the 本卦's candidates ──
            decision = self._choose(text, h0, hex_idx, cand_idxs, cand_values, wm_context)
            if decision["index"] != int(best_cand.reshape(-1)[0]):
                best_cand = torch.tensor([decision["index"]], device=cand_idxs.device)
                best_hex_id = cand_idxs.gather(1, best_cand.unsqueeze(-1)).squeeze(-1)
                action_ids, _ = self._model.decode_action(best_hex_id)

        # ── Build result ──
        hex_id = hex_idx.item()
        cand_idxs_list = cand_idxs.squeeze(0).tolist()
        cand_values_list = cand_values.squeeze().tolist()
        best_cand_val = best_cand.item() if hasattr(best_cand, 'item') else best_cand
        action_id = action_ids.item()

        # ── Probe vector → list ──
        if deterministic:
            probe_list = probe_tensor.tolist()  # from deterministic path
        else:
            probe_tensor_from_output = output.get("probes")
            probe_list = probe_tensor_from_output.tolist() if probe_tensor_from_output is not None else None

        candidates = []
        for i in range(8):
            cid = cand_idxs_list[i]
            candidates.append({
                "index": i,
                "hexagram_id": cid,
                "hexagram_name": HEXAGRAM_NAMES[cid] if cid < 64 else "???",
                "q_value": round(cand_values_list[i], 4),
            })

        pattern_lines = []
        for i in range(5, -1, -1):
            pattern_lines.append("—" if (hex_id >> i) & 1 else "- -")
        pattern = "\n".join(pattern_lines)

        # ── Build result ──
        result = {
            "hexagram_id": hex_id,
            "hexagram_name": HEXAGRAM_NAMES[hex_id] if hex_id < 64 else "???",
            "hexagram_number": hex_id + 1,
            "hexagram_pattern": pattern,
            "best_candidate": best_cand_val,
            "selected_hexagram_id": cand_idxs_list[best_cand_val],
            "selected_hexagram_name": (
                HEXAGRAM_NAMES[cand_idxs_list[best_cand_val]]
                if cand_idxs_list[best_cand_val] < 64 else "???"
            ),
            "candidates": candidates,
            "action_id": action_id,
            "action_name": (
                ACTION_NAMES[action_id]
                if action_id < len(ACTION_NAMES) else f"action_{action_id}"
            ),
            "q_values": [round(v, 4) for v in cand_values_list],
            "temperature": temperature if not deterministic else 0.0,
            "deterministic": deterministic,
            "probes": probe_list,
            "chosen_prob": decision["chosen_prob"],
        }
        # The decision record the flywheel learns from (DESIGN-phase4 §3.1).
        result["decision"] = {
            "base_hexagram": hex_id,
            "chosen_hexagram": cand_idxs_list[best_cand_val],
            "candidates": cand_idxs_list,
            "candidate_q": [round(v, 4) for v in cand_values_list],
            "chosen_prob": decision["chosen_prob"],
            "action": result["action_name"],
            "wm_lambda": decision["wm_lambda"],
        }
        if decision.get("wm_utility") is not None:
            result["decision"]["wm_utility"] = decision["wm_utility"]
        if decision.get("shadow_chosen") is not None:
            result["decision"]["shadow_chosen"] = decision["shadow_chosen"]
            result["decision"]["shadow_version"] = self._shadow_version

        # ── Environment confidence (derived from probe structural signals) ──
        env_conf, ctx_status, ctx_hint = compute_env_confidence(
            probe_list, cand_values_list
        )
        result["env_confidence"] = env_conf
        result["context_status"] = ctx_status
        if ctx_hint:
            result["context_hint"] = ctx_hint

        # ── Cross-attention prescription (Phase 1) ──
        if return_prescription and session_id:
            encoder_np = h.squeeze(0).cpu().numpy().astype(np.float32)
            encoder_np = encoder_np / (np.linalg.norm(encoder_np) + 1e-10)

            bank = get_memory_bank()
            bank.init_session(session_id)

            # Store this turn's encoder output
            bank.store_turn(
                session_id=session_id,
                turn_id=turn_id,
                encoder_output=encoder_np,
                hexagram_id=hex_id,
                summary=turn_summary,
            )

            # Run cross-attention against historical turns
            keys, meta = bank.get_session_keys(session_id)
            if keys.shape[0] <= 1:
                # First turn: no history, inject empty prescription (full mode)
                result["context_prescription"] = {
                    "mode": "full",
                    "retain_turns": [turn_id] if keys.shape[0] > 0 else [],
                    "summarize_turns": [],
                    "discard_turns": [],
                    "attention_entropy": 0.0,
                    "compression_ratio": 0.0,
                    "key_insight": "首輪 — 無歷史可參考",
                }
            else:
                # Use the stored encoder as query (from keys, which was just appended)
                query = encoder_np
                # Run attention against all EXCEPT the current turn
                past_keys = keys[:-1]
                past_meta = meta[:-1]

                attn = CrossAttention()
                weights = attn.compute(query, past_keys)

                rx = ContextPrescription(weights, past_meta, n_turns_total=keys.shape[0])
                prescription = rx.generate()
                pdict = prescription.to_dict()
                result["context_prescription"] = pdict

                # Cache attention_entropy for use in the NEXT turn's env_vec
                attn_e = pdict.get("attention_entropy", 0.0)
                self._session_env_cache.setdefault(session_id, {})["attention_entropy"] = float(attn_e)

        return result

    # ── 察言观色: world-model re-ranking (DESIGN-phase4 §3.3) ─────────────

    def _refresh_learning_state(self) -> None:
        """Load the active world model and shadow head when registry.json changes."""
        from .config import get_learning_config, yicenet_checkpoint_dir
        if self._learning is None:
            self._learning = get_learning_config()
        reg_path = yicenet_checkpoint_dir() / "registry.json"
        try:
            stamp = reg_path.stat().st_mtime_ns
        except OSError:
            stamp = None
        if stamp == self._registry_stamp:
            return
        self._registry_stamp = stamp
        self._wm, self._wm_entry, self._wm_lambda = None, None, 0.0
        self._shadow_value_net, self._shadow_version = None, ""
        if stamp is None:
            return
        try:
            reg = json.loads(reg_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        # A promoted prior (卦理 update) takes over without a restart — unless the
        # caller pinned a checkpoint.
        active_path = (reg.get("active") or {}).get("path")
        if not self._checkpoint and active_path and self._model is not None:
            path = str(yicenet_checkpoint_dir() / active_path)
            if os.path.exists(path) and os.path.normcase(os.path.abspath(path)) != os.path.normcase(
                    os.path.abspath(self.active_checkpoint)):
                try:
                    self.switch_model(path)
                    print(f"[YiCeNet] Switched to active prior {path}", file=sys.stderr)
                except Exception as exc:
                    print(f"[YiCeNet] active prior not loaded: {exc}", file=sys.stderr)
        from .world_model_v4 import WorldModelV4, rerank_lambda
        entry = (reg.get("world_model") or {}).get("active")
        if entry and entry.get("path"):
            try:
                self._wm = WorldModelV4.load(str(yicenet_checkpoint_dir() / entry["path"]))
                self._wm_entry = entry
                self._wm_lambda = rerank_lambda(entry, self._learning["lambda_max"])
            except Exception as exc:
                print(f"[YiCeNet] world model not loaded: {exc}", file=sys.stderr)
        shadow = reg.get("shadow")
        if shadow and shadow.get("path") and self._model is not None:
            try:
                from .prior_update import load_value_head
                self._shadow_value_net = load_value_head(
                    str(yicenet_checkpoint_dir() / shadow["path"]), self._model)
                self._shadow_version = shadow.get("version", "")
            except Exception as exc:
                print(f"[YiCeNet] shadow prior not loaded: {exc}", file=sys.stderr)

    def _choose(self, question, h0, hex_idx, cand_idxs, cand_values, wm_context) -> dict:
        """Pick the 之卦: score = z(Q_prior) + λ·z(u_WM), λ gated by the WM's skill.

        λ = 0 (no WM, unproven WM, or no wm_context) ranks exactly as the prior does.
        The WM only re-ranks the candidates the 本卦 produced.
        """
        from .world_model_v4 import rerank_scores, zscore, Q_STD_FLOOR

        q = cand_values.reshape(-1).float().cpu()
        cands = cand_idxs.reshape(-1).cpu()
        use_wm = self._wm is not None and wm_context is not None
        lam = self._wm_lambda if use_wm else 0.0
        u = None
        if use_wm and (lam > 0 or self._shadow_value_net is not None):
            u = self._wm_utility(question, h0, int(hex_idx.reshape(-1)[0]), cands, wm_context)
        score = rerank_scores(q, u, lam) if u is not None else q

        temp = float(self._learning.get("select_temperature", 0.0) or 0.0)
        if temp > 0:
            probs = F.softmax(zscore(score, Q_STD_FLOOR) / temp, dim=-1)
            idx = int(torch.multinomial(probs, 1).item())
            chosen_prob = round(float(probs[idx]), 4)
        else:
            idx = int(score.argmax())
            chosen_prob = 1.0

        out = {"index": idx, "chosen_prob": chosen_prob, "wm_lambda": round(lam, 4),
               "wm_utility": [round(float(v), 4) for v in u] if u is not None else None}
        if self._shadow_value_net is not None:
            with torch.no_grad():
                embeds = self._model.hexagram_embed(cand_idxs)
                sq = self._shadow_value_net(embeds).reshape(-1).float().cpu()
            s_score = rerank_scores(sq, u, lam) if u is not None else sq
            out["shadow_chosen"] = int(cands[int(s_score.argmax())])
        return out

    def _wm_utility(self, question, h0, base: int, cands, wm_context: dict) -> torch.Tensor:
        from .world_model_v4 import context_features
        ctx, has_ctx = context_features(wm_context.get("context_pre"), wm_context.get("turn_id"), question)
        k = len(cands)
        h = F.normalize(h0.float(), dim=-1).cpu().expand(k, -1)
        return self._wm.utility(
            h, torch.full((k,), base, dtype=torch.long), cands.long(), torch.ones(k),
            torch.tensor([ctx] * k, dtype=torch.float32), torch.full((k,), float(has_ctx)),
            weights=self._learning.get("utility_weights"),
        )

    def predict_structured(self, text: str, temperature: float = 0.1,
                           deterministic: bool = False) -> str:
        """Predict and return a human-readable formatted string."""
        r = self.predict(text, temperature, deterministic)
        mode = "DET" if deterministic else f"τ={r['temperature']}"
        lines = [
            f"┌─ YiCeNet [{mode}] ───────────────────────┐",
            f"│ Task: {text[:48]:48s} │",
            f"│ 起卦: {r['hexagram_name']} (#{r['hexagram_number']})        │",
            f"│ {r['hexagram_pattern']}",
            f"│ ── 评估 (选中={r['best_candidate']}) ──",
        ]
        for c in r["candidates"]:
            mark = " ◀" if c["index"] == r["best_candidate"] else "  "
            lines.append(
                f"│ [{c['index']}] {c['hexagram_name']:6s} "
                f"Q={c['q_value']:+.4f}{mark}"
            )
        lines.append(
            f"│ → {r['selected_hexagram_name']} → "
            f"[{r['action_id']}] {r['action_name']}"
        )
        lines.append(f"└────────────────────────────────────────┘")
        return "\n".join(lines)

    def attend(
        self,
        text: str,
        session_id: str,
        turn_id: int = 0,
        turn_summary: str = "",
        embedding: Optional[np.ndarray] = None,
        environment: Optional[dict] = None,
    ) -> dict:  # returns {"context_prescription": dict}
        """Lightweight encode + store + cross-attention. No hexagram overhead.

        ~3ms with TinyEncoder; ~2ms with external bge-small embedding.

        Args:
            text: User input text (for fallback TinyEncoder path)
            session_id: Session identifier
            turn_id: Sequential turn number
            turn_summary: Optional short summary
            embedding: Optional pre-computed 384d vector — skips TinyEncoder.
            environment: Optional structural env signals (see env_context.py).
                         Applied only on the TinyEncoder path; ignored when
                         an external embedding is provided.

        Returns:
            dict with context_prescription key (always present)
        """
        if embedding is not None:
            # External embedding provided — skip TinyEncoder
            encoder_np = embedding.astype(np.float32)
            encoder_np = encoder_np / (np.linalg.norm(encoder_np) + 1e-10)
        else:
            # Fallback: use TinyEncoder (+ optional env residual)
            from .env_context import build_env_vec
            self._lazy_load()
            _ensure_vocab()

            config = self._config
            device = next(self._model.parameters()).device

            input_ids, attention_mask = yicenet_encode(
                text, max_len=config.max_seq_len
            )
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            env_vec = build_env_vec(environment)
            if env_vec is not None:
                env_vec = env_vec.to(device)

            with torch.no_grad():
                h = self._model.encode_context(input_ids, attention_mask, env_vec)

            encoder_np = h.squeeze(0).cpu().numpy().astype(np.float32)
            encoder_np = encoder_np / (np.linalg.norm(encoder_np) + 1e-10)
        
        bank = get_memory_bank()
        bank.init_session(session_id)
        
        # Store this turn's encoder output (no hexagram_id — not computed)
        bank.store_turn(
            session_id=session_id,
            turn_id=turn_id,
            encoder_output=encoder_np,
            hexagram_id=-1,
            summary=turn_summary,
        )
        
        # Run cross-attention
        keys, meta = bank.get_session_keys(session_id)
        if keys.shape[0] <= 1:
            prescription = {
                "mode": "full",
                "retain_turns": [turn_id] if keys.shape[0] > 0 else [],
                "summarize_turns": [],
                "discard_turns": [],
                "attention_entropy": 0.0,
                "compression_ratio": 0.0,
                "key_insight": "首輪 — 無歷史可參考",
            }
        else:
            query = encoder_np
            past_keys = keys[:-1]
            past_meta = meta[:-1]
            
            attn = CrossAttention()
            weights = attn.compute(query, past_keys)

            rx = ContextPrescription(weights, past_meta, n_turns_total=keys.shape[0])
            prescription = rx.generate().to_dict()

            # Cache attention_entropy for the NEXT turn's env_vec
            attn_e = prescription.get("attention_entropy", 0.0)
            self._session_env_cache.setdefault(session_id, {})["attention_entropy"] = float(attn_e)

        return {"context_prescription": prescription}

    def analyze(
        self,
        task_brief: str,
        environment: Optional[dict] = None,
    ) -> "EnvAnalysis":
        """Fast environment analysis (~3ms): encode + probes only, no routing.

        Returns EnvAnalysis-shaped dict with probes, env_confidence,
        context_status, context_hint. Does NOT update MemoryBank.
        """
        from .env_context import build_env_vec, compute_env_confidence

        self._lazy_load()
        _ensure_vocab()

        config = self._config
        device = next(self._model.parameters()).device

        input_ids, attention_mask = yicenet_encode(task_brief, max_len=config.max_seq_len)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        env_vec = build_env_vec(environment)
        if env_vec is not None:
            env_vec = env_vec.to(device)

        with torch.no_grad():
            h = self._model.encode_context(input_ids, attention_mask, env_vec)

        from yicenet.probes import extract_probes_tensor
        probe_tensor = extract_probes_tensor(
            h=h,
            router_logits=torch.zeros(1, 64, device=device),
            router_probs=torch.ones(1, 64, device=device) / 64,
            candidate_values=torch.zeros(1, 8, 1, device=device),
            hexagram_idx=torch.zeros(1, dtype=torch.long, device=device),
            prev_hexagram_idx=None,
            action_logits=torch.zeros(1, len(ACTION_NAMES), device=device),
        )
        probe_list = probe_tensor.tolist()
        dummy_q = [0.0] * 8
        conf, status, hint = compute_env_confidence(probe_list, dummy_q)

        result: dict = {
            "probes": probe_list,
            "env_confidence": conf,
            "context_status": status,
        }
        if hint:
            result["context_hint"] = hint
        return result

    def switch_model(self, checkpoint_path: str) -> bool:
        """Hot-switch to a different checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        device = next(self._model.parameters()).device if self._model else self._resolve_device()
        from yicenet.model import YiCeNet
        from yicenet.config import YiCeNetConfig

        config = self._config or YiCeNetConfig()
        new_model = YiCeNet(config).to(device).eval()
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        new_model.load_state_dict(saved["model_state_dict"], strict=False)
        if "tau" in saved:
            new_model.tau = saved["tau"]

        self._model = new_model
        self._active_checkpoint = checkpoint_path
        return True

    def check_for_switch(self) -> dict | None:
        """
        Check registry.json for a ready model to switch to.
        Call periodically from Hermes cron.
        Returns switch result dict, or None if no switch needed.
        """
        from .config import yicenet_checkpoint_dir
        reg_path = yicenet_checkpoint_dir() / "registry.json"
        if not reg_path.exists():
            return None

        with open(reg_path) as f:
            reg = json.load(f)

        if not reg.get("ready"):
            return None

        ready = reg["ready"]
        active = reg.get("active")
        # Perform switch (≥3% improvement threshold)
        if ready.get("win_rate", 0) < active.get("win_rate", 0) + 0.03:
            return {"should_switch": False, "reason": "Insufficient improvement"}

        # Perform switch
        self.switch_model(ready["path"])

        # Update registry: promote ready to active
        reg["fallback"] = reg.get("active")
        reg["active"] = reg["ready"]
        reg["ready"] = None
        with open(reg_path, "w") as f:
            json.dump(reg, f, indent=2)

        return {
            "should_switch": True,
            "new_version": reg["active"]["version"],
            "new_avg_reward": reg["active"]["avg_reward"],
        }

    @property
    def active_checkpoint(self) -> str:
        return getattr(self, "_active_checkpoint", "")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self):
        """Free GPU memory."""
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Singleton ──
_engine: Optional[YiCeNetEngine] = None


def get_engine(checkpoint: str = "") -> YiCeNetEngine:
    global _engine
    if _engine is None:
        from .config import yicenet_home
        project_root = str(yicenet_home())
        _engine = YiCeNetEngine(checkpoint=checkpoint, project_root=project_root)
    return _engine


def predict(text: str, temperature: float = 0.1,
            deterministic: bool = False) -> dict:
    """Quick one-shot predict using global engine."""
    return get_engine().predict(text, temperature, deterministic)


if __name__ == "__main__":
    engine = get_engine()
    # Compare stochastic vs deterministic
    for text in [
        "search knowledge base",
        "route to multiple APIs and merge",
        "handle error with retry",
    ]:
        print(engine.predict_structured(text, temperature=0.5))
        print(engine.predict_structured(text, deterministic=True))
        print()
