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
from typing import Optional

import torch
import torch.nn.functional as F

# ── Real Qwen BPE tokenizer ──
from .tokenizer import encode as yicenet_encode, build_vocab

# Check if vocab exists, build if not
_VOCAB_CHECKED = False
def _ensure_vocab():
    global _VOCAB_CHECKED
    if _VOCAB_CHECKED:
        return
    map_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "data", "qwen_to_yicenet.json")
    if not os.path.exists(map_path):
        print("[YiCeNet] Building vocabulary from session DB...")
        build_vocab()
    _VOCAB_CHECKED = True

# ── Hexagram names ──
HEXAGRAM_NAMES = [
    "乾", "坤", "屯", "蒙", "需", "讼", "师", "比",
    "小畜", "履", "泰", "否", "同人", "大有", "谦", "豫",
    "随", "蛊", "临", "观", "噬嗑", "贲", "剥", "复",
    "无妄", "大畜", "颐", "大过", "坎", "离", "咸", "恒",
    "遯", "大壮", "晋", "明夷", "家人", "睽", "蹇", "解",
    "损", "益", "夬", "姤", "萃", "升", "困", "井",
    "革", "鼎", "震", "艮", "渐", "归妹", "丰", "旅",
    "巽", "兑", "涣", "节", "中孚", "小过", "既济", "未济",
]

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

        # Resolve paths
        if not project_root:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        sys.path.insert(0, self._project_root)
        from src.model import YiCeNet
        from src.config import YiCeNetConfig

        self._config = YiCeNetConfig()
        self._model = YiCeNet(self._config).to(device).eval()

        saved = torch.load(ckpt, map_location=device, weights_only=False)
        self._model.load_state_dict(saved["model_state_dict"], strict=False)
        if "tau" in saved:
            self._model.tau = saved["tau"]

        self._active_checkpoint = ckpt
        _device_used = device
        _mem = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        print(f"[YiCeNet] Loaded {ckpt} on {_device_used} ({_mem:.0f}MB)")

    def predict(
        self,
        text: str,
        temperature: float = 0.1,
        deterministic: bool = False,
    ) -> dict:
        """
        Run full inference: encode → divine → evaluate → act.

        Args:
            text: Natural language task description
            temperature: Gumbel sampling temperature (ignored when deterministic=True)
            deterministic: If True, bypass Gumbel noise entirely.
                           Purely argmax over hexagram logits.
                           Use this for rigid/fixed workflows where
                           exploration must not interfere.

        Returns:
            dict with keys:
                hexagram_id, hexagram_name, hexagram_number, hexagram_pattern,
                best_candidate, selected_hexagram_id, selected_hexagram_name,
                candidates[{index, hexagram_id, hexagram_name, q_value}],
                action_id, action_name, q_values, temperature, deterministic
        """
        self._lazy_load()
        _ensure_vocab()

        config = self._config
        device = next(self._model.parameters()).device

        # ── REAL BPE tokenization ──
        input_ids, attention_mask = yicenet_encode(
            text, max_len=config.max_seq_len
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            # Encode context
            h = self._model.encode_context(input_ids, attention_mask)

            if deterministic:
                # ── Deterministic path: no Gumbel noise ──
                # Direct argmax over router logits
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
                from src.probes import extract_probes_tensor
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
                    tau=max(temperature, 0.01), hard=True
                )
                hex_idx = output["hexagram_idx"]
                hex_probs = output["hexagram_probs"]
                best_cand = output["best_candidate_idx"]
                cand_idxs = output["candidate_idxs"]
                cand_values = output["candidate_values"]
                action_ids = output["action_ids"]
                best_hex_id = cand_idxs.gather(1, best_cand.unsqueeze(-1)).squeeze(-1)
                probes = output.get("probes")
                # probes already extracted and prev_hexagram already updated in model.forward()

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
            h = cand_idxs_list[i]
            candidates.append({
                "index": i,
                "hexagram_id": h,
                "hexagram_name": HEXAGRAM_NAMES[h] if h < 64 else "???",
                "q_value": round(cand_values_list[i], 4),
            })

        pattern_lines = []
        for i in range(5, -1, -1):
            pattern_lines.append("—" if (hex_id >> i) & 1 else "- -")
        pattern = "\n".join(pattern_lines)

        return {
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
        }

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

    def switch_model(self, checkpoint_path: str) -> bool:
        """Hot-switch to a different checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        device = next(self._model.parameters()).device if self._model else self._resolve_device()
        from src.model import YiCeNet
        from src.config import YiCeNetConfig

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
        reg_path = os.path.join(self._project_root, "checkpoints", "registry.json")
        if not os.path.exists(reg_path):
            return None

        with open(reg_path) as f:
            reg = json.load(f)

        if not reg.get("ready"):
            return None

        ready = reg["ready"]
        active = reg.get("active")

        if active and ready.get("win_rate", 0) <= active.get("win_rate", 0) + 0.05:
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
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
