#!/usr/bin/env python3
"""
YiCeNet inference demo — test the trained model end-to-end.

Usage:
    python scripts/demo.py                          # uses rl_best checkpoint
    python scripts/demo.py --checkpoint checkpoints/yicenet_pretrained.pt
    python scripts/demo.py --interactive            # interactive mode
    python scripts/demo.py --scenario "search knowledge base"
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import YiCeNet
from src.config import YiCeNetConfig
from src.tokenizer import encode as yicenet_encode

# ── Hexagram names (King Wen order) ──
HEXAGRAM_NAMES = [
    "乾 (Creative)", "坤 (Receptive)", "屯 (Difficulty)", "蒙 (Youthful Folly)",
    "需 (Waiting)", "讼 (Conflict)", "师 (Army)", "比 (Union)",
    "小畜 (Small Accumulation)", "履 (Treading)", "泰 (Peace)", "否 (Standstill)",
    "同人 (Fellowship)", "大有 (Great Possession)", "谦 (Modesty)", "豫 (Enthusiasm)",
    "随 (Following)", "蛊 (Decay)", "临 (Approach)", "观 (Contemplation)",
    "噬嗑 (Biting Through)", "贲 (Grace)", "剥 (Falling Away)", "复 (Return)",
    "无妄 (Innocence)", "大畜 (Great Accumulation)", "颐 (Nourishment)", "大过 (Great Excess)",
    "坎 (Darkness)", "离 (Clarity)", "咸 (Influence)", "恒 (Duration)",
    "遯 (Retreat)", "大壮 (Great Power)", "晋 (Progress)", "明夷 (Darkening)",
    "家人 (Family)", "睽 (Opposition)", "蹇 (Obstruction)", "解 (Deliverance)",
    "损 (Decrease)", "益 (Increase)", "夬 (Resolution)", "姤 (Meeting)",
    "萃 (Gathering)", "升 (Pushing Upward)", "困 (Oppression)", "井 (The Well)",
    "革 (Revolution)", "鼎 (The Cauldron)", "震 (Shock)", "艮 (Stillness)",
    "渐 (Gradual Progress)", "归妹 (Marrying Maiden)", "丰 (Abundance)", "旅 (Traveling)",
    "巽 (Gentle Penetration)", "兑 (Joy)", "涣 (Dispersion)", "节 (Limitation)",
    "中孚 (Inner Truth)", "小过 (Small Excess)", "既济 (Completion)", "未济 (Incomplete)",
]

HEXAGRAM_TRANSFORM_NAMES = ["本卦 (Original)", "错卦 (Opposite)", "综卦 (Upside-down)", "互卦 (Inner)",
                            "之卦1", "之卦2", "之卦3", "之卦4"]

ACTION_NAMES = [
    "route_to_service_A", "parallel_invoke_B_C", "sequential_D_to_E",
    "aggregate_results", "wait_poll", "notify_user", "cache_lookup",
    "intent_classify", "context_retrieve", "response_generate",
    "fan_out_3_merge", "model_select", "entity_extract_db_query",
    "summarize_doc", "translate_format", "validate_process",
    "stream_progress", "load_balance", "retry_backoff", "subagent_delegate",
    "tool_registry_select", "schedule_recurring", "chain_of_thought",
    "context_window_mgmt", "error_recovery", "search_knowledge_base",
    "route_multi_api", "conditional_branch", "parallel_fetch_aggregate",
    "caching_fallback", "multi_step_form", "monitor_poll",
    "sequential_chain", "binary_decision", "extract_transform",
    "streaming_response", "retry_failover", "rate_limit_enforce",
    "auth_check", "data_validation", "format_conversion",
    "batch_process", "incremental_sync", "dead_letter_handle",
    "circuit_breaker", "health_check", "log_audit", "metric_collect",
    "alert_trigger", "rollback",
]


def bit_pattern(hex_idx: int) -> str:
    """Render as 6-line I Ching pattern: — solid, - - broken."""
    lines = []
    for i in range(5, -1, -1):
        lines.append("———" if (hex_idx >> i) & 1 else "─ ─")
    return "\n      ".join(lines)


def hexagram_info(hex_idx: int) -> str:
    """Return hexagram number, name, and bit pattern."""
    name = HEXAGRAM_NAMES[hex_idx] if hex_idx < 64 else "???"
    pattern = bit_pattern(hex_idx)
    yang = bin(hex_idx).count("1")
    yin = 6 - yang
    return (f"    {name} (#{hex_idx+1})  "
            f"⚊{yang} ⚋{yin}\n"
            f"      {pattern}")


def demo_single(model: YiCeNet, device: str, input_ids, attention_mask, label: str):
    """Run one inference and print full decision trace."""
    model.eval()
    with torch.no_grad():
        output = model(input_ids, attention_mask, tau=0.1, hard=True)

    h = output["h"]
    hex_idx = output["hexagram_idx"].item()
    best_cand = output["best_candidate_idx"].item()
    cand_idxs = output["candidate_idxs"].squeeze(0).tolist()
    cand_values = output["candidate_values"].squeeze().tolist()
    action_id = output["action_ids"].item()

    print(f"\n{'='*60}")
    print(f"  Scenario: {label}")
    print(f"{'='*60}")

    # State vector summary
    h_norm = h.norm().item()
    print(f"\n  ── Encoder ──")
    print(f"  State vector h: 256-dim, norm={h_norm:.3f}")

    # Divination result
    print(f"\n  ── Divination (起卦) ──")
    print(f"  Sampled hexagram: {hex_idx}")
    print(f"  {hexagram_info(hex_idx)}")

    # All candidates with Q-values
    print(f"\n  ── Candidate Evaluation (错综互变) ──")
    for i in range(8):
        name = HEXAGRAM_TRANSFORM_NAMES[i]
        hname = HEXAGRAM_NAMES[cand_idxs[i]] if cand_idxs[i] < 64 else "???"
        marker = " ◀ BEST" if i == best_cand else ""
        print(f"  [{i}] {name:22s} → #{cand_idxs[i]+1:2d} {hname:20s}  Q={cand_values[i]:+.4f}{marker}")

    # Selected action
    action_name = ACTION_NAMES[action_id] if action_id < len(ACTION_NAMES) else f"action_{action_id}"
    print(f"\n  ── Decision ──")
    print(f"  Best hexagram: #{cand_idxs[best_cand]+1} "
          f"{HEXAGRAM_NAMES[cand_idxs[best_cand]]}")
    print(f"  Selected action: [{action_id}] {action_name}")

    return output


def generate_scenario_input(text: str, config: YiCeNetConfig, device: str):
    """Generate tokenized input from a text description using Qwen BPE."""
    input_ids, mask = yicenet_encode(text, max_len=config.max_seq_len)
    return input_ids.to(device), mask.to(device)


def main():
    parser = argparse.ArgumentParser(description="YiCeNet inference demo")
    parser.add_argument("--checkpoint", default="checkpoints/yicenet_rl_best.pt",
                        help="Checkpoint path")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Single scenario to test")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Resolve path
    project_root = Path(__file__).parent.parent
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = project_root / ckpt_path

    print(f"YiCeNet Inference Demo")
    print(f"  Device: {device}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Exists: {ckpt_path.exists()}")
    print()

    # Load model
    config = YiCeNetConfig()
    model = YiCeNet(config).to(device)
    if ckpt_path.exists():
        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state_dict"], strict=False)
        if "tau" in saved:
            model.tau = saved["tau"]
        print(f"  Loaded checkpoint (tau={model.tau:.3f})")
    else:
        print(f"  WARNING: checkpoint not found, using random weights")

    # ── Predefined scenarios ──
    scenarios = [
        "search knowledge base for SAP PM",
        "route to multiple APIs and merge results",
        "parallel data fetch then aggregate",
        "user query classification first",
        "retry with backoff on service failure",
        "orchestrate multi-step approval workflow",
        "cache lookup then generate response",
        "stream response with progress updates",
        "validate input then process",
        "sequential API call chain with error handling",
    ]

    if args.interactive:
        print("\nInteractive mode — type queries (or 'quit'):")
        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text or text.lower() in ("quit", "exit", "q"):
                break
            inp, mask = generate_scenario_input(text, config, device)
            demo_single(model, device, inp, mask, text[:50])

    elif args.scenario:
        inp, mask = generate_scenario_input(args.scenario, config, device)
        demo_single(model, device, inp, mask, args.scenario[:50])

    else:
        # Demo mode: run all scenarios
        print("Running 10 predefined scenarios...\n")
        for i, text in enumerate(scenarios):
            inp, mask = generate_scenario_input(text, config, device)
            demo_single(model, device, inp, mask, text)

    print(f"\n{'='*60}")
    print("  Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
