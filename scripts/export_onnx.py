#!/usr/bin/env python3
"""
Export YiCeNet trained checkpoint to ONNX format.
Tests loading via ONNX Runtime for Plan B verification.
"""
import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import onnxruntime as ort
import numpy as np

from src.model import YiCeNet
from src.config import YiCeNetConfig

CKPT = "checkpoints/yicenet_rl_best.pt"
ONNX_PATH = "checkpoints/yicenet.onnx"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"YiCeNet ONNX Export")
print(f"  Device: {DEVICE}")
print(f"  Checkpoint: {CKPT}")

# 1. Load model
config = YiCeNetConfig()
model = YiCeNet(config).to(DEVICE).eval()
saved = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(saved["model_state_dict"], strict=False)
print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# 2. Export to ONNX
B, T = 1, 16
dummy_input = torch.randint(1, config.vocab_size, (B, T)).to(DEVICE)
dummy_mask = torch.ones(B, T).to(DEVICE)

torch.onnx.export(
    model,
    (dummy_input, dummy_mask),
    ONNX_PATH,
    input_names=["input_ids", "attention_mask"],
    output_names=[
        "h", "hexagram_idx", "hexagram_probs",
        "best_candidate_idx", "candidate_idxs",
        "candidate_values", "action_ids", "action_logits",
    ],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
    },
    opset_version=17,
    do_constant_folding=True,
)
print(f"  ONNX exported: {ONNX_PATH}")
print(f"  File size: {os.path.getsize(ONNX_PATH) / 1024 / 1024:.1f} MB")

# 3. Verify with ONNX Runtime
print(f"\n  ── ONNX Runtime Verification ──")
session = ort.InferenceSession(ONNX_PATH)
input_name_ids = session.get_inputs()[0].name
input_name_mask = session.get_inputs()[1].name

# Run inference
ort_inputs = {
    input_name_ids: dummy_input.cpu().numpy().astype(np.int64),
    input_name_mask: dummy_mask.cpu().numpy().astype(np.int64),
}
ort_outs = session.run(None, ort_inputs)

output_names = [
    "h", "hexagram_idx", "hexagram_probs",
    "best_candidate_idx", "candidate_idxs",
    "candidate_values", "action_ids", "action_logits"
]
print(f"  Outputs:")
for name, arr in zip(output_names, ort_outs):
    shape = arr.shape
    dtype = arr.dtype
    if np.prod(shape) <= 10:
        val = arr.flatten().tolist()
        print(f"    {name:25s}  {str(shape):18s}  {dtype!s:10s}  {val}")
    else:
        print(f"    {name:25s}  {str(shape):18s}  {dtype!s:10s}")

# 4. Benchmark
import time
warmup = 10
trials = 100
for _ in range(warmup):
    session.run(None, ort_inputs)

start = time.perf_counter()
for _ in range(trials):
    session.run(None, ort_inputs)
elapsed = (time.perf_counter() - start) / trials

print(f"\n  ── Benchmark ──")
print(f"  Avg inference: {elapsed*1000:.2f} ms")
print(f"  Throughput:    {1/elapsed:.0f} inferences/sec")
print(f"\n  ✓ ONNX Runtime works! Plan B is feasible.")
