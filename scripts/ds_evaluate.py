#!/usr/bin/env python3
"""
Batch evaluation script for YiCeNet training.
Reads samples from file, sends to DeepSeek API for evaluation,
saves results as JSONL.
"""
import json
import os
import time
import sys
from pathlib import Path

# Load API key from Hermes .env
ENV_PATH = Path.home() / ".hermes" / ".env"
api_key = None
if ENV_PATH.exists():
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY="):
                api_key = line.split("=", 1)[1].strip("'\"")
                break

if not api_key:
    sys.exit("ERROR: DEEPSEEK_API_KEY not found")

API_URL = "https://api.deepseek.com/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

def evaluate_batch(samples, batch_id):
    """Evaluate a batch of samples via DeepSeek API."""
    # Build evaluation prompt
    sample_lines = []
    for i, s in enumerate(samples):
        text = s["user_text"][:300]  # Truncate to 300 chars for API efficiency
        sample_lines.append(f"[{i}] msg_id={s['msg_id']}: {text}")

    prompt = (
        "You are evaluating conversation quality for YiCeNet training. "
        "For each sample below, assess the interaction:\n"
        "- satisfaction_score: -1.0 to 1.0 (fine-grained)\n"
        "  * +0.5 to +1.0: praising, enthusiastic, satisfied\n"
        "  * +0.1 to +0.4: neutral regular request\n"
        "  * -0.1 to -0.4: correction, frustration, disagreement\n"
        "  * -0.5 to -1.0: strong criticism, abandonment\n"
        "- signals: which of {continued, corrected, completed, praised, abandoned} are True\n"
        "- reasoning: one short sentence\n\n"
        + "\n\n".join(sample_lines) +
        "\n\nReturn ONLY a valid JSON array. No other text.\n"
        "Format: [{\"msg_id\": N, \"satisfaction\": 0.3, \"signals\": {\"continued\": false, \"corrected\": false, \"completed\": false, \"praised\": false, \"abandoned\": false}, \"reasoning\": \"...\"}]\n"
    )

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 4096
    }

    import urllib.request
    import urllib.error

    data = json.dumps(payload).encode()
    req = urllib.request.Request(API_URL, data=data, headers=HEADERS, method="POST")

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
            content = result["choices"][0]["message"]["content"]

            # Extract JSON array from response
            content = content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1]
            if content.startswith("```"):
                content = content.split("```")[1]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
            content = content.strip()

            result_data = json.loads(content)

            # Validate and merge with original msg_id
            validated = []
            for item in result_data:
                if isinstance(item, dict) and "msg_id" in item:
                    validated.append(item)
            return validated

        except (json.JSONDecodeError, KeyError, urllib.error.HTTPError, urllib.error.URLError,
                TimeoutError, OSError) as e:
            print(f"  Attempt {attempt+1} failed: {e}", file=sys.stderr)
            if attempt < 2:
                time.sleep(5 * (attempt + 1))

    return []

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    args = parser.parse_args()

    # Load samples
    samples = []
    with open(args.input) as f:
        for line in f:
            samples.append(json.loads(line))

    if args.end_idx:
        samples = samples[args.start_idx:args.end_idx]
    elif args.start_idx:
        samples = samples[args.start_idx:]

    print(f"Total: {len(samples)} samples, batch_size={args.batch_size}", file=sys.stderr)

    all_results = []

    for i in range(0, len(samples), args.batch_size):
        batch = samples[i:i+args.batch_size]
        batch_id = i // args.batch_size
        total_batches = (len(samples) + args.batch_size - 1) // args.batch_size
        print(f"  Batch {batch_id+1}/{total_batches} ({len(batch)} samples)...", file=sys.stderr)

        results = evaluate_batch(batch, batch_id)
        all_results.extend(results)
        print(f"    Got {len(results)} evaluations", file=sys.stderr)

        # Rate limiting: 2 batches per second
        if i + args.batch_size < len(samples):
            time.sleep(0.5)

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved {len(all_results)} evaluations to {args.output}", file=sys.stderr)
    print(f"Stats: mean_satisfaction={sum(r['satisfaction'] for r in all_results)/len(all_results):+.3f}")

if __name__ == "__main__":
    main()
