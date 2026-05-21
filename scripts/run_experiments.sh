#!/bin/bash
# Run experiments with different configurations
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

DEVICE="${1:-auto}"

echo "YiCeNet Experiments"
echo "Device: $DEVICE"
echo ""

# Quick dry-run test
echo "=== Quick test ==="
python src/train.py --stage dry-run --device "$DEVICE"

# Ablation: different temperature schedules
for tau_init in 1.0 2.0 0.5; do
    echo ""
    echo "=== RL with tau_init=$tau_init ==="
    python src/train.py \
        --stage rl \
        --episodes 2000 \
        --batch_size 64 \
        --lr 3e-4 \
        --device "$DEVICE" \
        --checkpoint_dir "checkpoints/tau_${tau_init}"
done

echo ""
echo "All experiments complete!"
