#!/bin/bash
# YiCeNet full training pipeline
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

DEVICE="${1:-auto}"
MODEL="${2:-full}"  # full, tiny, or micro

echo "=========================================="
echo " YiCeNet Training Pipeline"
echo " Device: $DEVICE"
echo " Model:  $MODEL"
echo "=========================================="

# 0. Dry-run test
echo ""
echo "--- Step 0: Dry-run test ---"
python src/train.py --stage dry-run --device "$DEVICE"

# 1. Pre-train (unsupervised clustering)
echo ""
echo "--- Step 1: Unsupervised pre-training ---"
python src/train.py \
    --stage pretrain \
    --num_samples 10000 \
    --batch_size 128 \
    --pretrain_epochs 50 \
    --lr 1e-3 \
    --device "$DEVICE" \
    --checkpoint_dir checkpoints

# 2. RL fine-tune
echo ""
echo "--- Step 2: RL fine-tuning ---"
python src/train.py \
    --stage rl \
    --episodes 5000 \
    --batch_size 64 \
    --lr 3e-4 \
    --device "$DEVICE" \
    --checkpoint_dir checkpoints

echo ""
echo "=========================================="
echo " Training complete!"
echo " Checkpoints saved to: $PROJECT_DIR/checkpoints/"
echo "=========================================="
