#!/usr/bin/env bash
# YiCeNet Flywheel — Hermes cron wrapper
# Runs model update training, logs to /home/ahill/YiCeNet/logs/

cd /home/ahill/YiCeNet
python3 -m yicenet.flywheel >> logs/flywheel.log 2>&1
echo "flywheel: done at $(date)"
