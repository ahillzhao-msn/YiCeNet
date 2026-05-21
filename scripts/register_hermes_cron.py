"""
Register YiCeNet self-training pipeline as a Hermes cron job.

Run this once to set up the auto-training schedule:
    python scripts/register_hermes_cron.py

This creates two cron jobs:
  1. check-training — every 2 hours: check if ≥500 new trajectories, train if so
  2. daily-report   — every day at 8am: model health summary
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Cron job configurations ──

CHECK_TRAINING_JOB = {
    "name": "yicenet-check-training",
    "schedule": "every 2h",
    "prompt": """
Run the YiCeNet training pipeline:

1. Check the trajectory count in ~/YiCeNet/data/metrics.db
2. If 500+ new trajectories since last training, run:
   python3 ~/YiCeNet/scripts/training_worker.py --once
3. Check registry.json (~/YiCeNet/checkpoints/registry.json) for a 'ready' model
4. If ready model's win_rate > active model's win_rate + 5% threshold:
   - Load the ready model via yicenet_switch(checkpoint=ready_path)
   - Update registry.json: promote ready to active, old active to fallback
5. Report: whether training ran, whether switch occurred, key metrics

Use terminal to run commands. Read registry.json and metrics.db to determine status.
Be concise — one paragraph summary.
""".strip(),
    "workdir": str(Path.home() / "YiCeNet"),
}

DAILY_REPORT_JOB = {
    "name": "yicenet-daily-report",
    "schedule": "0 8 * * *",
    "prompt": """
Generate a daily health report for YiCeNet (易策网络):

1. Read ~/YiCeNet/checkpoints/registry.json — active model version & metrics
2. Read ~/YiCeNet/data/metrics.db — trajectory count, success rate, abandon rate
3. Run: python3 ~/YiCeNet/scripts/training_worker.py --evaluate
4. Read ~/YiCeNet/checkpoints/registry.json again for any ready model

Output:
- Active model version & avg reward
- 24h stats: trajectories / success rate / abandon rate / top 3 hexagrams
- Training status: last training run, any model ready to switch
- Recommendation: switch or hold
- Tai Chi balance: yang% / yin% (from last 100 trajectories)

Use terminal to read files. Be informative but concise — bullet points.
""".strip(),
    "workdir": str(Path.home() / "YiCeNet"),
}


def register_job(job_def: dict):
    """
    Print instructions for registering the cron job.
    Hermes cron jobs are managed via the cronjob tool.
    """
    print(f"\n{'='*50}")
    print(f"Job: {job_def['name']}")
    print(f"  Schedule: {job_def['schedule']}")
    print(f"  Workdir:  {job_def['workdir']}")
    print(f"{'='*50}")
    print(f"\nRegister via Hermes cronjob tool:\n")
    print(f"  cronjob(action='create',")
    print(f"          name='{job_def['name']}',")
    print(f"          schedule='{job_def['schedule']}',")
    print(f"          prompt='''{job_def['prompt'][:80]}...''',")
    print(f"          workdir='{job_def['workdir']}',")
    print(f"          enabled_toolsets=['terminal', 'file'])")
    print()


def register_all():
    print("YiCeNet Hermes Cron Registration")
    print("=" * 50)
    print()
    print("Run these commands inside a Hermes session to register the cron jobs:\n")

    for j in [CHECK_TRAINING_JOB, DAILY_REPORT_JOB]:
        register_job(j)

    print(f"\nAfter registration, verify with:\n")
    print(f"  cronjob(action='list')")
    print()
    print(f"To trigger immediately:\n")
    print(f"  cronjob(action='run', job_id='<id>')")


if __name__ == "__main__":
    register_all()
