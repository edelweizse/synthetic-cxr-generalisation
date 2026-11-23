#!/usr/bin/env bash
# Stage 3 (time-boxed): two architectures in parallel on one GPU, seeds 0-1 first.
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONWARNINGS="ignore::UserWarning" HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
EXPS="E2_ACQ E2_ERASE E2_MIX LP E2_ROBUST"
python scripts/run.py --exp $EXPS --arch efficientnet_b0 --seeds 0 1 --n-boot 200 > artifacts/logs/stage3_effnet.log 2>&1 &
python scripts/run.py --exp $EXPS --arch vit_small --seeds 0 1 --n-boot 200 > artifacts/logs/stage3_vit.log 2>&1 &
wait
echo "=== $(date) stage 3 fast done" >> artifacts/logs/stage3_fast.log
