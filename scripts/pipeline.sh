#!/usr/bin/env bash
# Full training queue on the GPU box. Idempotent and resumable: re-running skips finished work.
#   nohup setsid scripts/pipeline.sh > artifacts/logs/pipeline.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONWARNINGS="ignore::UserWarning" HF_HUB_DISABLE_PROGRESS_BARS=1
ARCHS=${ARCHS:-"resnet18 efficientnet_b0 densenet121 resnet50 vit_small convnext_tiny swin_tiny"}
TRIALS=${TRIALS:-25}
for arch in $ARCHS; do
  echo "=== $(date) HPO $arch"
  python scripts/hpo.py --arch "$arch" --trials "$TRIALS"
  echo "=== $(date) E1/E2 $arch"
  python scripts/run.py --exp E1 E2 --arch "$arch" --seeds 0 1 2
  python scripts/aggregate.py > /dev/null
done
echo "=== $(date) pipeline stage 1 done"
