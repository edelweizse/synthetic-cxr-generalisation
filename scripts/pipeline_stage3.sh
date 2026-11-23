#!/usr/bin/env bash
# Stage 3: anti-overfitting variants built on E2 (augmentation types, mixup, linear probe).
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONWARNINGS="ignore::UserWarning" HF_HUB_DISABLE_PROGRESS_BARS=1
ARCHS=${ARCHS:-"efficientnet_b0 vit_small densenet121"}
for arch in $ARCHS; do
  python scripts/run.py --exp E2_ACQ E2_ERASE E2_ROBUST E2_MIX LP --arch "$arch" --seeds 0 1 2
done
python scripts/aggregate.py > /dev/null
echo "=== $(date) pipeline stage 3 done"
