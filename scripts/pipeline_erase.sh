#!/usr/bin/env bash
# Synthetic data and real-data controls on top of the strongest AUGMENTATION (E2_ERASE, selected on B-dev).
#   ERASE_SYN250/500  = E2_ERASE augmentation + 250/500 SYNTHETIC images  (E3/E4 on the strong baseline)
#   ERASE_REAL250/500 = E2_ERASE augmentation + 250/500 more REAL images  (R25/R50 on the strong baseline)
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONWARNINGS="ignore::UserWarning" HF_HUB_DISABLE_PROGRESS_BARS=1
ARCHS=${ARCHS:-"efficientnet_b0 vit_small densenet121"}
for arch in $ARCHS; do
  python scripts/run.py --exp ERASE_REAL250 ERASE_REAL500 --arch "$arch"
  python scripts/run.py --exp ERASE_SYN250 ERASE_SYN500 --arch "$arch" \
    --synthetic artifacts/manifests/synthetic_sd_lora.csv --tag sd_lora
done
echo "=== $(date) pipeline_erase done"
