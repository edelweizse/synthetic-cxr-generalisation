#!/usr/bin/env bash
# Quality-harmonised preprocessing (histogram equalisation, `harmonize224`): does removing the contrast
# shortcut help transfer, and does it stop the SD-LoRA synthetic images from hurting? EfficientNet-B0, 3 seeds.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONWARNINGS="ignore::UserWarning" HF_HUB_DISABLE_PROGRESS_BARS=1
ARCH=${ARCH:-efficientnet_b0}
python scripts/run.py --exp E1 E2 E2_ERASE --arch "$ARCH" --resize-mode harmonize224 --tag HARM
python scripts/run.py --exp E3 E4 --arch "$ARCH" --resize-mode harmonize224 --tag sd_lora_HARM \
  --synthetic artifacts/manifests/synthetic_sd_lora.csv
echo "=== $(date) pipeline_harm done"
