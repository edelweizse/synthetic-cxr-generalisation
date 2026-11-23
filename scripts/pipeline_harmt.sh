#!/usr/bin/env bash
# Harmonised preprocessing with its OWN hyper-parameters: re-run the Optuna search (same space, same A-validation
# objective) under harmonize224, then E1 / E2_ERASE / E3 / E4 on EfficientNet-B0 x 3 seeds (tag HARMT).
# Answers whether the HARM runs (which reuse the squash-tuned parameters) were held back by mismatched tuning.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=harmonize224
python scripts/hpo.py --arch efficientnet_b0 --resize-mode $M --trials 25 || { echo "FAILED: hpo"; exit 1; }
python scripts/run.py --exp E1 E2_ERASE --arch efficientnet_b0 --resize-mode $M --hpo-variant $M --tag HARMT
python scripts/run.py --exp E3 E4 --arch efficientnet_b0 --resize-mode $M --hpo-variant $M \
  --synthetic artifacts/manifests/synthetic_sd_lora.csv --tag sd_lora_HARMT
echo "pipeline_harmt done"
