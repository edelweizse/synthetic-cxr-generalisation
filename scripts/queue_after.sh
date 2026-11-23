#!/usr/bin/env bash
# Runs once scripts/queue_remaining.sh has finished: harmonised ERASE_SYN follow-up, aggregate, notebooks 06 and 07 (CPU).
set -u
cd "$(dirname "$0")/.."
until grep -q "^queue done" artifacts/logs/queue.log 2>/dev/null; do sleep 300; done
source .venv/bin/activate
# follow-up: does harmonised preprocessing also rescue synthetic data on the augmented baseline?
python scripts/run.py --exp ERASE_SYN250 ERASE_SYN500 --arch efficientnet_b0 --resize-mode harmonize224 \
  --synthetic artifacts/manifests/synthetic_sd_lora.csv --tag sd_lora_HARM
# dose-response: much more synthetic data, and synthetic data only (EfficientNet-B0 x 3 seeds, tag sd_lora)
python scripts/run.py --exp SYN1000 SYN2000 SYN_ONLY --arch efficientnet_b0 \
  --synthetic artifacts/manifests/synthetic_sd_lora.csv --tag sd_lora
python scripts/aggregate.py > /dev/null
CUDA_VISIBLE_DEVICES="" scripts/run_notebook.sh 06_overfitting
CUDA_VISIBLE_DEVICES="" scripts/run_notebook.sh 07_synthetic
echo "queue_after done"
