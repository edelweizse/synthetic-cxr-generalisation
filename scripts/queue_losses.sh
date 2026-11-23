#!/usr/bin/env bash
# Robust training losses (cxr/engine.LOSSES) on E1 (real only) and E3 (+250 synthetic), EfficientNet-B0, 3 seeds.
# Hyper-parameters stay the BCE-tuned ones. Waits for queue_quality (tmux "qual") so the GPU is not shared.
cd "$(dirname "$0")/.."
source .venv/bin/activate
while tmux has -t =qual 2>/dev/null; do sleep 60; done
S=artifacts/manifests/synthetic_sd_lora.csv
for loss in qcons qdro gce focal; do
  T=$(echo $loss | tr a-z A-Z)
  python scripts/run.py --exp E1 --arch efficientnet_b0 --loss $loss --tag L$T || echo "FAILED: E1 $loss"
  python scripts/run.py --exp E3 --arch efficientnet_b0 --loss $loss --synthetic $S --tag sd_lora_L$T || echo "FAILED: E3 $loss"
done
python scripts/aggregate.py > /dev/null
echo "queue_losses done"
