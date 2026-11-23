#!/usr/bin/env bash
# Quality-aware synthetic selection (scripts/synth_probe/quality_select.py): does choosing WHICH synthetic images to add
# remove the penalty? Waits for pipeline_synth (tmux "synth") so the GPU is not shared.
cd "$(dirname "$0")/.."
source .venv/bin/activate
while tmux has -t =synth 2>/dev/null; do sleep 60; done
for sel in qm qc; do
  TAG=$(echo $sel | tr a-z A-Z)
  python scripts/run.py --exp E3 E4 --arch efficientnet_b0 --seeds 0 1 2 \
    --synthetic artifacts/manifests/synthetic_sd_lora_$sel.csv --tag sd_lora_$TAG || echo "FAILED: $sel"
done
python scripts/aggregate.py > /dev/null
echo "queue_quality done"
