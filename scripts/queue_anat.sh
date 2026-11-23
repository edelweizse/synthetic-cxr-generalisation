#!/usr/bin/env bash
# Anatomy arm (docs/RESEARCH_PIVOT.md arm C; scripts/synth_probe/anatomy.py + anatomy_select.py select 0.2):
#   ANAT     synthetic images with the most atypical 20% per class (segmented anatomy vs real Kermany) removed
#   ANATBAD  most atypical synthetic images first
# Both keep the synthetic quality shortcut (low-contrast AUROC ~0.90), so a difference is anatomy, not the shortcut.
# Waits for queue_mimic so the GPU is not shared.
cd "$(dirname "$0")/.."
source .venv/bin/activate
# match only the bash processes: the tmux server keeps the launching command line (which names the script) for its lifetime
while pgrep -f "^bash .*scripts/queue_(gaps|mimic).sh" > /dev/null; do sleep 60; done
for arch in efficientnet_b0 resnet18 densenet121; do
  for sel in anat:ANAT anatbad:ANATBAD; do
    python scripts/run.py --exp E3 E4 --arch $arch --synthetic artifacts/manifests/synthetic_sd_lora_${sel%%:*}.csv \
      --tag sd_lora_${sel##*:} || echo "FAILED: $arch $sel"
  done
done
python scripts/aggregate.py > /dev/null
echo "queue_anat done"
