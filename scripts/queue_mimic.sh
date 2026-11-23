#!/usr/bin/env bash
# Real-image mimics of the synthetic set (docs/RESEARCH_PIVOT.md §3, scripts/synth_probe/mimic_select.py): which property
# of the synthetic images causes the penalty? Each manifest fills the E3 (+250) / E4 (+500) slot with REAL Kermany images:
#   MIMIC_Q   synthetic-strength quality shortcut      MIMIC_N   30% of pneumonia slots = mislabelled normals (weak signal)
#   MIMIC_QN  both
# Compare with E3/E4_sd_lora (synthetic) and R25/R50 (random real). Waits for the gap queue so the GPU is not shared.
cd "$(dirname "$0")/.."
source .venv/bin/activate
# match only the bash processes: the tmux server keeps the launching command line (which names the script) for its lifetime
while pgrep -f "^bash .*scripts/queue_gaps.sh" > /dev/null; do sleep 60; done
for arch in efficientnet_b0 resnet18 densenet121; do
  for m in quality:Q noise:N both:QN; do
    python scripts/run.py --exp E3 E4 --arch $arch --synthetic artifacts/manifests/mimic_${m%%:*}.csv \
      --tag MIMIC_${m##*:} || echo "FAILED: $arch $m"
  done
done
python scripts/aggregate.py > /dev/null
echo "queue_mimic done"
