#!/usr/bin/env bash
# Stronger weak-signal doses for the real-image mimics (scripts/synth_probe/mimic_select.py 0.5 / 0.7): 50% / 70% of the
# pneumonia slots hold a mislabelled normal (MIMIC_N50/N70), alone or with the synthetic quality profile (MIMIC_QN50/QN70).
# EfficientNet-B0 first, then ResNet-18 and DenseNet-121.
cd "$(dirname "$0")/.."
source .venv/bin/activate
for arch in efficientnet_b0 resnet18 densenet121; do
  for m in noise50:N50 noise70:N70 both50:QN50 both70:QN70; do
    python scripts/run.py --exp E3 E4 --arch $arch --synthetic artifacts/manifests/mimic_${m%%:*}.csv \
      --tag MIMIC_${m##*:} || echo "FAILED: $arch $m"
  done
done
python scripts/aggregate.py > /dev/null
echo "queue_mimic_dose done"
