#!/usr/bin/env bash
# Counterfactual pairs (scripts/synth_cf/cf_generate.py, docs/RESEARCH_PIVOT.md §7): E3 / E4 with lung-only edits of real
# Kermany images, strength 1.0 and 0.8 (tags CF_S10 / CF_S08). Waits for the generation and the mimic-dose queue.
cd "$(dirname "$0")/.."
source .venv/bin/activate
# match only the worker processes: the tmux server keeps the launching command line (which names the scripts)
while pgrep -f "^\.venv-gen/bin/python scripts/synth_cf/cf_generate.py|^bash .*scripts/queue_mimic_dose.sh" > /dev/null; do
  sleep 60
done
for arch in efficientnet_b0 resnet18 densenet121; do
  for s in 10 08; do
    python scripts/run.py --exp E3 E4 --arch $arch --synthetic artifacts/manifests/synthetic_cf_lung_s$s.csv \
      --tag CF_S$s || echo "FAILED: $arch s$s"
  done
done
python scripts/aggregate.py > /dev/null
echo "queue_cf done"
