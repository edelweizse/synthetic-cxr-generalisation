#!/usr/bin/env bash
# After queue_gen_v2.sh: 100 samples per class from the intermediate v2 adapters (steps 2,500 / 5,000; same seeds as the
# full sets), then the generator evaluation of every CheXpert set (scripts/generator/evaluate.py).
cd "$(dirname "$0")/.."
while tmux has-session -t =gen2 2>/dev/null; do sleep 60; done
G=data/generator; M=artifacts/manifests
for s in 2500 5000; do
  HF_HUB_OFFLINE=1 .venv-gen/bin/python scripts/generator/sample.py --adapter $G/lora_cx_v2/step_$s --name cx_sd_v2_s$s --per-class 100 || { echo "FAILED: sample $s"; exit 1; }
  [ -f $M/synthetic_cx_sd_v2_s$s.csv ] || .venv/bin/python scripts/prepare_synthetic.py data/synthetic/cx_sd_v2_s$s --name cx_sd_v2_s$s || { echo "FAILED: register $s"; exit 1; }
done
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/generator/evaluate.py chexpert cx_sd cx_sd_v2_s2500 cx_sd_v2_s5000 cx_sd_v2 || { echo "FAILED: evaluate"; exit 1; }
echo "queue_gen_v2_eval done"
