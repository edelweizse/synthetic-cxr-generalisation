#!/usr/bin/env bash
# (1) SYN_ONLY v2 without any real image in the classifier pipeline: validation (early stopping, checkpoint, threshold)
#     on 250 + 250 held-out synthetic images (sampler seed 1, disjoint from the training samples). Tag SD2V.
# (2) CFG sweep on the v2 adapter, sampling only: 100 per class at CFG 1.5 / 2.5 / 3.5 (same seeds as the first 100 of
#     cx_sd_v2, which used 4.5), then the generator evaluation (memory-capped).
cd "$(dirname "$0")/.."
G=data/generator; M=artifacts/manifests; ARCHS="efficientnet_b0 resnet18 densenet121"
HF_HUB_OFFLINE=1 .venv-gen/bin/python scripts/generator/sample.py --adapter $G/lora_cx_v2 --name cx_sd_v2_val --per-class 250 --seed 1 || { echo "FAILED: sample val"; exit 1; }
[ -f $M/synthetic_cx_sd_v2_val.csv ] || .venv/bin/python scripts/prepare_synthetic.py data/synthetic/cx_sd_v2_val --name cx_sd_v2_val || { echo "FAILED: register val"; exit 1; }
.venv/bin/python scripts/run.py --exp SYN_ONLY --arch $ARCHS --source chexpert --synthetic $M/synthetic_cx_sd_v2.csv \
  --val $M/synthetic_cx_sd_v2_val.csv --tag SD2V || echo "FAILED: SYN_ONLY SD2V"
for c in 15 25 35; do
  HF_HUB_OFFLINE=1 .venv-gen/bin/python scripts/generator/sample.py --adapter $G/lora_cx_v2 --name cx_sd_v2_cfg$c --per-class 100 --cfg ${c:0:1}.${c:1:1} || { echo "FAILED: sample cfg $c"; exit 1; }
  [ -f $M/synthetic_cx_sd_v2_cfg$c.csv ] || .venv/bin/python scripts/prepare_synthetic.py data/synthetic/cx_sd_v2_cfg$c --name cx_sd_v2_cfg$c || { echo "FAILED: register cfg $c"; exit 1; }
done
# memory cap when the user systemd instance is up (no lingering: it may not be); the GPU is idle at this point anyway
systemctl --user show-environment > /dev/null 2>&1 && CAP="systemd-run --user --scope -q -p MemoryMax=8G" || CAP=""
[ -f reports/results/generator/eval_chexpert_cfg.csv ] || CUDA_VISIBLE_DEVICES= $CAP \
  .venv/bin/python scripts/generator/evaluate.py chexpert cx_sd_v2 cx_sd_v2_cfg15 cx_sd_v2_cfg25 cx_sd_v2_cfg35 --out=eval_chexpert_cfg || echo "FAILED: evaluate cfg"
.venv/bin/python scripts/aggregate.py > /dev/null
echo "queue_gen_v2_val done"
