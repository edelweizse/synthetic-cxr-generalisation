#!/usr/bin/env bash
# CFG 2.5 won the sweep (same class signal as 4.5, better realism / coverage): full sets at CFG 2.5 from the v2 adapter
# (2,000 training samples with seed 0 = the cx_sd_v2 seeds, 500 validation samples with seed 1), then the fully synthetic
# pipeline SYN_ONLY + synthetic validation (tag SD2C25V; compare SD2V = CFG 4.5), then the generator evaluation of the set.
cd "$(dirname "$0")/.."
G=data/generator; M=artifacts/manifests; ARCHS="efficientnet_b0 resnet18 densenet121"
for spec in "cx_sd_v2c25 1000 0" "cx_sd_v2c25_val 250 1"; do
  set -- $spec
  HF_HUB_OFFLINE=1 .venv-gen/bin/python scripts/generator/sample.py --adapter $G/lora_cx_v2 --name $1 --per-class $2 --seed $3 --cfg 2.5 || { echo "FAILED: sample $1"; exit 1; }
  [ -f $M/synthetic_$1.csv ] || .venv/bin/python scripts/prepare_synthetic.py data/synthetic/$1 --name $1 || { echo "FAILED: register $1"; exit 1; }
done
.venv/bin/python scripts/run.py --exp SYN_ONLY --arch $ARCHS --source chexpert --synthetic $M/synthetic_cx_sd_v2c25.csv \
  --val $M/synthetic_cx_sd_v2c25_val.csv --tag SD2C25V || echo "FAILED: SYN_ONLY SD2C25V"
systemctl --user show-environment > /dev/null 2>&1 && CAP="systemd-run --user --scope -q -p MemoryMax=8G" || CAP=""
[ -f reports/results/generator/eval_chexpert_c25.csv ] || CUDA_VISIBLE_DEVICES= $CAP \
  .venv/bin/python scripts/generator/evaluate.py chexpert cx_sd_v2c25 --out=eval_chexpert_c25 || echo "FAILED: evaluate c25"
.venv/bin/python scripts/aggregate.py > /dev/null
echo "queue_gen_v2_cfg25 done"
