#!/usr/bin/env bash
# Scaling round (docs/RESEARCH_PIVOT.md §12), frozen generator (v2 adapter, CFG 2.5). 1) real reference curve on CheXpert
# (250, 500, all 9,968 = what the generator saw); 2) scarce real data + 2,000 synthetic (prompt labels, real validation):
# 250 real (vs CX_N250) and 1,000 real (vs CX_E1); 3) 10,000 synthetic images (seed 0; the 2,000 of cx_sd_v2c25 are the
# same seeds and are hard-linked, not regenerated), SYN_ONLY with 5,000 / 10,000 + synthetic validation (compare SD2C25V =
# 2,000); 4) generator evaluation of the 10k set (memorisation over all 10,000).
cd "$(dirname "$0")/.."
G=data/generator; M=artifacts/manifests; S=data/synthetic; ARCHS="efficientnet_b0 resnet18 densenet121"
.venv/bin/python scripts/run.py --exp N250 N500 FULL --arch $ARCHS --source chexpert || echo "FAILED: real curve"
.venv/bin/python scripts/run.py --exp N250_SYN2000 SYN2000 --arch $ARCHS --source chexpert \
  --synthetic $M/synthetic_cx_sd_v2c25.csv --tag SD2C25 || echo "FAILED: mixed"
if [ ! -f $M/synthetic_cx_sd_v2c25_10k.csv ]; then
  mkdir -p $S/cx_sd_v2c25_10k
  for c in NORMAL PNEUMONIA; do [ -d $S/cx_sd_v2c25_10k/$c ] || cp -al $S/cx_sd_v2c25/$c $S/cx_sd_v2c25_10k/; done  # PNGs only
  HF_HUB_OFFLINE=1 .venv-gen/bin/python scripts/generator/sample.py --adapter $G/lora_cx_v2 --name cx_sd_v2c25_10k \
    --per-class 5000 --seed 0 --cfg 2.5 || { echo "FAILED: sample 10k"; exit 1; }
  .venv/bin/python scripts/prepare_synthetic.py $S/cx_sd_v2c25_10k --name cx_sd_v2c25_10k || { echo "FAILED: register 10k"; exit 1; }
fi
.venv/bin/python scripts/run.py --exp SYN_ONLY5K SYN_ONLY10K --arch $ARCHS --source chexpert \
  --synthetic $M/synthetic_cx_sd_v2c25_10k.csv --val $M/synthetic_cx_sd_v2c25_val.csv --tag SD2C25V || echo "FAILED: SYN_ONLY 5k/10k"
systemctl --user show-environment > /dev/null 2>&1 && CAP="systemd-run --user --scope -q -p MemoryMax=16G" || CAP=""
[ -f reports/results/generator/eval_chexpert_10k.csv ] || CUDA_VISIBLE_DEVICES= $CAP \
  .venv/bin/python scripts/generator/evaluate.py chexpert cx_sd_v2c25_10k --out=eval_chexpert_10k || echo "FAILED: evaluate 10k"
.venv/bin/python scripts/aggregate.py > /dev/null
echo "queue_gen_scale done"
