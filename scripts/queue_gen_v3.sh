#!/usr/bin/env bash
# Generator v3 (docs/RESEARCH_PIVOT.md §13): the v2 recipe (LoRA r16, lr 1e-4, 7,500 steps, CFG 2.5) with rich captions
# (view, sex, age, all report findings) instead of two class prompts; sampled with the training captions of each class
# (seed 0: 1,000 per class for training, seed 1: 250 per class for validation); fully synthetic SYN_ONLY (tag SD3V,
# compare SD2C25V) and the generator evaluation. Runs under nice: the box is also used interactively.
cd "$(dirname "$0")/.."
G=data/generator; M=artifacts/manifests; ARCHS="efficientnet_b0 resnet18 densenet121"; CAP=$G/chexpert/metadata_rich.jsonl
if [ ! -f $G/lora_cx_v3/adapter_model.safetensors ]; then
  echo "=== $(date) LoRA v3 training"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 .venv-gen/bin/python scripts/generator/train_lora.py \
    --metadata_path $CAP --output_dir $G/lora_cx_v3 --variant fp16 --seed 0 \
    --lr 1e-4 --max_train_steps 7500 || { echo "FAILED: train_lora"; exit 1; }
fi
for spec in "cx_sd_v3 1000 0" "cx_sd_v3_val 250 1"; do
  set -- $spec
  HF_HUB_OFFLINE=1 .venv-gen/bin/python scripts/generator/sample.py --adapter $G/lora_cx_v3 --name $1 --per-class $2 --seed $3 \
    --cfg 2.5 --captions $CAP || { echo "FAILED: sample $1"; exit 1; }
  [ -f $M/synthetic_$1.csv ] || .venv/bin/python scripts/prepare_synthetic.py data/synthetic/$1 --name $1 || { echo "FAILED: register $1"; exit 1; }
done
.venv/bin/python scripts/run.py --exp SYN_ONLY --arch $ARCHS --source chexpert --synthetic $M/synthetic_cx_sd_v3.csv \
  --val $M/synthetic_cx_sd_v3_val.csv --tag SD3V || echo "FAILED: SYN_ONLY SD3V"
[ -f reports/results/generator/eval_chexpert_v3.csv ] || CUDA_VISIBLE_DEVICES= \
  .venv/bin/python scripts/generator/evaluate.py chexpert cx_sd_v3 --out=eval_chexpert_v3 || echo "FAILED: evaluate v3"
echo "queue_gen_v3 done"
