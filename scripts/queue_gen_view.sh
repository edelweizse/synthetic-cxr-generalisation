#!/usr/bin/env bash
# De-confounding (docs/RESEARCH_PIVOT.md §13): the v2 recipe (r16, lr 1e-4, 7,500 steps, CFG 2.5) with view in the caption
# (prepare_chexpert.py view), sampled 50/50 AP/PA per class (--balance-view); 1,000 per class (seed 0) + 250 per class
# validation (seed 1); SYN_ONLY tag SDVIEWV vs SD2C25V.
cd "$(dirname "$0")/.."
G=data/generator; M=artifacts/manifests; ARCHS="efficientnet_b0 resnet18 densenet121"; CAP=$G/chexpert/metadata_view.jsonl
[ -f $CAP ] || .venv/bin/python scripts/generator/prepare_chexpert.py view || { echo "FAILED: captions"; exit 1; }
if [ ! -f $G/lora_cx_view/adapter_model.safetensors ]; then
  echo "=== $(date) LoRA view training"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 .venv-gen/bin/python scripts/generator/train_lora.py \
    --metadata_path $CAP --output_dir $G/lora_cx_view --variant fp16 --seed 0 \
    --lr 1e-4 --max_train_steps 7500 --lora_r 16 --lora_alpha 16 || { echo "FAILED: train_lora"; exit 1; }
fi
for spec in "cx_sd_view 1000 0" "cx_sd_view_val 250 1"; do
  set -- $spec
  HF_HUB_OFFLINE=1 .venv-gen/bin/python scripts/generator/sample.py --adapter $G/lora_cx_view --name $1 --per-class $2 --seed $3 \
    --cfg 2.5 --balance-view || { echo "FAILED: sample $1"; exit 1; }
  [ -f $M/synthetic_$1.csv ] || .venv/bin/python scripts/prepare_synthetic.py data/synthetic/$1 --name $1 || { echo "FAILED: register $1"; exit 1; }
done
.venv/bin/python scripts/run.py --exp SYN_ONLY --arch $ARCHS --source chexpert --synthetic $M/synthetic_cx_sd_view.csv \
  --val $M/synthetic_cx_sd_view_val.csv --tag SDVIEWV || echo "FAILED: SYN_ONLY SDVIEWV"
[ -f reports/results/generator/eval_chexpert_view.csv ] || CUDA_VISIBLE_DEVICES= \
  .venv/bin/python scripts/generator/evaluate.py chexpert cx_sd_view --out=eval_chexpert_view || echo "FAILED: evaluate view"
echo "queue_gen_view done"
