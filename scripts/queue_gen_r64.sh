#!/usr/bin/env bash
# Capacity check (docs/RESEARCH_PIVOT.md §13): the v2 recipe (two class prompts, lr 1e-4, 7,500 steps, CFG 2.5) with LoRA rank 64
# instead of 16 (alpha 64); 1,000 per class (seed 0) + 250 per class validation (seed 1); SYN_ONLY tag SDR64V vs SD2C25V.
cd "$(dirname "$0")/.."
G=data/generator; M=artifacts/manifests; ARCHS="efficientnet_b0 resnet18 densenet121"; CAP=$G/chexpert/metadata.jsonl
if [ ! -f $G/lora_cx_r64/adapter_model.safetensors ]; then
  echo "=== $(date) LoRA r64 training"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 .venv-gen/bin/python scripts/generator/train_lora.py \
    --metadata_path $CAP --output_dir $G/lora_cx_r64 --variant fp16 --seed 0 \
    --lr 1e-4 --max_train_steps 7500 --lora_r 64 --lora_alpha 64 || { echo "FAILED: train_lora"; exit 1; }
fi
for spec in "cx_sd_r64 1000 0" "cx_sd_r64_val 250 1"; do
  set -- $spec
  HF_HUB_OFFLINE=1 .venv-gen/bin/python scripts/generator/sample.py --adapter $G/lora_cx_r64 --name $1 --per-class $2 --seed $3 \
    --cfg 2.5 || { echo "FAILED: sample $1"; exit 1; }
  [ -f $M/synthetic_$1.csv ] || .venv/bin/python scripts/prepare_synthetic.py data/synthetic/$1 --name $1 || { echo "FAILED: register $1"; exit 1; }
done
.venv/bin/python scripts/run.py --exp SYN_ONLY --arch $ARCHS --source chexpert --synthetic $M/synthetic_cx_sd_r64.csv \
  --val $M/synthetic_cx_sd_r64_val.csv --tag SDR64V || echo "FAILED: SYN_ONLY SDR64V"
[ -f reports/results/generator/eval_chexpert_r64.csv ] || CUDA_VISIBLE_DEVICES= \
  .venv/bin/python scripts/generator/evaluate.py chexpert cx_sd_r64 --out=eval_chexpert_r64 || echo "FAILED: evaluate r64"
echo "queue_gen_r64 done"
