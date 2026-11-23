#!/usr/bin/env bash
# Generator round 2 (docs/RESEARCH_PIVOT.md §11): the same recipe with lr 1e-4 (2e-5 before) and 7,500 steps (2,500 before,
# ~3 epochs of the 9,968 CheXpert train images), adapters saved every 2,500 steps. Then 1,000 samples per class with the
# study-1 prompts/settings and the main test SYN_ONLY (2,000 synthetic, prompt labels) vs CX_N2000. Stages skip when done.
cd "$(dirname "$0")/.."
G=data/generator; M=artifacts/manifests; ARCHS="efficientnet_b0 resnet18 densenet121"
if [ ! -f $G/lora_cx_v2/adapter_model.safetensors ]; then
  echo "=== $(date) LoRA v2 training"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 .venv-gen/bin/python scripts/generator/train_lora.py \
    --metadata_path $G/chexpert/metadata.jsonl --output_dir $G/lora_cx_v2 --variant fp16 --seed 0 \
    --lr 1e-4 --max_train_steps 7500 --save_every 2500 || { echo "FAILED: train_lora"; exit 1; }
fi
HF_HUB_OFFLINE=1 .venv-gen/bin/python scripts/generator/sample.py --adapter $G/lora_cx_v2 --name cx_sd_v2 --per-class 1000 || { echo "FAILED: sample"; exit 1; }
[ -f $M/synthetic_cx_sd_v2.csv ] || .venv/bin/python scripts/prepare_synthetic.py data/synthetic/cx_sd_v2 --name cx_sd_v2 || { echo "FAILED: register"; exit 1; }
.venv/bin/python scripts/run.py --exp SYN_ONLY --arch $ARCHS --source chexpert --synthetic $M/synthetic_cx_sd_v2.csv --tag SD2 || echo "FAILED: SYN_ONLY SD2"
.venv/bin/python scripts/aggregate.py > /dev/null
echo "queue_gen_v2 done"
