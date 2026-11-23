#!/usr/bin/env bash
# E3/E4 with the teammate's SD-1.5 LoRA images (data/synthetic/sd_lora), all architectures x 3 seeds.
# EfficientNet-B0 and ViT-S first: they have the matched real-data controls R25/R50.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
[ -f artifacts/manifests/synthetic_sd_lora.csv ] || python scripts/prepare_synthetic.py data/synthetic/sd_lora --name sd_lora
for arch in efficientnet_b0 vit_small densenet121 resnet18 resnet50 convnext_tiny swin_tiny; do
  for try in 1 2 3 4 5; do  # retry: another pipeline may share the GPU (CUDA OOM); finished runs are skipped
    python scripts/run.py --exp E3 E4 --arch "$arch" --seeds 0 1 2 \
      --synthetic artifacts/manifests/synthetic_sd_lora.csv --tag sd_lora && break
    sleep 300
  done
done
python scripts/aggregate.py > /dev/null
echo "pipeline_synth done"
