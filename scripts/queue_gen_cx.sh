#!/usr/bin/env bash
# Study 2 synthetic arms (docs/RESEARCH_PIVOT.md §9): our generator recipe (teammate's train_lora.py, unchanged
# hyper-parameters) retrained on CheXpert train, sampled with the study-1 prompts and settings, then the study-1 arms.
# Every stage is skipped when its output exists, so the queue can be re-run after a failure.
cd "$(dirname "$0")/.."
G=data/generator; M=artifacts/manifests; ARCHS="efficientnet_b0 resnet18 densenet121"
[ -f $G/chexpert/metadata.jsonl ] || .venv/bin/python scripts/generator/prepare_chexpert.py || { echo "FAILED: prepare"; exit 1; }
if [ ! -f $G/lora_cx/adapter_model.safetensors ]; then
  echo "=== $(date) LoRA training"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 .venv-gen/bin/python scripts/generator/train_lora.py --metadata_path $G/chexpert/metadata.jsonl \
    --output_dir $G/lora_cx --variant fp16 --seed 0 || { echo "FAILED: train_lora"; exit 1; }
fi
HF_HUB_OFFLINE=1 .venv-gen/bin/python scripts/generator/sample.py --adapter $G/lora_cx --name cx_sd --per-class 1000 || { echo "FAILED: sample"; exit 1; }
[ -f $M/synthetic_cx_sd.csv ] || .venv/bin/python scripts/prepare_synthetic.py data/synthetic/cx_sd --name cx_sd || { echo "FAILED: register"; exit 1; }
[ -f $M/real_teach_cx.csv ] || .venv/bin/python scripts/generator/cx_manifests.py || { echo "FAILED: teacher"; exit 1; }
source .venv/bin/activate
# main question: can synthetic data REPLACE the real dataset? 2,000 synthetic only vs CX_N2000 (2,000 real)
for spec in "SD synthetic_cx_sd" "TEACH synthetic_cx_sd_teach"; do
  set -- $spec
  python scripts/run.py --exp SYN_ONLY --arch $ARCHS --source chexpert --synthetic $M/$2.csv --tag $1 || echo "FAILED: SYN_ONLY $1"
done
python scripts/run.py --exp R25 --arch $ARCHS --source chexpert || echo "FAILED: CX_R25"
for spec in "SD synthetic_cx_sd" "TEACH synthetic_cx_sd_teach" "RTEACH real_teach_cx"; do
  set -- $spec
  python scripts/run.py --exp E3 E4 --arch $ARCHS --source chexpert --synthetic $M/$2.csv --tag $1 || echo "FAILED: $1"
done
python scripts/aggregate.py > /dev/null
echo "queue_gen_cx done"
