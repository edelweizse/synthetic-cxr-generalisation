#!/usr/bin/env bash
# Obvious cases + more data (docs/RESEARCH_PIVOT.md §13): the fetch.py select-gen obvious set (14,524 No Finding + 14,524
# Consolidation / Pneumonia images) as data/generator/chexpert_obv, the v2 recipe with 15,000 steps (~2 epochs) at rank 16,
# CFG 2.5, then the fully synthetic pipeline (tag SDOBVV vs SD2C25V) and the generator evaluation. Starts after the tmux
# download session (scripts/fetch_gen.sh) ends.
cd "$(dirname "$0")/.."
G=data/generator; M=artifacts/manifests; ARCHS="efficientnet_b0 resnet18 densenet121"; R=16
while tmux has-session -t =fetch 2>/dev/null; do sleep 300; done
.venv/bin/python scripts/chexpert/fetch.py images selection_obv.csv || { echo "FAILED: fetch"; exit 1; }
[ -f $G/chexpert_obv/metadata.jsonl ] || .venv/bin/python scripts/generator/prepare_chexpert.py big selection_obv.csv chexpert_obv || { echo "FAILED: prepare"; exit 1; }
while tmux has-session -t =r64 2>/dev/null || tmux has-session -t =view 2>/dev/null; do sleep 300; done  # one GPU job at a time
if [ ! -f $G/lora_cx_obv/adapter_model.safetensors ]; then
  echo "=== $(date) LoRA obv training (rank $R)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 .venv-gen/bin/python scripts/generator/train_lora.py \
    --metadata_path $G/chexpert_obv/metadata.jsonl --output_dir $G/lora_cx_obv --variant fp16 --seed 0 \
    --lr 1e-4 --max_train_steps 15000 --save_every 7500 --lora_r $R --lora_alpha $R || { echo "FAILED: train_lora"; exit 1; }
fi
for spec in "cx_sd_obv 1000 0" "cx_sd_obv_val 250 1"; do
  set -- $spec
  HF_HUB_OFFLINE=1 .venv-gen/bin/python scripts/generator/sample.py --adapter $G/lora_cx_obv --name $1 --per-class $2 --seed $3 \
    --cfg 2.5 || { echo "FAILED: sample $1"; exit 1; }
  [ -f $M/synthetic_$1.csv ] || .venv/bin/python scripts/prepare_synthetic.py data/synthetic/$1 --name $1 || { echo "FAILED: register $1"; exit 1; }
done
.venv/bin/python scripts/run.py --exp SYN_ONLY --arch $ARCHS --source chexpert --synthetic $M/synthetic_cx_sd_obv.csv \
  --val $M/synthetic_cx_sd_obv_val.csv --tag SDOBVV || echo "FAILED: SYN_ONLY SDOBVV"
[ -f reports/results/generator/eval_chexpert_obv.csv ] || CUDA_VISIBLE_DEVICES= \
  .venv/bin/python scripts/generator/evaluate.py chexpert cx_sd_obv --out=eval_chexpert_obv || echo "FAILED: evaluate obv"
echo "queue_gen_obv done"
