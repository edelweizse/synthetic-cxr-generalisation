#!/usr/bin/env bash
# More data (docs/RESEARCH_PIVOT.md §13): download the fetch.py select-gen set ON THE BOX (Redivis, the box's network), build
# data/generator/chexpert_big (29,538 images, two class prompts), train the v2 recipe on it with 15,000 steps (~2 epochs,
# v2: 7,500 steps on 9,968 = 3 epochs) at rank R (env, default 16; set from the r64 result), CFG 2.5, then the fully
# synthetic pipeline (tag SDBIGV vs SD2C25V) and the generator evaluation. Needs the Redivis sign-in on the box first.
cd "$(dirname "$0")/.."
G=data/generator; M=artifacts/manifests; ARCHS="efficientnet_b0 resnet18 densenet121"; R=${R:-16}
.venv/bin/python scripts/chexpert/fetch.py images selection_gen.csv || { echo "FAILED: fetch"; exit 1; }
[ -f $G/chexpert_big/metadata.jsonl ] || .venv/bin/python scripts/generator/prepare_chexpert.py big || { echo "FAILED: prepare"; exit 1; }
while tmux has-session -t =r64 2>/dev/null || tmux has-session -t =view 2>/dev/null; do sleep 300; done  # one GPU job at a time
if [ ! -f $G/lora_cx_big/adapter_model.safetensors ]; then
  echo "=== $(date) LoRA big training (rank $R)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 .venv-gen/bin/python scripts/generator/train_lora.py \
    --metadata_path $G/chexpert_big/metadata.jsonl --output_dir $G/lora_cx_big --variant fp16 --seed 0 \
    --lr 1e-4 --max_train_steps 15000 --save_every 7500 --lora_r $R --lora_alpha $R || { echo "FAILED: train_lora"; exit 1; }
fi
for spec in "cx_sd_big 1000 0" "cx_sd_big_val 250 1"; do
  set -- $spec
  HF_HUB_OFFLINE=1 .venv-gen/bin/python scripts/generator/sample.py --adapter $G/lora_cx_big --name $1 --per-class $2 --seed $3 \
    --cfg 2.5 || { echo "FAILED: sample $1"; exit 1; }
  [ -f $M/synthetic_$1.csv ] || .venv/bin/python scripts/prepare_synthetic.py data/synthetic/$1 --name $1 || { echo "FAILED: register $1"; exit 1; }
done
.venv/bin/python scripts/run.py --exp SYN_ONLY --arch $ARCHS --source chexpert --synthetic $M/synthetic_cx_sd_big.csv \
  --val $M/synthetic_cx_sd_big_val.csv --tag SDBIGV || echo "FAILED: SYN_ONLY SDBIGV"
[ -f reports/results/generator/eval_chexpert_big.csv ] || CUDA_VISIBLE_DEVICES= \
  .venv/bin/python scripts/generator/evaluate.py chexpert cx_sd_big --out=eval_chexpert_big || echo "FAILED: evaluate big"
echo "queue_gen_big done"
