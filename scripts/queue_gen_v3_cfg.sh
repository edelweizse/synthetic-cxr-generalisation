#!/usr/bin/env bash
# v3 guidance check (docs/RESEARCH_PIVOT.md §13): with rich captions the class is a few tokens of a long caption, so CFG 2.5
# may under-draw it. 100 per class (seed 0, training captions) at CFG 4.5 / 7, then the generator evaluation (teacher signal).
cd "$(dirname "$0")/.."
G=data/generator; M=artifacts/manifests; CAP=$G/chexpert/metadata_rich.jsonl
for c in 45 70; do
  HF_HUB_OFFLINE=1 .venv-gen/bin/python scripts/generator/sample.py --adapter $G/lora_cx_v3 --name cx_sd_v3_cfg$c --per-class 100 \
    --seed 0 --cfg ${c:0:1}.${c:1:1} --captions $CAP || { echo "FAILED: sample cfg$c"; exit 1; }
  [ -f $M/synthetic_cx_sd_v3_cfg$c.csv ] || .venv/bin/python scripts/prepare_synthetic.py data/synthetic/cx_sd_v3_cfg$c --name cx_sd_v3_cfg$c || { echo "FAILED: register cfg$c"; exit 1; }
done
[ -f reports/results/generator/eval_chexpert_v3cfg.csv ] || CUDA_VISIBLE_DEVICES= systemd-run --user --scope -q -p MemoryMax=8G \
  .venv/bin/python scripts/generator/evaluate.py chexpert cx_sd_v3_cfg45 cx_sd_v3_cfg70 --out=eval_chexpert_v3cfg || echo "FAILED: evaluate"
echo "queue_gen_v3_cfg done"
