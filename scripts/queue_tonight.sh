#!/usr/bin/env bash
# Short queue, highest value first; the GPU must be free by 18:35 (watchdog in tmux `deadline` kills this at 18:35).
# Everything is resumable: re-run this script or scripts/queue_remaining.sh later to finish.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=artifacts/manifests/synthetic_sd_lora.csv
python scripts/run.py --exp SYN1000 SYN2000 SYN_ONLY --arch efficientnet_b0 --synthetic $S --tag sd_lora
python scripts/run.py --exp ERASE_SYN250 ERASE_SYN500 --arch efficientnet_b0 --resize-mode harmonize224 --synthetic $S --tag sd_lora_HARM
ARCHS="convnext_tiny" scripts/pipeline_stage3.sh 2>&1 | tee -a artifacts/logs/pipeline_stage3b.log
python scripts/aggregate.py > /dev/null
CUDA_VISIBLE_DEVICES="" scripts/run_notebook.sh 07_synthetic
CUDA_VISIBLE_DEVICES="" scripts/run_notebook.sh 06_overfitting
echo "queue_tonight done"
