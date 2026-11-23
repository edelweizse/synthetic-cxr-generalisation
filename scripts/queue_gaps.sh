#!/usr/bin/env bash
# Fill the architecture gaps of the design (state 2026-10-13), most important first:
#   1. R25 / R50 (matched real controls for E3 / E4) on the 5 archs that only had synthetic runs
#   2. qcons on E1 for the 3 archs without it, and E3 + qcons on the 5 archs without it
#   3. E2_ERASE + 250 synthetic vs + 250 real on the 4 archs without them
# Idempotent: finished runs are skipped, so it can be re-run with scripts/retry_until_done.sh.
cd "$(dirname "$0")/.."
source .venv/bin/activate
S=artifacts/manifests/synthetic_sd_lora.csv
OTHER5="resnet18 resnet50 densenet121 convnext_tiny swin_tiny"
python scripts/run.py --exp R25 R50 --arch $OTHER5 || echo "FAILED: R25/R50"
python scripts/run.py --exp E1 --arch resnet18 resnet50 convnext_tiny --loss qcons --tag LQCONS --grad-ckpt || echo "FAILED: E1 qcons"
python scripts/run.py --exp E3 --arch densenet121 vit_small resnet18 resnet50 convnext_tiny --loss qcons --synthetic $S \
  --tag sd_lora_LQCONS --grad-ckpt || echo "FAILED: E3 qcons"
python scripts/run.py --exp ERASE_REAL250 --arch resnet18 resnet50 convnext_tiny swin_tiny || echo "FAILED: ERASE_REAL250"
python scripts/run.py --exp ERASE_SYN250 --arch resnet18 resnet50 convnext_tiny swin_tiny --synthetic $S --tag sd_lora \
  || echo "FAILED: ERASE_SYN250"
python scripts/aggregate.py > /dev/null
echo "queue_gaps done"
