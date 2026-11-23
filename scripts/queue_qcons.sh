#!/usr/bin/env bash
# Follow-ups for the quality-consistency loss (cxr/engine.LOSSES["qcons"]): on top of augmentation, with synthetic data
# + augmentation, lambda sweep, other architectures. Waits for queue_losses (tmux "loss").
cd "$(dirname "$0")/.."
source .venv/bin/activate
while tmux has -t =loss 2>/dev/null; do sleep 60; done
S=artifacts/manifests/synthetic_sd_lora.csv
python scripts/run.py --exp E2_ERASE --arch efficientnet_b0 --loss qcons --tag LQCONS || echo "FAILED: E2_ERASE qcons"
python scripts/run.py --exp ERASE_SYN250 --arch efficientnet_b0 --loss qcons --synthetic $S --tag sd_lora_LQCONS || echo "FAILED: ERASE_SYN250 qcons"
python scripts/run.py --exp E1 --arch densenet121 vit_small --loss qcons --tag LQCONS || echo "FAILED: E1 qcons archs"
python scripts/run.py --exp E1 --arch efficientnet_b0 --loss qcons --loss-params '{"lam": 0.3}' --tag LQCONS03 || echo "FAILED: lam 0.3"
python scripts/run.py --exp E1 --arch efficientnet_b0 --loss qcons --loss-params '{"lam": 3.0}' --tag LQCONS3 || echo "FAILED: lam 3"
python scripts/aggregate.py > /dev/null
echo "queue_qcons done"
