#!/usr/bin/env bash
# "Can synthetic data be made to help?" (docs/RESEARCH_PIVOT.md §8; manifests from scripts/synth_fix/make_manifests.py):
# SRC_S10/S08 counterfactual pairs with the source label (no-op-edit control), TEACH teacher soft labels on the sampled
# synthetic images, CTX real lungs with generator-redrawn context (scripts/synth_fix/ctx_generate.py, waits for it),
# CONF teacher-confident synthetic images, RTEACH teacher labels on the extra real images (distillation control).
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=artifacts/manifests
for spec in "SRC_S10 synthetic_cf_lung_s10_src" "TEACH synthetic_sd_lora_teach" "CTX synthetic_ctx_e1" \
            "RTEACH real_teach" "CONF synthetic_sd_lora_conf" "SRC_S08 synthetic_cf_lung_s08_src"; do
  set -- $spec
  while [ "$1" = CTX ] && pgrep -f "^\.venv-gen/bin/python scripts/synth_fix/ctx_generate.py" > /dev/null; do sleep 60; done
  [ -f $M/$2.csv ] || { echo "FAILED: missing $M/$2.csv"; continue; }
  for arch in efficientnet_b0 resnet18 densenet121; do
    python scripts/run.py --exp E3 E4 --arch $arch --synthetic $M/$2.csv --tag $1 || echo "FAILED: $arch $1"
  done
done
python scripts/aggregate.py > /dev/null
echo "queue_fix done"
