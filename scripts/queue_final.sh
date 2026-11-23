#!/usr/bin/env bash
# Last controls (2026-11-12, docs/RESEARCH_PIVOT.md §13.x): is synthetic-only + teacher labels (RSNA 0.92) the images or the
# teacher? All arms: 2,000 training images, SYN_ONLY, real CheXpert validation (as CX_SYN_ONLY_TEACH), 3 archs x 3 seeds.
#   RTEACH  real unseen images + teacher labels (distillation control)
#   TEACHV2 final generator images (v2, CFG 2.5) + teacher labels
#   SD2C25  the same final images + prompt labels (pair for TEACHV2; SD2C25V used synthetic validation)
# Finished runs are skipped. Log: logs_final.log
cd "$(dirname "$0")/.."
M=artifacts/manifests; ARCHS="efficientnet_b0 resnet18 densenet121"
[ -f $M/real_teach_cx_2k.csv ] || .venv/bin/python scripts/generator/teach_final.py || { echo "FAILED: teacher"; exit 1; }
for spec in "RTEACH real_teach_cx_2k" "TEACHV2 synthetic_cx_sd_v2c25_teach" "SD2C25 synthetic_cx_sd_v2c25"; do
  set -- $spec
  .venv/bin/python scripts/run.py --exp SYN_ONLY --arch $ARCHS --source chexpert --synthetic $M/$2.csv --tag $1 || echo "FAILED: SYN_ONLY $1"
done
echo "queue_final done"
