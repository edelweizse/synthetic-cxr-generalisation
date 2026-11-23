#!/usr/bin/env bash
# Study 2 baselines (docs/RESEARCH_PIVOT.md §9): the same experiments with CheXpert as the real source
# (manifests from scripts/chexpert/make_manifest.py). CX_ = natural view mix, CXVB_ = view-balanced training subsets.
# E1 first for a quick answer to "is CheXpert a better source than Kermany for RSNA?", then the strong augmentation
# (also the teacher for a CheXpert TEACH arm), then more real data. Ends by scoring every finished run on C_test.
cd "$(dirname "$0")/.."
source .venv/bin/activate
ARCHS="efficientnet_b0 resnet18 densenet121"
for exp in E1 E2_ERASE R50 N2000; do
  for src in chexpert chexpert_vb; do
    python scripts/run.py --exp $exp --arch $ARCHS --source $src || echo "FAILED: $exp $src"
  done
done
python scripts/run.py --exp E2_ERASE --arch vit_small --source chexpert || echo "FAILED: E2_ERASE vit_small"
python scripts/run.py --eval-only || echo "FAILED: eval-only"
python scripts/aggregate.py > /dev/null
echo "queue_cx done"
