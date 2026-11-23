#!/usr/bin/env bash
# Stage 2: data experiments on the selected architectures (+ E3/E4 plumbing test on placeholder data).
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONWARNINGS="ignore::UserWarning" HF_HUB_DISABLE_PROGRESS_BARS=1
SEL=$(python scripts/select_archs.py | tail -1)
echo "=== $(date) selected architectures: $SEL"
for arch in $SEL; do
  python scripts/run.py --exp R25 R50 N250 N500 N2000 FULL FULL_AUG --arch "$arch" --seeds 0 1 2
  python scripts/aggregate.py > /dev/null
done
# E3/E4 plumbing test only (placeholder data, tagged, excluded from analysis)
PRIMARY=$(echo $SEL | cut -d' ' -f1)
if [ ! -f artifacts/manifests/synthetic_placeholder.csv ]; then
  python scripts/make_placeholder_synthetic.py && python scripts/prepare_synthetic.py data/synthetic/placeholder --name placeholder
fi
python scripts/run.py --exp E3 E4 --arch "$PRIMARY" --seeds 0 --synthetic artifacts/manifests/synthetic_placeholder.csv --tag PLACEHOLDER
python scripts/aggregate.py > /dev/null
echo "=== $(date) pipeline stage 2 done"
