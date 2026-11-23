#!/usr/bin/env bash
# Execute one notebook in place on this machine: scripts/run_notebook.sh 06_overfitting  (NB_TIMEOUT=-1: no cell timeout)
cd "$(dirname "$0")/../notebooks"
s=$(date +%s)
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=${NB_TIMEOUT:-1500} "$1.ipynb" \
  > "../artifacts/logs/nb_$1.log" 2>&1 && echo "OK $(( $(date +%s) - s ))s" || { echo FAIL; grep -E "Error" "../artifacts/logs/nb_$1.log" | grep -v Kernel | tail -5; }
