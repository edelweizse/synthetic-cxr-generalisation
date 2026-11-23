#!/usr/bin/env bash
# Detached finisher (survives laptop shutdown): notebook 06 now, then remaining stage-3 runs, then notebook 06 again.
cd "$(dirname "$0")/.."
LOG=artifacts/logs/finish_stage3.log
echo "=== $(date) notebook 06 (seeds 0-1)" >> $LOG
scripts/run_notebook.sh 06_overfitting >> $LOG 2>&1
cp notebooks/06_overfitting.ipynb notebooks/06_overfitting_seeds01.ipynb
echo "=== $(date) remaining stage-3 runs (seed 2, densenet121)" >> $LOG
scripts/pipeline_stage3.sh >> $LOG 2>&1
echo "=== $(date) notebook 06 (all seeds)" >> $LOG
scripts/run_notebook.sh 06_overfitting >> $LOG 2>&1
.venv/bin/python scripts/aggregate.py > /dev/null 2>&1
echo "=== $(date) finish_stage3 done" >> $LOG
