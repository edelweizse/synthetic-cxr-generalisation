#!/usr/bin/env bash
# View confound check (AP vs PA in RSNA): view-balanced metrics for every run and the key comparisons, on CPU.
# Waits for any running analysis script so the box is not overloaded.
cd "$(dirname "$0")/.."
source .venv/bin/activate
while pgrep -f "[s]ynth_probe/(natural_quality|loss_compare)" > /dev/null; do sleep 30; done
CUDA_VISIBLE_DEVICES= python scripts/synth_probe/view_confound.py || echo "FAILED: view_confound"
echo "queue_view done"
