#!/usr/bin/env bash
# Serial GPU queue for the remaining work, highest priority first (one job at a time: no CUDA OOM from sharing).
# Every pipeline is idempotent, so finished runs are skipped and an interrupted run resumes.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
scripts/retry_until_done.sh scripts/pipeline_harm.sh artifacts/logs/pipeline_harm.log 3    # harmonised E1 seed 2, E2, E2_ERASE
scripts/retry_until_done.sh scripts/pipeline_erase.sh artifacts/logs/pipeline_erase.log 3  # ERASE series
scripts/retry_until_done.sh scripts/pipeline_harmt.sh artifacts/logs/pipeline_harmt.log 3  # HPO re-tuned for harmonize224
ARCHS="resnet18 resnet50 swin_tiny convnext_tiny" scripts/pipeline_stage3.sh 2>&1 | tee -a artifacts/logs/pipeline_stage3b.log
scripts/pipeline_synth.sh 2>&1 | tee -a artifacts/logs/pipeline_synth.log
echo "queue done"
