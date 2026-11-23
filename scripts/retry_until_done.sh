#!/usr/bin/env bash
# Re-run an idempotent pipeline until it reports no failures (finished runs are skipped, partial ones resume).
# Used when two pipelines share the GPU and one of them hits CUDA OOM.  usage: scripts/retry_until_done.sh <script> <log> [max_tries]
set -u
cd "$(dirname "$0")/.."
for i in $(seq 1 "${3:-20}"); do
  out=$("$1" 2>&1 | tee -a "$2")
  echo "$out" | grep -q "^FAILED" || { echo "=== $(date) $1 complete after $i tries" | tee -a "$2"; exit 0; }
  sleep 300
done
echo "=== $(date) $1 still failing" | tee -a "$2"; exit 1
