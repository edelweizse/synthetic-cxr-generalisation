#!/usr/bin/env bash
# Pull results back from the GPU box: MLflow store, run summaries/predictions (no checkpoints),
# HPO results, reports (notebooks are generated locally and pulled explicitly). Checkpoints stay on the GPU box except exported demo models.
set -euo pipefail
HOST=${GPU_HOST:-desktop}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
rsync -a "$HOST:$ROOT/mlflow.db" "$ROOT/" 2>/dev/null || true
rsync -a "$HOST:$ROOT/mlruns" "$ROOT/" 2>/dev/null || true
rsync -a --exclude '*.pt' --exclude '*.pt.tmp' "$HOST:$ROOT/artifacts/runs" "$ROOT/artifacts/" 2>/dev/null || true
rsync -a --exclude 'trials/' "$HOST:$ROOT/artifacts/optuna" "$ROOT/artifacts/" 2>/dev/null || true
rsync -a "$HOST:$ROOT/artifacts/manifests" "$HOST:$ROOT/artifacts/logs" "$ROOT/artifacts/" 2>/dev/null || true
for d in configs/hpo reports models; do
  rsync -a "$HOST:$ROOT/$d/" "$ROOT/$d/" 2>/dev/null || true
done
