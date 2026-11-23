#!/usr/bin/env bash
# Push code (not data/results) to the GPU box. Same absolute path on both machines keeps MLflow URIs valid.
set -euo pipefail
HOST=${GPU_HOST:-desktop}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
rsync -a --exclude data/ --exclude .venv/ --exclude artifacts/ --exclude notebooks/ --exclude reports/ --exclude configs/hpo/ --exclude mlruns/ --exclude mlflow.db \
  --exclude '__pycache__/' --exclude .ipynb_checkpoints/ "$ROOT/" "$HOST:$ROOT/"
