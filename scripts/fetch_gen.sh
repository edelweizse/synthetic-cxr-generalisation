#!/usr/bin/env bash
# Download the generator set images of data/chexpert/${SEL:-selection_obv.csv} (resumable; docs/RESEARCH_PIVOT.md §13).
cd "$(dirname "$0")/.."
.venv/bin/python scripts/chexpert/fetch.py images ${SEL:-selection_obv.csv} || echo "FAILED: fetch"
