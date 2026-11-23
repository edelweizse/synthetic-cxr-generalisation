"""Collect every finished run into reports/results/runs.csv (one row per run x eval set).

    python scripts/aggregate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cxr.analysis import RESULTS_DIR, collect_runs

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = collect_runs(include_placeholder=True)
    df.to_csv(RESULTS_DIR / "runs.csv", index=False)
    print(f"{len(df)} rows -> {RESULTS_DIR / 'runs.csv'}")
    if "eval" in df:
        print(df.groupby(["experiment", "arch", "eval"]).auroc.agg(["mean", "std", "count"]).round(4).to_string())
