"""Headline tables on RSNA-test with the view-balanced AUROC (mean of the AP and PA AUROC; cxr/analysis.py).

  CUDA_VISIBLE_DEVICES= python scripts/headline_tables.py [n_boot]
Writes reports/results/
  headline_runs.csv     per run: balanced (headline), AP, PA, pooled
  headline_summary.csv  per (experiment, arch): mean and SD over seeds
  headline_vs_ref.csv   Δ vs the reference recipe (E1 for everything; also E2 for E2_*, E2_ERASE for ERASE_*),
                        balanced and pooled: per-seed mean / SD, 3-seed ensemble Δ with a patient-clustered bootstrap CI
"""
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from cxr.analysis import HEADLINE_KEY, collect_runs, compare, headline_runs

OUT = ROOT / "reports/results"
N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 500


def refs(exp):
    out = ["E1"]
    if exp.startswith("E2_"):
        out.append("E2")
    if exp.startswith("ERASE_"):
        out.append("E2_ERASE")
    return [r for r in out if r != exp]


def job(args):
    a, b, arch, metric = args
    return compare(a, b, arch, HEADLINE_KEY, n_boot=N_BOOT, metric=metric)


if __name__ == "__main__":
    runs = collect_runs()
    per_run = headline_runs(runs).sort_values(["experiment", "arch", "seed"])
    per_run.round(4).to_csv(OUT / "headline_runs.csv", index=False)
    g = per_run.groupby(["experiment", "arch"])[["balanced", "AP", "PA", "pooled"]]
    summ = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).join(g.size().rename("seeds"))
    summ.round(4).to_csv(OUT / "headline_summary.csv")
    print(f"{len(per_run)} runs, {len(summ)} configurations", flush=True)

    have = set(zip(per_run.experiment, per_run.arch))
    jobs = [(r, e, a, m) for e, a in sorted(have) for r in refs(e) if (r, a) in have for m in ("balanced", "auroc")]
    print(f"{len(jobs)} comparisons, n_boot {N_BOOT}", flush=True)
    with Pool(4) as pool:
        res = [r for r in pool.imap(job, jobs, chunksize=4) if r]
    pd.DataFrame(res).round(4).to_csv(OUT / "headline_vs_ref.csv", index=False)
    print("headline tables done", flush=True)
