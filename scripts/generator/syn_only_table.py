"""Main test of the project goal (docs/RESEARCH_PIVOT.md §10-12): synthetic vs real CheXpert images (2,000 each, scaling, mixed).
Per-seed view-balanced RSNA-test AUROC and CheXpert-test AUROC, mean and SD over seeds. pandas only (no torch import).

    .venv/bin/python scripts/generator/syn_only_table.py   -> reports/results/generator/syn_only.csv
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "artifacts/runs"
EXPS = ["CX_N250", "CX_N500", "CX_E1", "CX_N2000", "CX_FULL", "CX_SYN_ONLY_SD", "CX_SYN_ONLY_SD2", "CX_SYN_ONLY_SD2V",
        "CX_SYN_ONLY_SD2C25V", "CX_SYN_ONLY_SD3V", "CX_SYN_ONLY_SDR64V", "CX_SYN_ONLY_SDVIEWV", "CX_SYN_ONLY_SDBIGV", "CX_SYN_ONLY_SDOBVV", "CX_SYN_ONLY5K_SD2C25V", "CX_SYN_ONLY10K_SD2C25V", "CX_N250_SYN2000_SD2C25", "CX_SYN2000_SD2C25",
        "CX_SYN_ONLY_TEACH", "CX_SYN_ONLY_RTEACH", "CX_SYN_ONLY_TEACHV2", "CX_SYN_ONLY_SD2C25"]
ARCHS = ["efficientnet_b0", "resnet18", "densenet121"]


def b_role(pid):  # = cxr.analysis.b_role
    return "dev" if int(hashlib.sha256(str(pid).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < 0.3 else "test"


def balanced_auroc(y, s, v):  # = cxr.analysis.balanced_auroc
    return float(np.mean([roc_auc_score(y[v == k], s[v == k]) for k in ("AP", "PA")]))


rows = []
for e in EXPS:
    for a in ARCHS:
        for d in sorted((RUNS / e / a).glob("seed*")):
            if not (d / "preds_C_test.csv").exists():
                continue
            b = pd.read_csv(d / "preds_B_external.csv", usecols=["label", "view", "prob", "patient_id"], dtype={"patient_id": str})
            b = b[b.label.notna() & (b.patient_id.map(b_role) == "test")]
            c = pd.read_csv(d / "preds_C_test.csv", usecols=["label", "prob"])
            rows.append(dict(exp=e, arch=a, seed=d.name, rsna_balanced=balanced_auroc(b.label.values, b.prob.values, b.view.values),
                             chexpert_test=roc_auc_score(c.label, c.prob)))
df = pd.DataFrame(rows)
g = df.groupby(["exp", "arch"], sort=False)[["rsna_balanced", "chexpert_test"]].agg(["mean", "std"]).round(3)
g.to_csv(ROOT / "reports/results/generator/syn_only.csv")
pd.set_option("display.width", 200)
print(g.to_string())
