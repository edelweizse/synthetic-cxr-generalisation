"""Google's claim on our runs (Ktena et al. 2024; docs/RESEARCH_PIVOT.md §13): does adding synthetic images to real ones help outside
the training hospital, and does it shrink the sex gap? Per run: view-balanced RSNA-test AUROC, CheXpert-test AUROC, |AUROC male -
AUROC female| on both; mean and SD over arch x seed. pandas only.

    .venv/bin/python scripts/generator/mixed_fairness.py   -> reports/results/generator/mixed_fairness.csv
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "artifacts/runs"
ARCHS = ["efficientnet_b0", "resnet18", "densenet121"]
EXPS = ["CX_E1", "CX_R25", "CX_R50", "CX_N2000", "CX_E3_SD", "CX_E4_SD", "CX_E3_TEACH", "CX_E4_TEACH", "CX_E3_RTEACH", "CX_E4_RTEACH",
        "CX_SYN2000_SD2C25", "CX_N250", "CX_N250_SYN2000_SD2C25", "CX_FULL"]
M = ROOT / "artifacts/manifests"


def b_role(pid):  # = cxr.analysis.b_role
    return "dev" if int(hashlib.sha256(str(pid).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < 0.3 else "test"


def balanced_auroc(y, s, v):  # = cxr.analysis.balanced_auroc
    return float(np.mean([roc_auc_score(y[v == k], s[v == k]) for k in ("AP", "PA")]))


sex = {k: pd.read_csv(M / f, usecols=["patient_id", "sex"], dtype={"patient_id": str}).drop_duplicates("patient_id")
       .assign(sex=lambda d: d.sex.astype(str).str[0].str.upper()).set_index("patient_id").sex for k, f in (("b", "rsna.csv"), ("c", "chexpert.csv"))}


def gap(d):
    a = [roc_auc_score(d.label[d.sex == s], d.prob[d.sex == s]) for s in "MF"]
    return abs(a[0] - a[1])


rows = []
for e in EXPS:
    for a in ARCHS:
        for r in sorted((RUNS / e / a).glob("seed*")):
            if not (r / "preds_C_test.csv").exists():
                continue
            b = pd.read_csv(r / "preds_B_external.csv", usecols=["label", "view", "prob", "patient_id"], dtype={"patient_id": str})
            b = b[b.label.notna() & (b.patient_id.map(b_role) == "test")].assign(sex=lambda d: d.patient_id.map(sex["b"]))
            c = pd.read_csv(r / "preds_C_test.csv", usecols=["label", "prob", "patient_id"], dtype={"patient_id": str})
            c = c.assign(sex=c.patient_id.map(sex["c"]))
            rows.append(dict(exp=e, arch=a, seed=r.name, rsna=balanced_auroc(b.label.values, b.prob.values, b.view.values),
                             chexpert=roc_auc_score(c.label, c.prob), rsna_sex_gap=gap(b), chexpert_sex_gap=gap(c)))
df = pd.DataFrame(rows)
g = df.groupby("exp", sort=False)[["rsna", "chexpert", "rsna_sex_gap", "chexpert_sex_gap"]].agg(["mean", "std"]).round(3)
g.insert(0, "runs", df.groupby("exp", sort=False).size())
g.to_csv(ROOT / "reports/results/generator/mixed_fairness.csv")
pd.set_option("display.width", 200)
print(g.to_string())
