"""Experiment catalogue: every classifier run in one table (docs/CATALOGUE.md).

Joins each run directory's run.json with its MLflow record (mlflow.db, experiment `cxr-pneumonia`, read-only) and
recomputes the headline metric from the saved predictions, since MLflow only has pooled RSNA (dev + test) AUROC.
CPU only, pandas + sklearn (no torch).

    .venv/bin/python scripts/catalogue.py
  reports/results/catalogue_runs.csv     one row per run (config, MLflow metrics, RSNA-test view-balanced / AP / PA)
  reports/results/catalogue_summary.csv  one row per run directory group (experiment + tag) x arch: mean / SD over seeds
  reports/results/catalogue_summary.md   the same, as Markdown tables for docs/CATALOGUE.md
"""
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts/runs"
OUT = ROOT / "reports/results"


def b_role(pid):  # = cxr.analysis.b_role
    return "dev" if int(hashlib.sha256(str(pid).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < 0.3 else "test"


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan


# MLflow: params, latest metrics and start time per run id
con = sqlite3.connect(f"file:{ROOT / 'mlflow.db'}?mode=ro", uri=True)
ml_runs = pd.read_sql("select run_uuid, start_time, end_time, status from runs where experiment_id = "
                      "(select experiment_id from experiments where name = 'cxr-pneumonia')", con).set_index("run_uuid")
keys = ["A_test/label/auroc", "B_external/label/auroc", "B_external/label_any/auroc", "C_test/label/auroc",
        "B_external/label/sensitivity", "B_external/label/specificity"]
met = pd.read_sql(f"select run_uuid, key, value from latest_metrics where key in ({','.join('?' * len(keys))})", con,
                  params=keys).pivot(index="run_uuid", columns="key", values="value")
commit = pd.read_sql("select run_uuid, value from tags where key = 'env.git_commit'", con).set_index("run_uuid").value

pid_role = {}
rows = []
for rj in sorted(RUNS.glob("*/*/seed*/run.json")):
    d = rj.parent
    r = json.loads(rj.read_text())
    cfg = r.get("cfg", {})
    group, arch, seed = d.parts[-3], d.parts[-2], int(d.name[4:])
    train = cfg.get("train", [])
    row = dict(group=group, experiment=cfg.get("experiment"), arch=arch, seed=seed, status=r.get("status"),
               source=cfg.get("source", "kermany"), augment=cfg.get("augment"), loss=cfg.get("loss"),
               resize_mode=cfg.get("resize_mode"), n_train=r.get("n_train"),
               train_sets=" + ".join(f"{Path(t['manifest']).stem}:{t.get('tag', '')}" for t in train),
               val_set=Path(cfg.get("val", {}).get("manifest", "")).stem, best_step=r.get("best_step"),
               steps_done=r.get("steps_done"), train_time_s=round(r.get("train_time_s") or 0, 1),
               mlflow_run_id=r.get("mlflow_run_id"))
    rid = row["mlflow_run_id"]
    if rid in ml_runs.index:
        row["started"] = pd.to_datetime(ml_runs.at[rid, "start_time"], unit="ms").strftime("%Y-%m-%d %H:%M")
        row["git_commit"] = str(commit.get(rid, ""))[:7]
    if rid in met.index:
        for k in keys:
            row["ml_" + k.replace("/label", "").replace("/", "_")] = met.at[rid, k] if k in met.columns else np.nan
    pb = d / "preds_B_external.csv"
    if pb.exists():
        b = pd.read_csv(pb, usecols=["label", "view", "prob", "patient_id"], dtype={"patient_id": str})
        b = b[b.label.notna()]
        b = b[b.patient_id.map(lambda p: pid_role.setdefault(p, b_role(p))) == "test"]
        ap, pa = (auc(b.label[b.view == v], b.prob[b.view == v]) for v in ("AP", "PA"))
        row.update(rsna_test_balanced=np.mean([ap, pa]), rsna_test_AP=ap, rsna_test_PA=pa,
                   rsna_test_pooled=auc(b.label, b.prob))
    for name, col in (("preds_C_test.csv", "chexpert_test"), ("preds_A_test.csv", "kermany_test")):
        if (d / name).exists():
            p = pd.read_csv(d / name, usecols=["label", "prob"]).dropna()
            row[col] = auc(p.label, p.prob)
    rows.append(row)

runs = pd.DataFrame(rows)
runs.round(4).to_csv(OUT / "catalogue_runs.csv", index=False)

metrics = ["rsna_test_balanced", "rsna_test_AP", "rsna_test_PA", "rsna_test_pooled", "chexpert_test", "kermany_test"]
summ = (runs.groupby(["group", "arch"], sort=False)
        .agg(source=("source", "first"), experiment=("experiment", "first"), n_train=("n_train", "first"),
             seeds=("seed", "nunique"), **{f"{m}_mean": (m, "mean") for m in metrics},
             rsna_test_balanced_sd=("rsna_test_balanced", "std"), started=("started", "min"))
        .reset_index().round(3))
summ.to_csv(OUT / "catalogue_summary.csv", index=False)

# Markdown: one table per study, configs as rows, archs as columns (view-balanced RSNA-test, mean ± SD over seeds)
ARCH_ORDER = ["efficientnet_b0", "resnet18", "resnet50", "densenet121", "convnext_tiny", "vit_small", "swin_tiny"]
short = {"efficientnet_b0": "EffNet-B0", "resnet18": "RN-18", "resnet50": "RN-50", "densenet121": "DN-121",
         "convnext_tiny": "ConvNeXt-T", "vit_small": "ViT-S", "swin_tiny": "Swin-T"}
cell = summ.assign(v=summ.apply(lambda r: f"{r.rsna_test_balanced_mean:.3f}" + (f" ±{r.rsna_test_balanced_sd:.3f}"
                                if r.seeds > 1 and pd.notna(r.rsna_test_balanced_sd) else "") + (f" ({r.seeds})" if r.seeds != 3 else ""), axis=1))
lines = ["<!-- generated by scripts/catalogue.py; do not edit by hand -->",
         f"{len(runs)} runs in {runs.group.nunique()} groups. Cell = view-balanced RSNA-test AUROC, mean ±SD over seeds "
         "(seed count in brackets when not 3).", ""]
for study, mask in (("Study 1: Kermany source", ~summ.group.str.startswith(("CX_", "CXVB_"))),
                    ("Study 2: CheXpert source", summ.group.str.startswith(("CX_", "CXVB_")))):
    t = cell[mask].pivot(index="group", columns="arch", values="v")
    t = t[[a for a in ARCH_ORDER if a in t.columns]].rename(columns=short)
    order = summ[mask].drop_duplicates("group").set_index("group").started.sort_values().index
    t = t.loc[order].fillna("")
    n = summ[mask].drop_duplicates("group").set_index("group").n_train
    lines += [f"### {study}", "", "| group | n train | " + " | ".join(t.columns) + " |",
              "|---|---|" + "---|" * len(t.columns)]
    lines += [f"| `{g}` | {int(n[g]) if pd.notna(n[g]) else ''} | " + " | ".join(t.loc[g]) + " |" for g in t.index]
    lines.append("")
(OUT / "catalogue_summary.md").write_text("\n".join(lines))
print(f"{len(runs)} runs, {runs.group.nunique()} groups -> {OUT}/catalogue_{{runs,summary}}.csv, catalogue_summary.md")
