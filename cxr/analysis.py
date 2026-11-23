"""Result collection and statistical comparison helpers used by the analysis notebooks."""

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import PROJECT_ROOT
from .engine import RUNS_DIR
from .metrics import _cluster_resample

RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
EVAL_LABELS = {
    "A_test:label": "Kermany test (internal)",
    "B_external:label": "RSNA: opacity vs normal",
    "B_external:label_any": "RSNA: opacity vs all",
}
# B within acquisition view: AP films carry far more opacity than PA films, so pooled B AUROC partly
# rewards recognising the view itself; within-view AUROC removes that confounder.
STRATA_LABELS = {
    "B_external:label@view=PA": "RSNA PA only: opacity vs normal",
    "B_external:label@view=AP": "RSNA AP only: opacity vs normal",
}
# B split by patient into a selection part and a reporting part. Choosing between configurations on B (e.g. picking
# E2_ERASE as the strong baseline) uses only B-dev; reported external numbers use only B-test.
B_DEV_FRACTION = 0.3
ROLE_LABELS = {
    "B_external:label@b_role=dev": "RSNA-dev: opacity vs normal (selection only)",
    "B_external:label@b_role=test": "RSNA-test: opacity vs normal (reporting)",
}
EXCLUDE_TAGS = ("PLACEHOLDER",)  # pipeline tests, never reported
# Headline external metric: RSNA-test, view-balanced AUROC = mean of the AUROC within AP films and within PA films.
# In RSNA the view is confounded with the label (AP 76% opacity vs PA 17%; view alone scores 0.796 pooled AUROC), so
# pooled AUROC partly rewards recognising the view. Pooled AUROC is kept as a secondary column.
HEADLINE_KEY = "B_external:label@b_role=test"
VIEWS = ("AP", "PA")


def balanced_auroc(y, s, view) -> float:
    """Mean of the within-view AUROCs (AP, PA); NaN if a view lacks one of the classes."""
    y, s, view = np.asarray(y), np.asarray(s), np.asarray(view)
    out = []
    for v in VIEWS:
        m = view == v
        if not m.any() or y[m].min() == y[m].max():
            return np.nan
        out.append(roc_auc_score(y[m], s[m]))
    return float(np.mean(out))


def b_role(patient_id: str) -> str:
    """Deterministic patient-level assignment to 'dev' (30%) or 'test' (70%), independent of run and seed."""
    h = int(hashlib.sha256(str(patient_id).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "dev" if h < B_DEV_FRACTION else "test"


def add_b_role(df: pd.DataFrame) -> pd.DataFrame:
    if "patient_id" in df and "b_role" not in df:
        df = df.assign(b_role=[b_role(p) for p in df["patient_id"]])
    return df


def collect_runs(include_placeholder: bool = False) -> pd.DataFrame:
    """One row per (finished run, eval set)."""
    rows = []
    for rj in sorted(RUNS_DIR.glob("*/*/seed*/run.json")):
        info = json.loads(rj.read_text())
        if info.get("status") != "complete":
            continue
        exp, arch, seed = rj.parent.parent.parent.name, rj.parent.parent.name, int(rj.parent.name[4:])
        if not include_placeholder and any(t in exp for t in EXCLUDE_TAGS):
            continue
        base = {"experiment": exp, "arch": arch, "seed": seed, "n_train": info["n_train"],
                "best_val_loss": info["best_val_loss"], "best_val_auroc": info["best_val_auroc"],
                "best_step": info["best_step"], "steps_done": info["steps_done"],
                "stopped_early": info["stopped_early"], "train_time_s": info["train_time_s"],
                "threshold": info["threshold"]}
        mp = rj.parent / "metrics.json"
        metrics = json.loads(mp.read_text()) if mp.exists() else {}
        if not metrics:
            rows.append(base)
        for key, m in metrics.items():
            row = {**base, "eval": key, **{k: v for k, v in m.items() if k != "ci"}}
            for k, (lo, hi) in m["ci"].items():
                row[f"{k}_lo"], row[f"{k}_hi"] = lo, hi
            rows.append(row)
    return pd.DataFrame(rows)


def mean_sd(df: pd.DataFrame, by, metrics, fmt="{:.3f}") -> pd.DataFrame:
    g = df.groupby(by)[metrics]
    m, s = g.mean(), g.std()
    out = m.copy().astype(object)
    for c in metrics:
        out[c] = [f"{fmt.format(a)} ± {fmt.format(b)}" if np.isfinite(b) else fmt.format(a) for a, b in zip(m[c], s[c])]
    out["seeds"] = g.size()
    return out


def seed_predictions(exp: str, arch: str, eval_set: str) -> pd.DataFrame:
    """Predictions of all seeds of one configuration, aligned by image_id, plus their mean."""
    frames = []
    for p in sorted((RUNS_DIR / exp / arch).glob(f"seed*/preds_{eval_set}.csv")):
        df = pd.read_csv(p, dtype={"patient_id": str})
        frames.append(df.set_index("image_id")["prob"].rename(p.parent.name))
        meta = df.set_index("image_id").drop(columns=["prob", "logit"], errors="ignore")
    if not frames:
        return pd.DataFrame()
    out = add_b_role(meta.reset_index()).set_index("image_id").join(pd.concat(frames, axis=1))
    out["prob_mean"] = out[[f.name for f in frames]].mean(axis=1)
    return out.reset_index()


def compare(exp_a: str, exp_b: str, arch: str, eval_key: str, n_boot: int = 2000, seed: int = 0,
            arch_b: str | None = None, metric: str = "auroc") -> dict:
    """Effect of switching from (exp_a, arch) to (exp_b, arch_b) on one eval set, optionally filtered
    ("B_external:label@b_role=test&view=AP").

    * per-seed AUROC differences (seed k vs seed k): mean, SD
    * paired, patient-clustered bootstrap on the seed-averaged scores: delta AUROC, 95% CI, two-sided p
    * metric="balanced": view-balanced AUROC (mean of AP and PA AUROC) instead of AUROC; B_external only
    """
    eval_part, _, flt = eval_key.partition("@")
    eval_set, col = eval_part.split(":")
    pa, pb = seed_predictions(exp_a, arch, eval_set), seed_predictions(exp_b, arch_b or arch, eval_set)
    if pa.empty or pb.empty:
        return {}
    df = pa.merge(pb[["image_id"] + [c for c in pb if c.startswith(("seed", "prob_mean"))]],
                  on="image_id", suffixes=("_a", "_b"))
    df = df[df[col].notna()]
    for f in filter(None, flt.split("&")):  # e.g. "@b_role=test&view=AP"
        fc, fv = f.split("=")
        df = df[df[fc].astype(str) == fv]
    y, g = df[col].to_numpy().astype(int), df["patient_id"].to_numpy()
    if metric == "balanced":
        view = df["view"].to_numpy()
        score = lambda yy, ss, idx=slice(None): balanced_auroc(yy, ss, view[idx])
    else:
        score = lambda yy, ss, idx=slice(None): roc_auc_score(yy, ss)
    seeds = sorted({c for c in pa if c.startswith("seed")} & {c for c in pb if c.startswith("seed")})
    per_seed = [score(y, df[f"{s}_b"].to_numpy()) - score(y, df[f"{s}_a"].to_numpy()) for s in seeds]
    a, b = df["prob_mean_a"].to_numpy(), df["prob_mean_b"].to_numpy()
    gen = _cluster_resample(g, np.random.default_rng(seed))
    deltas = []
    while len(deltas) < n_boot:
        idx = next(gen)
        if y[idx].min() == y[idx].max():
            continue
        d = score(y[idx], b[idx], idx) - score(y[idx], a[idx], idx)
        if np.isfinite(d):
            deltas.append(d)
    deltas = np.asarray(deltas)
    return {"from": exp_a, "to": exp_b, "arch": arch if arch_b is None else f"{arch}->{arch_b}", "eval": eval_key,
            "metric": metric, "auroc_from": score(y, a), "auroc_to": score(y, b),
            "delta_ensemble": score(y, b) - score(y, a),
            "ci_lo": float(np.percentile(deltas, 2.5)), "ci_hi": float(np.percentile(deltas, 97.5)),
            "p_boot": float(min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))),
            "delta_per_seed_mean": float(np.mean(per_seed)) if per_seed else np.nan,
            "delta_per_seed_sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else np.nan,
            "n_seeds": len(per_seed), "n": len(y)}


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    y, p = np.asarray(y), np.asarray(p)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return float(sum(np.abs(y[bins == k].mean() - p[bins == k].mean()) * (bins == k).mean()
                     for k in range(n_bins) if (bins == k).any()))


def select_archs(runs: pd.DataFrame, k: int = 2) -> list[str]:
    """Pre-registered rule: rank architectures by mean E1 A-validation loss (never by external results).

    Returns the best CNN and the best transformer when k == 2, otherwise the top-k overall.
    """
    e1 = runs[runs.experiment == "E1"].drop_duplicates(["arch", "seed"])
    order = e1.groupby("arch").best_val_loss.mean().sort_values()
    if k == 2:
        transformers = [a for a in order.index if a.startswith(("vit", "swin", "deit"))]
        cnns = [a for a in order.index if a not in transformers]
        return [a for a in (cnns[:1] + transformers[:1])] or list(order.index[:k])
    return list(order.index[:k])


def headline_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Per run (experiment, arch, seed) on RSNA-test: view-balanced (headline), AP, PA and pooled AUROC."""
    rows = []
    for (exp, arch, seed), _ in runs.groupby(["experiment", "arch", "seed"]):
        p = RUNS_DIR / exp / arch / f"seed{seed}" / "preds_B_external.csv"
        if not p.exists():
            continue
        df = add_b_role(pd.read_csv(p, usecols=["label", "view", "prob", "patient_id"], dtype={"patient_id": str}))
        df = df[df.label.notna() & (df.b_role == "test")]
        r = {"experiment": exp, "arch": arch, "seed": seed, "pooled": roc_auc_score(df.label, df.prob)}
        for v in VIEWS:
            g = df[df.view == v]
            r[v] = roc_auc_score(g.label, g.prob)
        r["balanced"] = (r["AP"] + r["PA"]) / 2
        rows.append(r)
    return pd.DataFrame(rows)


def strata_runs(runs: pd.DataFrame, keys=None) -> pd.DataFrame:
    """Per-run AUROC within B strata (acquisition view, dev/test role), computed from cached predictions."""
    keys = keys or STRATA_LABELS  # pass {**STRATA_LABELS, **ROLE_LABELS} for the dev/test split
    rows = []
    for (exp, arch, seed), _ in runs.groupby(["experiment", "arch", "seed"]):
        p = RUNS_DIR / exp / arch / f"seed{seed}" / "preds_B_external.csv"
        if not p.exists():
            continue
        df = add_b_role(pd.read_csv(p, usecols=["label", "view", "prob", "patient_id"], dtype={"patient_id": str}))
        df = df[df.label.notna()]
        for key in keys:
            col, v = key.split("@")[1].split("=")
            g = df[df[col].astype(str) == v]
            rows.append({"experiment": exp, "arch": arch, "seed": seed, "eval": key,
                         "auroc": roc_auc_score(g.label, g.prob), "n": len(g)})
    return pd.DataFrame(rows)
