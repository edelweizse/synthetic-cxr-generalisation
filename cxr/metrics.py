"""Discrimination/threshold metrics and patient-clustered bootstrap."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def choose_threshold(y, p, method: str = "youden") -> float:
    """Pick a decision threshold on source validation data only."""
    y, p = np.asarray(y), np.asarray(p)
    if method == "youden":
        fpr, tpr, thr = roc_curve(y, p)
        return float(np.clip(thr[np.argmax(tpr - fpr)], 0.0, 1.0))
    if method == "fixed":
        return 0.5
    raise ValueError(method)


def threshold_metrics(y, p, thr: float) -> dict:
    y, pred = np.asarray(y).astype(int), (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    return {
        "sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1,
        "balanced_accuracy": (sens + spec) / 2, "accuracy": (tp + tn) / len(y),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def all_metrics(y, p, thr: float) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    out = {
        "n": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
        "threshold": float(thr),
    }
    out.update(threshold_metrics(y, p, thr))
    return out


def _cluster_resample(groups, rng):
    """Indices of a cluster bootstrap sample (resample patients, keep all their images)."""
    uniq, inv = np.unique(groups, return_inverse=True)
    counts = np.bincount(inv)
    order = np.argsort(inv, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    while True:
        pick = rng.integers(0, len(uniq), len(uniq))
        if counts.max() == 1:                       # one image per patient: plain bootstrap
            yield order[pick]
            continue
        # vectorised gather of all members of the picked patients
        lens = counts[pick]
        offs = np.repeat(starts[pick] - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
        yield order[np.arange(lens.sum()) + offs]


def bootstrap_ci(y, p, thr, groups=None, n_boot=2000, seed=0, alpha=0.05,
                 keys=("auroc", "auprc", "sensitivity", "specificity", "f1")) -> dict:
    """Percentile CIs; resampling is at patient level when `groups` is given."""
    y, p = np.asarray(y), np.asarray(p)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    rng = np.random.default_rng(seed)
    gen = _cluster_resample(groups, rng)
    samples = {k: [] for k in keys}
    done = 0
    while done < n_boot:
        idx = next(gen)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        m = all_metrics(yb, p[idx], thr)
        for k in keys:
            samples[k].append(m[k])
        done += 1
    return {k: (float(np.nanpercentile(v, 100 * alpha / 2)), float(np.nanpercentile(v, 100 * (1 - alpha / 2))))
            for k, v in samples.items()}


def paired_bootstrap_delta(y, p_a, p_b, groups=None, n_boot=2000, seed=0, alpha=0.05) -> dict:
    """AUROC(b) - AUROC(a) on the same test set, with a patient-clustered paired bootstrap."""
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    gen = _cluster_resample(groups, np.random.default_rng(seed))
    deltas = []
    while len(deltas) < n_boot:
        idx = next(gen)
        if y[idx].min() == y[idx].max():
            continue
        deltas.append(roc_auc_score(y[idx], p_b[idx]) - roc_auc_score(y[idx], p_a[idx]))
    deltas = np.asarray(deltas)
    return {
        "delta_auroc": float(roc_auc_score(y, p_b) - roc_auc_score(y, p_a)),
        "ci": (float(np.percentile(deltas, 100 * alpha / 2)), float(np.percentile(deltas, 100 * (1 - alpha / 2)))),
        "p_two_sided": float(min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))),
    }
