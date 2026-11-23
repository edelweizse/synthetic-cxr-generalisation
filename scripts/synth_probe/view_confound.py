"""Is the acquisition view (AP vs PA) confounding the external RSNA evaluation, and do the conclusions survive a view-balanced
metric? RSNA-test, opacity vs normal, seed-ensemble scores of every run on disk.

Writes to reports/results/synth_probe/:
  view_facts.csv       counts and prevalence per view, how well view alone / sharpness alone predict opacity (pooled vs within view)
  view_metrics.csv     per run x arch: pooled, AP, PA, view-balanced (mean of AP and PA), cell-reweighted pooled AUROC, the four
                       pair-type AUROCs that make up the pooled AUROC, and how strongly the score separates AP from PA normals
  view_compare.csv     key comparisons: delta pooled / AP / PA / balanced, balanced with patient-clustered bootstrap 95% CI
  view_selection.csv   ranking of EfficientNet-B0 configurations on RSNA-dev, pooled vs balanced
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from cxr.analysis import seed_predictions
from cxr.engine import RUNS_DIR
from cxr.metrics import _cluster_resample

OUT = ROOT / "reports/results/synth_probe"
N_BOOT = 1000


def auc(y, s, w=None):
    return roc_auc_score(y, s, sample_weight=w) if 0 < y.mean() < 1 else np.nan


def pair_auc(pos, neg):
    """P(score of a random positive > score of a random negative), ties counted half."""
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return roc_auc_score(y, np.r_[pos, neg])


def load(exp, arch, role="test"):
    d = seed_predictions(exp, arch, "B_external")
    if d.empty:
        return d
    d = d[d.label.notna() & (d.b_role == role) & d.view.isin(["AP", "PA"])].reset_index(drop=True)
    return d.assign(y=d.label.astype(int), ap=(d.view == "AP").astype(int))


def metrics(d, s="prob_mean"):
    y, ap, x = d.y.values, d.ap.values, d[s].values
    cell = pd.Series(list(zip(ap, y)))
    w = 1.0 / cell.map(cell.value_counts()).values           # each view x label cell gets the same total weight
    m_ap, m_pa = ap == 1, ap == 0
    r = {"pooled": auc(y, x), "AP": auc(y[m_ap], x[m_ap]), "PA": auc(y[m_pa], x[m_pa]), "reweighted": auc(y, x, w),
         "pair_APpos_PAneg": pair_auc(x[m_ap & (y == 1)], x[m_pa & (y == 0)]),
         "pair_PApos_APneg": pair_auc(x[m_pa & (y == 1)], x[m_ap & (y == 0)]),
         "view_sep_normals": auc(ap[y == 0], x[y == 0]), "view_sep_opacity": auc(ap[y == 1], x[y == 1])}
    r["balanced"] = (r["AP"] + r["PA"]) / 2
    r["inflation"] = r["pooled"] - r["balanced"]
    return r


# ---------------------------------------------------------------- 1. facts
ref = load("E1", "efficientnet_b0")
facts = []
for v, g in ref.groupby("view"):
    facts.append(dict(item=f"{v} films", n=len(g), n_opacity=int(g.y.sum()), n_normal=int((1 - g.y).sum()), prevalence=g.y.mean()))
facts.append(dict(item="view alone (AP = 1) as opacity score, pooled AUROC", value=auc(ref.y.values, ref.ap.values)))
q = pd.read_csv(OUT / "rsna_quality_scores.csv", usecols=["image_id", "hf_energy", "std"])
rq = ref.merge(q, on="image_id")
for c, name in [("hf_energy", "low sharpness"), ("std", "low contrast")]:
    facts.append(dict(item=f"{name} as opacity score, pooled", value=auc(rq.y.values, -rq[c].values)))
    for v, g in rq.groupby("view"):
        facts.append(dict(item=f"{name} as opacity score, within {v}", value=auc(g.y.values, -g[c].values)))
    facts.append(dict(item=f"{name} as AP-vs-PA score", value=auc(rq.ap.values, -rq[c].values)))
# precision of each view's AUROC: patient-clustered bootstrap SD for E1
gen, bs = _cluster_resample(ref.patient_id.values, np.random.default_rng(0)), []
while len(bs) < N_BOOT:
    b = ref.iloc[next(gen)]
    bs.append([auc(g.y.values, g.prob_mean.values) for _, g in b.groupby("view")] + [auc(b.y.values, b.prob_mean.values)])
for k, v in zip(["AP", "PA", "pooled"], np.nanstd(np.array(bs), axis=0)):
    facts.append(dict(item=f"E1 bootstrap SD of AUROC, {k}", value=v))
F = pd.DataFrame(facts)
F.to_csv(OUT / "view_facts.csv", index=False)
pd.set_option("display.width", 250)
print(F.round(3).to_string())

# ---------------------------------------------------------------- 2. every run
rows = []
for run in sorted(p.name for p in RUNS_DIR.iterdir() if p.is_dir() and "PLACEHOLDER" not in p.name):
    for arch in sorted(p.name for p in (RUNS_DIR / run).iterdir() if p.is_dir()):
        d = load(run, arch)
        if len(d):
            rows.append(dict(run=run, arch=arch, n_seeds=sum(c.startswith("seed") for c in d), **metrics(d)))
M = pd.DataFrame(rows)
M.to_csv(OUT / "view_metrics.csv", index=False)
print("\n", M.round(3).to_string())
print("\nSpearman(pooled, balanced) over all configurations:", round(spearmanr(M.pooled, M.balanced).statistic, 3))
print("Spearman(inflation, view separation of normals):", round(spearmanr(M.inflation, M.view_sep_normals).statistic, 3))

# ---------------------------------------------------------------- 3. key comparisons, balanced metric with CI
EFF = "efficientnet_b0"
ARCHS = ["resnet18", "resnet50", "densenet121", "efficientnet_b0", "convnext_tiny", "vit_small", "swin_tiny"]
PAIRS = ([(EFF, "E1", b) for b in ["R25", "R50", "E2", "E2_ERASE", "E3_sd_lora", "E4_sd_lora", "SYN1000_sd_lora",
                                     "SYN2000_sd_lora", "SYN_ONLY_sd_lora", "E3_sd_lora_QM", "E4_sd_lora_QM", "E3_sd_lora_QC",
                                     "E4_sd_lora_QC", "E1_LQCONS", "E1_LQDRO", "E1_LFOCAL", "E1_LGCE", "E3_sd_lora_LQCONS"]]
         + [(EFF, "R25", "E3_sd_lora"), (EFF, "R50", "E4_sd_lora"), (EFF, "E2_ERASE", "ERASE_SYN250_sd_lora"),
            (EFF, "E2_ERASE", "ERASE_SYN500_sd_lora"), (EFF, "E2_ERASE", "ERASE_REAL500"), (EFF, "E1_HARMT", "E4_sd_lora_HARMT"),
            (EFF, "E1_LQCONS", "E3_sd_lora_LQCONS"), (EFF, "E2_ERASE_LQCONS", "ERASE_SYN250_sd_lora_LQCONS"),
            (EFF, "E2", "E1_LQCONS")]
         + [(a, "E1", e) for a in ARCHS for e in ["E3_sd_lora", "E4_sd_lora"]]
         + [(a, "E1", "E2_ERASE") for a in ARCHS]
         + [(a, x, y) for a in ARCHS if a != EFF for x, y in [("E1", "R25"), ("E1", "R50"), ("R25", "E3_sd_lora"),
                                                              ("R50", "E4_sd_lora"), ("E1", "E1_LQCONS"),
                                                              ("E1_LQCONS", "E3_sd_lora_LQCONS"), ("E2_ERASE", "ERASE_SYN250_sd_lora"),
                                                              ("E2_ERASE", "ERASE_REAL250")]])
cache, out = {}, []
for arch, a, b in PAIRS:
    for k in [(a, arch), (b, arch)]:
        if k not in cache:
            cache[k] = load(*k)
    da, db = cache[(a, arch)], cache[(b, arch)]
    if da.empty or db.empty:
        continue
    d = da[["image_id", "patient_id", "y", "ap", "prob_mean"]].merge(db[["image_id", "prob_mean"]], on="image_id",
                                                                      suffixes=("_a", "_b"))
    ma, mb = metrics(d, "prob_mean_a"), metrics(d, "prob_mean_b")
    gen, deltas = _cluster_resample(d.patient_id.values, np.random.default_rng(0)), []
    y, ap, sa, sb = d.y.values, d.ap.values, d.prob_mean_a.values, d.prob_mean_b.values
    while len(deltas) < N_BOOT:
        i = next(gen)
        yi, api = y[i], ap[i]
        bal = lambda s: np.mean([auc(yi[api == v], s[i][api == v]) for v in (0, 1)])
        deltas.append(bal(sb) - bal(sa))
    deltas = np.asarray(deltas)
    r = dict(arch=arch, **{"from": a, "to": b}, balanced_from=ma["balanced"], balanced_to=mb["balanced"],
             d_pooled=mb["pooled"] - ma["pooled"], d_AP=mb["AP"] - ma["AP"], d_PA=mb["PA"] - ma["PA"],
             d_reweighted=mb["reweighted"] - ma["reweighted"], d_balanced=mb["balanced"] - ma["balanced"],
             bal_lo=np.nanpercentile(deltas, 2.5), bal_hi=np.nanpercentile(deltas, 97.5))
    r["sign_flip"] = bool(np.sign(r["d_pooled"]) != np.sign(r["d_balanced"]) and (r["bal_lo"] > 0 or r["bal_hi"] < 0))
    out.append(r)
C = pd.DataFrame(out)
C.to_csv(OUT / "view_compare.csv", index=False)
print("\n", C.round(3).to_string())
print("\nsignificant sign flips pooled vs balanced:", int(C.sign_flip.sum()), "of", len(C))

# ---------------------------------------------------------------- 4. selection on RSNA-dev: does the winner change?
sel = []
for run in sorted(M[M.arch == EFF].run):
    d = load(run, EFF, role="dev")
    if len(d):
        m = metrics(d)
        sel.append(dict(run=run, dev_pooled=m["pooled"], dev_balanced=m["balanced"], dev_AP=m["AP"], dev_PA=m["PA"]))
S = pd.DataFrame(sel)
S["rank_pooled"], S["rank_balanced"] = S.dev_pooled.rank(ascending=False), S.dev_balanced.rank(ascending=False)
S = S.sort_values("rank_balanced")
S.to_csv(OUT / "view_selection.csv", index=False)
print("\n", S.round(3).head(15).to_string())
print("Spearman(dev pooled, dev balanced):", round(spearmanr(S.dev_pooled, S.dev_balanced).statistic, 3))
