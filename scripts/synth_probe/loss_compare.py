"""Robust training losses (cxr/engine.LOSSES) on RSNA-test, pooled and per view: paired, patient-clustered bootstrap on
seed-averaged scores (compare_losses.csv), and the natural-quality ratio -beta_hf / beta_label of each run
(loss_quality_ratio.csv; needs natural_quality.py to have been re-run with the loss runs). Missing runs are skipped."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from cxr.analysis import compare

OUT = ROOT / "reports/results/synth_probe"
KEYS = ["from", "to", "eval", "auroc_from", "auroc_to", "delta_ensemble", "ci_lo", "ci_hi",
        "delta_per_seed_mean", "delta_per_seed_sd", "n_seeds"]
VIEWS = ["b_role=test", "b_role=test&view=AP", "b_role=test&view=PA"]
LOSSES = ["LQCONS", "LQDRO", "LGCE", "LFOCAL"]
PAIRS = ([("E1", f"E1_{l}") for l in LOSSES]                                # loss alone, real data
         + [("E3_sd_lora", f"E3_sd_lora_{l}") for l in LOSSES]              # loss on +250 synthetic
         + [(f"E1_{l}", f"E3_sd_lora_{l}") for l in LOSSES]                 # synthetic penalty under the loss
         + [("E1", f"E3_sd_lora_{l}") for l in LOSSES]                      # does the loss rescue synthetic data?
         + [("E2", "E1_LQCONS"), ("E2_ERASE", "E1_LQCONS"),                 # qcons vs conventional augmentation
            ("E2_ERASE", "E2_ERASE_LQCONS"), ("E2_ERASE_LQCONS", "ERASE_SYN250_sd_lora_LQCONS"),
            ("E2_ERASE", "ERASE_SYN250_sd_lora"), ("E1", "E1_LQCONS03"), ("E1", "E1_LQCONS3")])
ARCHS = ["resnet18", "resnet50", "densenet121", "convnext_tiny", "vit_small", "swin_tiny"]
ARCH_PAIRS = [(a, x, y) for a in ARCHS for x, y in [("E1", "E1_LQCONS"), ("E3_sd_lora", "E3_sd_lora_LQCONS"),
                                                   ("E1_LQCONS", "E3_sd_lora_LQCONS"), ("E1", "E3_sd_lora_LQCONS")]]

rows = []
for arch, a, b in [("efficientnet_b0", a, b) for a, b in PAIRS] + ARCH_PAIRS:
    for f in VIEWS:
        r = compare(a, b, arch, f"B_external:label@{f}", n_boot=1000)
        if r:
            rows.append({"arch": arch, **{k: r[k] for k in KEYS}})
d = pd.DataFrame(rows)
d["eval"] = d["eval"].str.replace("B_external:label@b_role=", "rsna-")
d.to_csv(OUT / "compare_losses.csv", index=False)
pd.set_option("display.width", 250)
print(d.drop(columns=["delta_per_seed_sd"]).round(3).to_string())

# natural-quality ratio, same regression as §9 (standardised logit on label, z(hf), z(std), z(mean), z(age), per view)
q = pd.read_csv(OUT / "rsna_quality_scores.csv")
t = q[q.label.notna() & (q.b_role == "test")]
runs = [c[6:] for c in t if c.startswith("logit_") and (c[6:] in {"E1", "E2", "E2_ERASE", "E3_sd_lora"} or "_L" in c)]
Z = {c: ((t[c] - t[c].mean()) / t[c].std()).values for c in ["hf_energy", "std", "mean", "age"]}
X = np.column_stack([np.ones(len(t)), t.label.values, Z["hf_energy"], Z["std"], Z["mean"], Z["age"]])
rng, out = np.random.default_rng(0), []
for v in ["AP", "PA"]:
    m = (t.view == v).values
    idx = np.where(m)[0]
    for r in runs:
        y = t[f"logit_{r}"].values
        y = (y - y[m].mean()) / y[m].std()

        def ratio(ii):
            b = np.linalg.lstsq(X[ii], y[ii], rcond=None)[0]
            return -b[2] / b[1], -b[3] / b[1]

        est, bs = ratio(idx), np.array([ratio(rng.choice(idx, len(idx))) for _ in range(300)])
        out.append(dict(run=r, view=v, hf_per_label=est[0], hf_lo=np.percentile(bs[:, 0], 2.5),
                        hf_hi=np.percentile(bs[:, 0], 97.5), std_per_label=est[1],
                        std_lo=np.percentile(bs[:, 1], 2.5), std_hi=np.percentile(bs[:, 1], 97.5)))
R = pd.DataFrame(out)
R.to_csv(OUT / "loss_quality_ratio.csv", index=False)
print(R.round(2).to_string())
