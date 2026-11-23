"""Natural (not induced) image quality of every labelled RSNA film, joined with the seed-averaged scores of many runs.
Asks whether models trained with synthetic images rank real external films by quality rather than by opacity.
Writes reports/results/synth_probe/rsna_quality_scores.csv (one row per image; small enough to analyse anywhere)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cxr.analysis import add_b_role
from cxr.data import load_set
from cxr.engine import RUNS_DIR

torch.set_num_threads(4)
LAP = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
RUNS = ["E1", "R25", "R50", "E2", "E2_ERASE", "E3_sd_lora", "E4_sd_lora", "SYN1000_sd_lora", "SYN2000_sd_lora",
        "SYN_ONLY_sd_lora", "ERASE_REAL250", "ERASE_REAL500", "ERASE_SYN250_sd_lora", "ERASE_SYN500_sd_lora",
        "E1_HARMT", "E3_sd_lora_HARMT", "E4_sd_lora_HARMT", "LP",
        "E3_sd_lora_QM", "E4_sd_lora_QM", "E3_sd_lora_QC", "E4_sd_lora_QC"]
RUNS += [f"{e}_L{l}" for l in ["QCONS", "QDRO", "GCE", "FOCAL"] for e in ["E1", "E3_sd_lora"]]
RUNS += ["E2_ERASE_LQCONS", "ERASE_SYN250_sd_lora_LQCONS", "E1_LQCONS03", "E1_LQCONS3"]
ARCH = "efficientnet_b0"

rs = load_set("artifacts/manifests/rsna.csv", {"split": "external"}, 224, "squash", device=None, require_label=False)
out = []
for k in range(0, len(rs), 512):
    x = torch.from_numpy(np.ascontiguousarray(rs.images[k:k + 512])).float().div(255)
    x = x.unsqueeze(1) if x.ndim == 3 else x
    c = x[:, :, 32:-32, 32:-32]
    out.append(pd.DataFrame({"lap_var": F.conv2d(x, LAP).var((1, 2, 3)).numpy() * 1e3,
                             "hf_energy": (x - F.avg_pool2d(x, 5, 1, 2)).abs().mean((1, 2, 3)).numpy() * 1e2,
                             "mean": c.mean((1, 2, 3)).numpy(), "std": c.std((1, 2, 3)).numpy()}))
q = pd.concat(out, ignore_index=True)
q.insert(0, "image_id", rs.df["image_id"].values)

base = None
for run in RUNS:
    ps = []
    for seed in range(3):
        p = RUNS_DIR / run / ARCH / f"seed{seed}" / "preds_B_external.csv"
        if p.exists():
            d = pd.read_csv(p, dtype={"patient_id": str})
            if base is None:
                base = add_b_role(d[["image_id", "patient_id", "label", "label_any", "class3", "view", "age", "sex", "width"]])
            ps.append(d.set_index("image_id")["logit"])
    if ps:
        base[f"logit_{run}"] = base["image_id"].map(pd.concat(ps, axis=1).mean(axis=1))
d = base.merge(q, on="image_id", how="left")
d.to_csv(ROOT / "reports/results/synth_probe/rsna_quality_scores.csv", index=False)
print(d.shape, d.hf_energy.isna().sum(), "images without quality stats")
