"""ERASE series on RSNA-test: does adding synthetic (or real) images help on top of E2_ERASE augmentation?
Paired, patient-clustered bootstrap on seed-averaged scores (cxr.analysis.compare), pooled and per view."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from cxr.analysis import compare

PAIRS = [("E2_ERASE", "ERASE_SYN250_sd_lora"), ("E2_ERASE", "ERASE_SYN500_sd_lora"),
         ("E2_ERASE", "ERASE_REAL250"), ("E2_ERASE", "ERASE_REAL500"),
         ("ERASE_REAL250", "ERASE_SYN250_sd_lora"), ("ERASE_REAL500", "ERASE_SYN500_sd_lora")]
KEYS = ["from", "to", "arch", "eval", "auroc_from", "auroc_to", "delta_ensemble", "ci_lo", "ci_hi",
        "delta_per_seed_mean", "delta_per_seed_sd", "n_seeds"]
rows = []
for arch in ["efficientnet_b0", "vit_small", "densenet121", "resnet18", "resnet50", "convnext_tiny", "swin_tiny"]:  # missing runs skipped
    for a, b in PAIRS:
        for f in ["b_role=test", "b_role=test&view=AP", "b_role=test&view=PA"]:
            r = compare(a, b, arch, f"B_external:label@{f}", n_boot=1000)
            if r:
                rows.append({k: r[k] for k in KEYS})
d = pd.DataFrame(rows)
d["eval"] = d["eval"].str.replace("B_external:label@b_role=", "rsna-")
out = ROOT / "reports/results/synth_probe/erase_compare.csv"
d.to_csv(out, index=False)
pd.set_option("display.width", 250)
print(d.round(3).to_string())
