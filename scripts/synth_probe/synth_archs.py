"""E1 -> E3 / E4 (+250 / +500 synthetic) and the matched real controls E1 -> R25 / R50, R25 -> E3, R50 -> E4, for every architecture, RSNA-test pooled and per view (seed-ensemble, paired
patient-clustered bootstrap). Writes reports/results/synth_probe/synth_archs.csv."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from cxr.analysis import compare

KEYS = ["from", "to", "arch", "eval", "auroc_from", "auroc_to", "delta_ensemble", "ci_lo", "ci_hi",
        "delta_per_seed_mean", "delta_per_seed_sd", "n_seeds"]
rows = []
for arch in ["efficientnet_b0", "densenet121", "vit_small", "resnet18", "resnet50", "convnext_tiny", "swin_tiny"]:
    for a, b in [("E1", "E3_sd_lora"), ("E1", "E4_sd_lora"), ("E1", "R25"), ("E1", "R50"),
                 ("R25", "E3_sd_lora"), ("R50", "E4_sd_lora")]:
        for f in ["b_role=test", "b_role=test&view=AP", "b_role=test&view=PA"]:
            r = compare(a, b, arch, f"B_external:label@{f}", n_boot=1000)
            if r:
                rows.append({k: r[k] for k in KEYS})
d = pd.DataFrame(rows)
d["eval"] = d["eval"].str.replace("B_external:label@b_role=", "rsna-")
d.to_csv(ROOT / "reports/results/synth_probe/synth_archs.csv", index=False)
pd.set_option("display.width", 250)
print(d.round(3).to_string())
