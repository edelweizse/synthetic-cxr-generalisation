"""EfficientNet-B0 synthetic-data comparisons on RSNA-test (paired, patient-clustered bootstrap on seed-averaged scores),
pooled and per view. Writes compare_btest.csv (standard and harmonised preprocessing), compare_harmt.csv (re-tuned
harmonised preprocessing) and dose_response.csv (250 ... 2,000 synthetic images, synthetic only)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from cxr.analysis import compare

ARCH = "efficientnet_b0"
OUT = ROOT / "reports/results/synth_probe"
KEYS = ["from", "to", "eval", "auroc_from", "auroc_to", "delta_ensemble", "ci_lo", "ci_hi",
        "delta_per_seed_mean", "delta_per_seed_sd", "n_seeds"]
VIEWS = ["b_role=test", "b_role=test&view=AP", "b_role=test&view=PA"]


def table(pairs, extra=None):
    rows = []
    for a, b in pairs:
        for f in VIEWS:
            r = compare(a, b, ARCH, f"B_external:label@{f}", n_boot=1000)
            if r:
                rows.append({**(extra or {}).get(b, {}), **{k: r[k] for k in KEYS}})
    d = pd.DataFrame(rows)
    d["eval"] = d["eval"].str.replace("B_external:label@b_role=", "rsna-")
    return d


BTEST = [("E1", "E3_sd_lora"), ("E1", "E4_sd_lora"), ("R25", "E3_sd_lora"), ("R50", "E4_sd_lora"),
         ("E1", "R25"), ("E1", "R50"), ("E1", "E2_ERASE"),
         ("E1_HARM", "E3_sd_lora_HARM"), ("E1_HARM", "E4_sd_lora_HARM"),
         ("E3_sd_lora", "E3_sd_lora_HARM"), ("E4_sd_lora", "E4_sd_lora_HARM"),
         ("E1", "E1_HARM"), ("E2", "E2_HARM"), ("E2_ERASE", "E2_ERASE_HARM"),
         ("E2_ERASE_HARM", "ERASE_SYN250_sd_lora_HARM"), ("E2_ERASE_HARM", "ERASE_SYN500_sd_lora_HARM")]
HARMT = [("E1", "E1_HARMT"), ("E1_HARM", "E1_HARMT"), ("E2_ERASE", "E2_ERASE_HARMT"),
         ("E1_HARMT", "E3_sd_lora_HARMT"), ("E1_HARMT", "E4_sd_lora_HARMT")]
DOSE = {"E3_sd_lora": 250, "E4_sd_lora": 500, "SYN1000_sd_lora": 1000, "SYN2000_sd_lora": 2000, "SYN_ONLY_sd_lora": "only"}

pd.set_option("display.width", 250)
for name, d in [("compare_btest", table(BTEST)), ("compare_harmt", table(HARMT)),
                ("dose_response", table([("E1", e) for e in DOSE], {e: {"n_synth": n} for e, n in DOSE.items()}))]:
    d.to_csv(OUT / f"{name}.csv", index=False)
    print(f"== {name}\n{d.round(3).to_string()}")
