"""Counterfactual-pair training results (docs/RESEARCH_PIVOT.md §7): Δ vs E1 on RSNA-test for + real (R25/R50),
+ sampled synthetic (sd_lora) and + counterfactual pairs (CF_S10 / CF_S08): view-balanced, AP-only and PA-only AUROC.
    CUDA_VISIBLE_DEVICES= python scripts/synth_cf/cf_compare.py   -> reports/results/synth_probe/cf_compare.csv"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from cxr.analysis import HEADLINE_KEY, compare

rows = []
for arch in ["efficientnet_b0", "resnet18", "densenet121"]:
    for s, real in (("E3", "R25"), ("E4", "R50")):
        for t in [real, f"{s}_sd_lora", f"{s}_CF_S10", f"{s}_CF_S08"]:
            r = {"arch": arch, "size": s, "to": t}
            for name, key, metric in (("balanced", HEADLINE_KEY, "balanced"), ("AP", HEADLINE_KEY + "&view=AP", "auroc"),
                                      ("PA", HEADLINE_KEY + "&view=PA", "auroc")):
                c = compare("E1", t, arch, key, n_boot=300, metric=metric)
                if not c:
                    break
                r.update({name: c["delta_ensemble"], f"{name}_lo": c["ci_lo"], f"{name}_hi": c["ci_hi"],
                          f"{name}_seed_sd": c["delta_per_seed_sd"], "seeds": c["n_seeds"]})
            if "balanced" in r:
                rows.append(r)
                print(f"{arch:16s} {t:12s} bal {r['balanced']:+.3f} [{r['balanced_lo']:+.3f}, {r['balanced_hi']:+.3f}] "
                      f"AP {r.get('AP', float('nan')):+.3f} PA {r.get('PA', float('nan')):+.3f} "
                      f"seed-sd {r['balanced_seed_sd']:.3f} seeds {r['seeds']}", flush=True)
pd.DataFrame(rows).round(4).to_csv(ROOT / "reports/results/synth_probe/cf_compare.csv", index=False)
