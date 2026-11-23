"""Which property of the synthetic images causes the penalty? (docs/RESEARCH_PIVOT.md)

Every comparison starts from E1 (1,000 real) on RSNA-test, view-balanced AUROC (headline) and pooled AUROC:
  R25/R50            + random real images (reference: what an added real image is worth)
  E3/E4_sd_lora      + random synthetic images (the penalty to explain)
  E3/E4_MIMIC_*      + REAL images copying one synthetic property (Q shortcut strength, N weak class signal, QN both;
                     N/QN = 30% of pneumonia slots mislabelled, N50/N70/QN50/QN70 = 50% / 70%)
  E3/E4_sd_lora_ANAT / _ANATBAD   synthetic images without the most atypical 20% anatomy / worst anatomy first
plus direct ANATBAD -> ANAT. Writes reports/results/synth_probe/pivot_compare.csv."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from cxr.analysis import HEADLINE_KEY, compare

ARCHS = ["efficientnet_b0", "resnet18", "densenet121"]
TO = {"E3": ["R25", "E3_sd_lora", "E3_MIMIC_Q", "E3_MIMIC_N", "E3_MIMIC_QN", "E3_MIMIC_N50", "E3_MIMIC_N70", "E3_MIMIC_QN50",
             "E3_MIMIC_QN70", "E3_sd_lora_ANAT", "E3_sd_lora_ANATBAD"],
      "E4": ["R50", "E4_sd_lora", "E4_MIMIC_Q", "E4_MIMIC_N", "E4_MIMIC_QN", "E4_MIMIC_N50", "E4_MIMIC_N70", "E4_MIMIC_QN50",
             "E4_MIMIC_QN70", "E4_sd_lora_ANAT", "E4_sd_lora_ANATBAD"]}
rows = []
for arch in ARCHS:
    for size, tos in TO.items():
        pairs = [("E1", t) for t in tos] + [(f"{size}_sd_lora_ANATBAD", f"{size}_sd_lora_ANAT"),
                                            (f"{size}_sd_lora", f"{size}_sd_lora_ANAT")]
        for a, b in pairs:
            for metric in ("balanced", "auroc"):
                r = compare(a, b, arch, HEADLINE_KEY, n_boot=1000, metric=metric)
                if r:
                    rows.append({**r, "size": size})
                    print(f"{arch:16s} {a:20s} -> {b:20s} {metric:8s} {r['delta_ensemble']:+.3f} "
                          f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]", flush=True)
pd.DataFrame(rows).to_csv(ROOT / "reports/results/synth_probe/pivot_compare.csv", index=False)
