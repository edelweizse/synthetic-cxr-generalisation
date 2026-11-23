"""Label-noise doses for one architecture (docs/RESEARCH_PIVOT.md): Δ view-balanced AUROC vs E1 on RSNA-test for
real (R25/R50), synthetic (sd_lora) and the real-image mimics N / QN at 30%, 50%, 70% mislabelled pneumonia slots.
    CUDA_VISIBLE_DEVICES= python scripts/synth_probe/dose_compare.py resnet18
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cxr.analysis import HEADLINE_KEY, compare

arch = sys.argv[1] if len(sys.argv) > 1 else "efficientnet_b0"
for s, real in (("E3", "R25"), ("E4", "R50")):
    for t in [real, f"{s}_sd_lora"] + [f"{s}_MIMIC_{m}" for m in ("N", "N50", "N70", "QN", "QN50", "QN70")]:
        r = compare("E1", t, arch, HEADLINE_KEY, n_boot=300, metric="balanced")
        if r:
            print(f"{t:18s} {r['delta_ensemble']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] per-seed "
                  f"{r['delta_per_seed_mean']:+.3f} sd {r['delta_per_seed_sd']:.3f} seeds {r['n_seeds']}", flush=True)
