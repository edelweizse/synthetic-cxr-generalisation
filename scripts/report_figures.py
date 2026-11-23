"""Study-2 figures for the LaTeX report (docs/report/figures/), from reports/results/catalogue_runs.csv.

    .venv/bin/python scripts/report_figures.py
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/report/figures"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREY = "#2D6CB5", "#C4551A", "#7A8090"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

r = pd.read_csv(ROOT / "reports/results/catalogue_runs.csv")
r = r[r.arch.isin(["efficientnet_b0", "resnet18", "densenet121"])]
m = r.groupby("group").rsna_test_balanced.agg(["mean", "std", "count"])


def g(name):
    return m.loc[name, "mean"], m.loc[name, "std"]


# 1. training-set size: real vs synthetic-only
fig, ax = plt.subplots(figsize=(6.2, 3.4))
real = [(250, "CX_N250"), (500, "CX_N500"), (1000, "CX_E1"), (2000, "CX_N2000"), (9968, "CX_FULL")]
syn = [(2000, "CX_SYN_ONLY_SD2C25V"), (5000, "CX_SYN_ONLY5K_SD2C25V"), (10000, "CX_SYN_ONLY10K_SD2C25V")]
for pts, c, lab in ((real, BLUE, "real CheXpert images"), (syn, ORANGE, "synthetic only (final generator)")):
    x = [p[0] for p in pts]; y = [g(p[1])[0] for p in pts]; e = [g(p[1])[1] for p in pts]
    ax.errorbar(x, y, yerr=e, color=c, marker="o", capsize=3, lw=2, label=lab)
ax.set_xscale("log"); ax.set_xticks([250, 500, 1000, 2000, 5000, 10000]); ax.set_xticklabels(["250", "500", "1k", "2k", "5k", "10k"])
ax.set_xlabel("training images"); ax.set_ylabel("RSNA-test AUROC (view-balanced)")
ax.legend(frameon=False, loc="lower left"); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "s2_scaling.pdf"); plt.close(fig)

# 2. generator versions: synthetic-only RSNA vs 2,000 real
vers = [("v1", "CX_SYN_ONLY_SD"), ("v2", "CX_SYN_ONLY_SD2V"), ("v2 CFG 2.5\n(final)", "CX_SYN_ONLY_SD2C25V"),
        ("rich\ncaptions", "CX_SYN_ONLY_SD3V"), ("rank 64", "CX_SYN_ONLY_SDR64V"), ("view\ncaptions", "CX_SYN_ONLY_SDVIEWV"),
        ("obvious\ncases", "CX_SYN_ONLY_SDOBVV"), ("3x data", "CX_SYN_ONLY_SDBIGV")]
fig, ax = plt.subplots(figsize=(6.2, 3.2))
mu = [g(v[1])[0] for v in vers]; sd = [g(v[1])[1] for v in vers]
ax.bar(range(len(vers)), mu, yerr=sd, color=[BLUE if "final" in v[0] else ORANGE for v in vers], capsize=3)
ax.axhline(g("CX_N2000")[0], color=GREY, ls="--", lw=1); ax.text(len(vers) - .5, g("CX_N2000")[0] + .005, "2,000 real", ha="right", color=GREY)
ax.set_xticks(range(len(vers))); ax.set_xticklabels([v[0] for v in vers], fontsize=8)
ax.set_ylim(.6, .95); ax.set_ylabel("RSNA-test AUROC (view-balanced)")
fig.tight_layout(); fig.savefig(OUT / "s2_generator_versions.pdf"); plt.close(fig)

# 3. final controls: images vs teacher
ctl = [("real, hard labels", "CX_N2000", BLUE), ("real, teacher labels", "CX_SYN_ONLY_RTEACH", BLUE),
       ("synthetic v1, teacher", "CX_SYN_ONLY_TEACH", ORANGE), ("synthetic final, teacher", "CX_SYN_ONLY_TEACHV2", ORANGE),
       ("synthetic final, prompt", "CX_SYN_ONLY_SD2C25", ORANGE)]
fig, ax = plt.subplots(figsize=(6.2, 2.6))
y = np.arange(len(ctl))[::-1]
ax.barh(y, [g(c[1])[0] for c in ctl], xerr=[g(c[1])[1] for c in ctl], color=[c[2] for c in ctl], capsize=3)
for yi, c in zip(y, ctl):
    ax.text(g(c[1])[0] + g(c[1])[1] + .004, yi, f"{g(c[1])[0]:.3f}", va="center", fontsize=9)
ax.axvline(g("CX_FULL")[0], color=GREY, ls="--", lw=1); ax.text(g("CX_FULL")[0], len(ctl) - .45, "all 9,968 real", ha="center", color=GREY, fontsize=8)
ax.set_yticks(y); ax.set_yticklabels([c[0] for c in ctl]); ax.set_xlim(.75, .98); ax.set_ylim(-.6, len(ctl) - .2)
ax.set_xlabel("RSNA-test AUROC (view-balanced), 2,000 training images")
fig.tight_layout(); fig.savefig(OUT / "s2_final_controls.pdf"); plt.close(fig)

# study-1 figures and the memorisation grid, copied so the report folder compiles on its own
for f in ["a_acquisition_by_class.png", "real_data_scaling.png", "synth_montage.png", "synth_dose_response.png"]:
    shutil.copy(ROOT / "reports/figures" / f, OUT / f)
shutil.copy(ROOT / "reports/results/generator/nn_pairs_cx_sd_v2c25.png", OUT / "nn_pairs_final.png")
print(sorted(p.name for p in OUT.iterdir()))
