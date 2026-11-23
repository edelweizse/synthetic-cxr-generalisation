"""Summary figures for the progress digest (docs/digest/digest.tex), from result CSVs already on disk."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SP, OUT = ROOT / "reports/results/synth_probe", Path(__file__).parent
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
COL = {"base": "0.55", "real": "tab:green", "aug": "tab:blue", "syn": "tab:red", "loss": "tab:purple", "fix": "tab:orange"}

# ---------------------------------------------------------------- 1. scoreboard, EfficientNet-B0
M = pd.read_csv(SP / "view_metrics.csv")
e = M[M.arch == "efficientnet_b0"].set_index("run")
ROWS = [("E1", "E1  baseline (1,000 real)", "base"),
        ("R25", "R25  +250 real", "real"), ("R50", "R50  +500 real", "real"), ("FULL", "FULL  all 4,093 real", "real"),
        ("E2", "E2  conventional augmentation", "aug"), ("E2_ERASE", "E2_ERASE  augmentation + erasing", "aug"),
        ("E1_LQCONS", "E1 + qcons loss", "loss"),
        ("E3_sd_lora", "E3  +250 synthetic", "syn"), ("E4_sd_lora", "E4  +500 synthetic", "syn"),
        ("SYN2000_sd_lora", "+2,000 synthetic", "syn"), ("SYN_ONLY_sd_lora", "2,000 synthetic only", "syn"),
        ("ERASE_SYN250_sd_lora", "E2_ERASE +250 synthetic", "syn"),
        ("E4_sd_lora_HARMT", "E4, equalised + re-tuned", "fix"), ("E3_sd_lora_QC", "E3, quality-aware selection", "fix"),
        ("E3_sd_lora_LQCONS", "E3 + qcons loss", "fix"), ("ERASE_SYN250_sd_lora_LQCONS", "E2_ERASE + qcons +250 synthetic", "fix")]
ROWS = [r for r in ROWS if r[0] in e.index]
fig, ax = plt.subplots(figsize=(8.2, 5.8))
y = np.arange(len(ROWS))[::-1]
for yy, (run, lab, k) in zip(y, ROWS):
    ax.barh(yy, e.loc[run, "balanced"], color=COL[k], alpha=0.85, height=0.62)
    ax.plot(e.loc[run, "pooled"], yy, "k|", ms=11, mew=2)
    ax.text(e.loc[run, "balanced"] + 0.004, yy, f"{e.loc[run, 'balanced']:.3f}", va="center", fontsize=7.5)
ax.axvline(e.loc["E1", "balanced"], color="k", lw=0.8, ls=":")
ax.set_yticks(y, [r[1] for r in ROWS]); ax.set_xlim(0.45, 0.95)
ax.set_xlabel("RSNA-test AUROC, 3-seed ensemble   (bar = view-balanced, mean of AP and PA;  | = pooled)", labelpad=2)
handles = [plt.Rectangle((0, 0), 1, 1, color=COL[k], alpha=0.85) for k in ["base", "real", "aug", "loss", "syn", "fix"]]
ax.legend(handles, ["baseline", "+ real data", "augmentation", "robust loss", "+ synthetic", "synthetic + a fix"],
          fontsize=7.5, loc="upper center", bbox_to_anchor=(0.4, -0.1), ncol=6, frameon=False)
ax.set_title("EfficientNet-B0: synthetic images never beat the same recipe without them", fontsize=10, loc="right")
plt.tight_layout(); plt.savefig(OUT / "fig_scoreboard.png", dpi=200); plt.close()

# ---------------------------------------------------------------- 2. synthetic penalty, 7 architectures
S = pd.read_csv(SP / "synth_archs.csv")
S = S[S["from"] == "E1"]
ARCHS = ["efficientnet_b0", "resnet18", "swin_tiny", "resnet50", "convnext_tiny", "densenet121", "vit_small"]
NICE = {"efficientnet_b0": "EffNet-B0", "resnet18": "ResNet-18", "swin_tiny": "Swin-T", "resnet50": "ResNet-50",
        "convnext_tiny": "ConvNeXt-T", "densenet121": "DenseNet-121", "vit_small": "ViT-S"}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, metric in zip(axes, ["pooled", "balanced"]):
    for k, (to, c) in enumerate([("E3_sd_lora", "tab:red"), ("E4_sd_lora", "darkred")]):
        g = S[S.to == to].pivot_table(index="arch", columns="eval", values="delta_ensemble").reindex(ARCHS)
        v = g["rsna-test"] if metric == "pooled" else (g["rsna-test&view=AP"] + g["rsna-test&view=PA"]) / 2
        ax.bar(np.arange(len(ARCHS)) + (k - 0.5) * 0.38, v, 0.38, color=c, label={"E3_sd_lora": "+250 synthetic (E3)",
                                                                                "E4_sd_lora": "+500 synthetic (E4)"}[to])
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(len(ARCHS)), [NICE[a] for a in ARCHS], rotation=30, fontsize=8)
    ax.set_title(f"Δ AUROC vs E1, RSNA-test, {metric}", fontsize=9)
axes[0].set_ylabel("Δ AUROC"); axes[0].legend(fontsize=7.5, frameon=False)
plt.tight_layout(); plt.savefig(OUT / "fig_synth_archs.png", dpi=200); plt.close()

# ---------------------------------------------------------------- 3. view confound
F = pd.read_csv(SP / "view_facts.csv")
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), gridspec_kw={"width_ratios": [0.8, 1.2, 1.2]})
ax = axes[0]
ap, pa = F.iloc[0], F.iloc[1]
for x, r in [(0, ap), (1, pa)]:
    ax.bar(x, r.n_opacity, color="tab:red", alpha=0.8); ax.bar(x, r.n_normal, bottom=r.n_opacity, color="tab:blue", alpha=0.6)
    ax.text(x, r.n + 80, f"{r.prevalence:.0%} opacity", ha="center", fontsize=8)
ax.set_xticks([0, 1], ["AP films\n(bedside)", "PA films\n(standing)"]); ax.set_ylabel("RSNA-test films")
ax.legend(["opacity", "normal"], fontsize=7.5, frameon=False, loc="upper left"); ax.set_ylim(0, 7600)
ax.set_title("View is confounded with the label", fontsize=9)
ax = axes[1]
c = np.where(M.run.str.contains("sd_lora|SYN"), "tab:red", "0.45")
ax.scatter(M.pooled, M.balanced, s=10, c=c, alpha=0.7)
ax.axvline(0.796, color="k", ls=":", lw=0.8); ax.text(0.797, 0.52, "view alone\n= 0.796", fontsize=7.5)
ax.plot([0.5, 0.95], [0.5, 0.95], color="k", lw=0.5)
ax.set_xlabel("pooled AUROC"); ax.set_ylabel("view-balanced AUROC"); ax.set_xlim(0.6, 0.95); ax.set_ylim(0.5, 0.95)
ax.set_title("121 configurations: pooled inflates by 0.05–0.19\n(red = with synthetic images)", fontsize=9)
ax = axes[2]
E1 = M[M.run == "E1"].set_index("arch").reindex(ARCHS)
x = np.arange(len(ARCHS))
ax.bar(x - 0.2, E1.pair_APpos_PAneg, 0.4, color="tab:purple", label="AP opacity ranked above PA normal")
ax.bar(x + 0.2, E1.pair_PApos_APneg, 0.4, color="tab:orange", label="PA opacity ranked above AP normal")
ax.axhline(0.5, color="k", lw=0.8, ls=":"); ax.set_xticks(x, [NICE[a] for a in ARCHS], rotation=30, fontsize=8)
ax.set_ylim(0, 1.3); ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=1); ax.set_ylabel("probability (E1)")
ax.set_title("The baselines largely detect the view", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "fig_view_confound.png", dpi=200); plt.close()
print("ok")
