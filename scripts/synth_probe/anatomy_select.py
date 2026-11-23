"""Anatomy plausibility of synthetic images (docs/RESEARCH_PIVOT.md arm C), from anatomy.py's segmentation features.

Score = mean over the large structures (lungs, heart, diaphragm, mediastinum, spine, scapulae, trachea) of the robust z of
the mask area and centroid against REAL Kermany images of the same class (|z| clipped at 10; an absent structure scores 10).
High = anatomically atypical. Real images are scored leave-one-in (their own reference), which only flatters them slightly.

  python scripts/synth_probe/anatomy_select.py grid          # PNG grid: most atypical / random synthetic / real, with masks
  python scripts/synth_probe/anatomy_select.py select 0.2    # manifests: worst 20% per class dropped (anat) / worst first (anatbad)
Writes reports/results/synth_probe/anatomy_score.csv and, for `select`, synthetic_sd_lora_anat.csv / _anatbad.csv:
plausible images keep their random E3/E4 order (new per-class synth_rank), implausible ones are dropped."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd

from cxr.data import read_manifest
from cxr.experiments import A

S = ["Left_Lung", "Right_Lung", "Heart", "Facies_Diaphragmatica", "Mediastinum", "Spine", "Left_Scapula", "Right_Scapula",
     "Weasand"]
SP = ROOT / "reports/results/synth_probe"
SYN = "artifacts/manifests/synthetic_sd_lora.csv"

d = pd.read_csv(SP / "anatomy.csv")
parts = []
for lab, g in d.groupby("label"):
    ref = g[g.set == "real"]
    z = []
    for n in S:
        present = g[f"area_{n}"] >= 0.002
        for k in ("area", "cy", "cx"):
            col = f"{k}_{n}"
            r = ref.loc[ref[f"area_{n}"] >= 0.002, col]
            med, mad = r.median(), 1.4826 * (r - r.median()).abs().median()
            zz = ((g[col] - med) / max(mad, 1e-3)).abs().clip(upper=10)
            z.append(zz.where(present, 10.0))
    parts.append(g.assign(anat_score=np.mean(z, axis=0)))
sc = pd.concat(parts)
sc[["image_id", "label", "set", "synth_rank", "anat_score"]].to_csv(SP / "anatomy_score.csv", index=False)
q = sc.groupby("set").anat_score.describe(percentiles=[0.5, 0.9, 0.95, 0.99])
print(q.round(2).to_string())
for p in (0.9, 0.95, 0.99):
    for lab in (0, 1):
        g = sc[sc.label == lab]
        t = g[g.set == "real"].anat_score.quantile(p)
        print(f"label {lab}: synthetic above real q{p:.2f} ({t:.2f}): {(g[g.set == 'synthetic'].anat_score > t).mean():.1%}")

if len(sys.argv) > 1 and sys.argv[1] == "grid":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torchxrayvision as xrv
    from cxr.imaging import load_gray, to_square

    paths = pd.concat([read_manifest(SYN), read_manifest(A)]).set_index("image_id").image_path
    syn = sc[sc.set == "synthetic"]
    rng = np.random.default_rng(0)
    show = [("most atypical synthetic", syn.nlargest(12, "anat_score")),
            ("random synthetic", syn.iloc[rng.choice(len(syn), 12, replace=False)]),
            ("real Kermany", sc[sc.set == "real"].iloc[rng.choice(600, 6, replace=False)])]
    model = xrv.baseline_models.chestx_det.PSPNet().eval()
    keep = [model.targets.index(n.replace("_", " ")) for n in ["Heart", "Left Lung", "Right Lung", "Spine"]]
    fig, axes = plt.subplots(5, 6, figsize=(15, 13))
    cells = [(t, r) for t, g in show for _, r in g.iterrows()]
    for ax, (title, r) in zip(axes.flat, cells):
        img = to_square(load_gray(ROOT / paths[r.image_id]), 512, "squash")
        with torch.no_grad():
            m = torch.sigmoid(model(torch.from_numpy(img).float().div(255).mul(2048).sub(1024)[None, None]))[0].numpy() > 0.5
        ax.imshow(img, cmap="gray")
        for k, c in zip(keep, ["red", "cyan", "cyan", "yellow"]):
            ax.contour(m[k], levels=[0.5], colors=c, linewidths=0.6)
        ax.set_title(f"{title}\n{'pneu' if r.label else 'norm'}  score {r.anat_score:.1f}", fontsize=7)
        ax.axis("off")
    for ax in axes.flat[len(cells):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(SP / "anatomy_grid.png", dpi=80)
    print("grid written")

if len(sys.argv) > 2 and sys.argv[1] == "select":
    # Real pneumonia films segment messily (opacities), so a real-image threshold flags no synthetic pneumonia; instead drop
    # the most atypical DROP share of the scored synthetic images within each class (same share per class, no class confound).
    drop = float(sys.argv[2])
    man, syn = read_manifest(SYN), sc[sc.set == "synthetic"]
    cut = syn.groupby("label").anat_score.transform(lambda v: v.quantile(1 - drop))
    good = man[man.image_id.isin(syn.image_id[syn.anat_score <= cut])].sort_values("synth_rank")
    good = good.assign(synth_rank=good.groupby("label").cumcount())
    good.to_csv(ROOT / "artifacts/manifests/synthetic_sd_lora_anat.csv", index=False)
    # counter: most atypical first, so E3 / E4 get the worst-anatomy images
    bad = man.merge(syn[["image_id", "anat_score"]], on="image_id").sort_values("anat_score", ascending=False)
    bad = bad.assign(synth_rank=bad.groupby("label").cumcount()).drop(columns="anat_score")
    bad.to_csv(ROOT / "artifacts/manifests/synthetic_sd_lora_anatbad.csv", index=False)
    qual = pd.read_csv(SP / "synth_quality.csv")
    from sklearn.metrics import roc_auc_score
    for name, m in [("random", man), ("anat", good), ("anatbad", bad)]:
        for n in (125, 250):
            g = m[m.synth_rank < n].merge(qual.drop(columns="label"), on="image_id").merge(
                sc[["image_id", "anat_score"]], on="image_id", how="left")
            print(f"{name:8s} per class {n}: n={len(g)} low-sharpness AUROC {roc_auc_score(g.label, -g.hf_energy):.3f} "
                  f"low-contrast AUROC {roc_auc_score(g.label, -g['std']):.3f} mean anat score {g.anat_score.mean():.2f}")
