"""Probe the teammate's SD-LoRA synthetic images (runs on CPU so it doesn't fight the training job).

A. label signal: do real-trained classifiers see pneumonia in synthetic 'PNEUMONIA' images?
B. provenance / memorisation: nearest real-train neighbour of each synthetic image (feature space)
C. realism: how separable are synthetic vs real images?
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch
import timm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

from cxr.data import load_set
from cxr.engine import load_model, predict_logits, sigmoid, RUNS_DIR

torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
dev = torch.device("cpu")
A, S = "artifacts/manifests/kermany.csv", "artifacts/manifests/synthetic_sd_lora.csv"
OUT = ROOT / "reports/results/synth_probe"

syn = load_set(S, None, 224, "squash", device=None)
trn = load_set(A, {"split": "train"}, 224, "squash", device=None)
val = load_set(A, {"split": "val"}, 224, "squash", device=None)
print("synthetic", len(syn), "real train", len(trn), "real val", len(val), flush=True)

# ---- A. label signal
rows = []
for arch in ["efficientnet_b0", "densenet121", "vit_small"]:
    for exp in ["E1", "E2", "E2_ERASE"]:
        rd = RUNS_DIR / exp / arch / "seed0"
        model, _ = load_model(rd, dev)
        p = sigmoid(predict_logits(model, syn, dev, batch_size=64, amp=False))
        y = syn.labels.numpy()
        pv = sigmoid(predict_logits(model, val, dev, batch_size=64, amp=False))
        yv = val.labels.numpy()
        rows.append(dict(arch=arch, exp=exp, auroc_synth=roc_auc_score(y, p), auroc_realval=roc_auc_score(yv, pv),
                         p_synth_normal=p[y == 0].mean(), p_synth_pneu=p[y == 1].mean(),
                         p_real_normal=pv[yv == 0].mean(), p_real_pneu=pv[yv == 1].mean()))
        print(rows[-1], flush=True)
        pd.DataFrame({"image_id": syn.df.image_id, "label": y, f"p_{exp}_{arch}": p}).to_csv(
            OUT / f"pred_{exp}_{arch}.csv", index=False)
        del model
pd.DataFrame(rows).to_csv(OUT / "label_signal.csv", index=False)


# ---- B/C. features: ImageNet EfficientNet-B0 (generic) and E2_ERASE EfficientNet-B0 (CXR-tuned)
@torch.no_grad()
def feats(net_fn, tset):
    out = []
    for x in tset.batches(64, dev):
        out.append(net_fn(x).float())
    f = torch.cat(out)
    return torch.nn.functional.normalize(f, dim=1).numpy()


imnet = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0).eval()
mean = torch.tensor(imnet.pretrained_cfg["mean"]).view(1, 3, 1, 1)
std = torch.tensor(imnet.pretrained_cfg["std"]).view(1, 3, 1, 1)
tuned, _ = load_model(RUNS_DIR / "E2_ERASE" / "efficientnet_b0" / "seed0", dev)
spaces = {
    "imagenet": lambda x: imnet((x.expand(-1, 3, -1, -1) - mean) / std),
    "cxr_tuned": lambda x: tuned.net.forward_head(tuned.net.forward_features((x.expand(-1, 3, -1, -1) - tuned.mean) / tuned.std), pre_logits=True),
}
in1k = trn.df.subset_1k.astype(bool).to_numpy()
for name, fn in spaces.items():
    fs, ft, fv = feats(fn, syn), feats(fn, trn), feats(fn, val)
    np.savez_compressed(ROOT / f"artifacts/cache/synth_probe_feats_{name}.npz", syn=fs, trn=ft, val=fv)
    sim = fs @ ft.T
    nn_idx, nn_sim = sim.argmax(1), sim.max(1)
    # real->real reference: val images vs train (different patients)
    ref = (fv @ ft.T).max(1)
    tt = ft @ ft.T
    np.fill_diagonal(tt, -1)
    same_grp = trn.df.group_id.to_numpy()
    tt[same_grp[:, None] == same_grp[None, :]] = -1  # exclude near-duplicate group
    ref_tt = tt.max(1)
    frac1k_all = in1k[nn_idx].mean()
    frac1k_top = in1k[nn_idx[nn_sim >= np.quantile(nn_sim, 0.9)]].mean()
    nn_label_agree = (trn.labels.numpy()[nn_idx] == syn.labels.numpy()).mean()
    uniq = len(np.unique(nn_idx))
    print(f"[{name}] synth->train NN sim: median {np.median(nn_sim):.4f} p99 {np.quantile(nn_sim, .99):.4f} max {nn_sim.max():.4f}"
          f" | val->train median {np.median(ref):.4f} p99 {np.quantile(ref, .99):.4f}"
          f" | train->train(other grp) median {np.median(ref_tt):.4f} p99 {np.quantile(ref_tt, .99):.4f}", flush=True)
    print(f"[{name}] NN in subset_1k: all {frac1k_all:.3f}, top-10% closest {frac1k_top:.3f} (base rate {in1k.mean():.3f});"
          f" NN label agrees {nn_label_agree:.3f}; unique NN targets {uniq}/{len(nn_idx)}", flush=True)
    pd.DataFrame({"image_id": syn.df.image_id, "label": syn.labels.numpy(), "nn_sim": nn_sim,
                  "nn_image_id": trn.df.image_id.to_numpy()[nn_idx], "nn_in_1k": in1k[nn_idx],
                  "nn_label": trn.labels.numpy()[nn_idx]}).to_csv(OUT / f"nn_{name}.csv", index=False)
    # C. real vs synthetic separability (5-fold logistic regression on features)
    X = np.vstack([fs, ft]); yr = np.r_[np.ones(len(fs)), np.zeros(len(ft))]
    pr = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0), X, yr, cv=5, method="predict_proba")[:, 1]
    print(f"[{name}] real-vs-synthetic linear AUROC {roc_auc_score(yr, pr):.4f}", flush=True)
    # class separability inside synthetic vs inside real (same probe)
    for tag, F, Y in [("synthetic", fs, syn.labels.numpy()), ("real", ft, trn.labels.numpy())]:
        pc = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0), F, Y, cv=5, method="predict_proba")[:, 1]
        print(f"[{name}] normal-vs-pneumonia linear AUROC within {tag}: {roc_auc_score(Y, pc):.4f}", flush=True)
print("PROBE DONE", flush=True)
