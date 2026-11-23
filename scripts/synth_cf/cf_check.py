"""Sanity checks of counterfactual pairs (docs/RESEARCH_PIVOT.md §7), usable while cf_generate.py is still running.

  CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/synth_cf/cf_check.py cf_lung_s10
1. label signal: robust real-trained classifiers (E2_ERASE, seed 0) score both edits of each pair. Does the "pneumonia" edit
   score higher than the "normal" edit of the same source image? And does the score follow the prompt or the source image?
2. quality cue: within the CF set, do low sharpness / low contrast still predict the label (sd_lora: AUROC 0.88-0.90)?
Writes reports/results/synth_probe/cf_check_<name>.csv (one row per pair) and prints the summary."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from cxr.data import load_set
from cxr.engine import load_model, predict_logits, sigmoid, RUNS_DIR

torch.set_num_threads(4)
name = sys.argv[1] if len(sys.argv) > 1 else "cf_lung_s10"
m = pd.read_csv(ROOT / "data/synthetic" / name / "manifest.csv")
m = pd.DataFrame({"label": (m.label == "PNEUMONIA").astype(int),
                  "image_id": [f"S/{name}/{r}" for r in m.relative_path.str.replace(".png", "")],
                  "image_path": [f"data/synthetic/{name}/{r}" for r in m.relative_path],
                  "source": f"synthetic:{name}", "split": "train", "patient_id": m.source_id.astype(str),
                  "synth_rank": m.pair, "source_label": m.source_label})
part = f"artifacts/manifests/_partial_{name}.csv"  # snapshot of the pairs generated so far
m.to_csv(ROOT / part, index=False)
ts = load_set(part, None, 224, "squash", device=None)
df = ts.df.copy()
x = torch.from_numpy(np.ascontiguousarray(ts.images[:])).float().div(255)
df["hf_energy"] = (x - F.avg_pool2d(x, 5, 1, 2)).abs().mean((1, 2, 3)).numpy() * 1e2
df["std"] = x[:, :, 32:-32, 32:-32].std((1, 2, 3)).numpy()
dev = torch.device("cpu")
for arch in ["efficientnet_b0", "densenet121"]:
    model, _ = load_model(RUNS_DIR / "E2_ERASE" / arch / "seed0", dev)
    df[f"p_{arch}"] = sigmoid(predict_logits(model, ts, dev, batch_size=32, amp=False))
    del model

y = df.label.to_numpy()
print(f"{name}: {df.synth_rank.nunique()} pairs")
w = df.pivot(index="synth_rank", columns="label")
for arch in ["efficientnet_b0", "densenet121"]:
    p = f"p_{arch}"
    d = w[p][1] - w[p][0]
    src = w["source_label"][0]
    print(f"  {arch}: mean p normal-edit {w[p][0].mean():.2f}, pneumonia-edit {w[p][1].mean():.2f}; "
          f"pneumonia-edit higher in {(d > 0).mean():.0%} of pairs (median Δp {d.median():+.2f}); "
          f"label AUROC {roc_auc_score(y, df[p]):.3f}, source-label AUROC {roc_auc_score(df.source_label, df[p]):.3f}")
    for s in (0, 1):
        print(f"      source {'pneumonia' if s else 'normal   '}: p normal-edit {w[p][0][src == s].mean():.2f}, "
              f"pneumonia-edit {w[p][1][src == s].mean():.2f}")
print(f"  quality cue: low-sharpness AUROC {roc_auc_score(y, -df.hf_energy):.3f}, "
      f"low-contrast AUROC {roc_auc_score(y, -df['std']):.3f}  (sd_lora 0.88 / 0.90; 0.5 = no cue)")
df.drop(columns=["image_path"]).to_csv(ROOT / f"reports/results/synth_probe/cf_check_{name}.csv", index=False)
