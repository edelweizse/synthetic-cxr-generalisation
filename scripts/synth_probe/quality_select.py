"""Quality-aware selection of synthetic images: can choosing *which* synthetic images to add cancel the
"soft / low-contrast = pneumonia" shortcut (FINDINGS_SYNTHETIC §3, §9)?

Writes two manifests with the same 5,000 images and a new per-class `synth_rank` (E3 takes rank < 125, E4 rank < 250):
  synthetic_sd_lora_qm.csv  quality-MATCHED: greedy nearest-neighbour pairs (pneumonia, normal) in standardised
                            (sharpness, contrast); the closest pairs get the lowest ranks, so small subsets are matched.
  synthetic_sd_lora_qc.csv  quality-COUNTER: sharpest / most contrasty pneumonia first, softest / flattest normals first,
                            so in the selected subset quality points the *other* way than in Kermany.
Also writes reports/results/synth_probe/synth_quality.csv (stats of every synthetic image)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from cxr.data import load_set, read_manifest

torch.set_num_threads(4)
SRC = "artifacts/manifests/synthetic_sd_lora.csv"

syn = load_set(SRC, None, 224, "squash", device=None)
x = torch.from_numpy(np.ascontiguousarray(syn.images[:])).float().div(255)
x = x.unsqueeze(1) if x.ndim == 3 else x
q = pd.DataFrame({"image_id": syn.df["image_id"].values, "label": syn.labels.numpy(),
                  "hf_energy": (x - F.avg_pool2d(x, 5, 1, 2)).abs().mean((1, 2, 3)).numpy() * 1e2,
                  "std": x[:, :, 32:-32, 32:-32].std((1, 2, 3)).numpy()})
q.to_csv(ROOT / "reports/results/synth_probe/synth_quality.csv", index=False)
z = np.column_stack([(q[c] - q[c].mean()) / q[c].std() for c in ["hf_energy", "std"]])

man = read_manifest(SRC)
assert set(man.image_id) == set(q.image_id)
pos, neg = np.where(q.label == 1)[0], np.where(q.label == 0)[0]

# matched: greedy global nearest-neighbour pairing
dist = ((z[pos][:, None, :] - z[neg][None, :, :]) ** 2).sum(-1)
order = np.argsort(dist, axis=None)
used_p, used_n, rank = set(), set(), {}
for flat in order:
    i, j = divmod(int(flat), len(neg))
    if i in used_p or j in used_n:
        continue
    r = len(used_p); used_p.add(i); used_n.add(j)
    rank[q.image_id[pos[i]]] = r; rank[q.image_id[neg[j]]] = r
    if len(used_p) == len(pos):
        break
rest = [q.image_id[k] for k in neg if q.image_id[k] not in rank]
rank.update({iid: len(pos) + k for k, iid in enumerate(rest)})
man.assign(synth_rank=man.image_id.map(rank)).to_csv(ROOT / "artifacts/manifests/synthetic_sd_lora_qm.csv", index=False)

# counter: invert the quality cue
s = pd.Series(z.sum(1), index=q.image_id)
rk = pd.concat([s[q.image_id[pos]].rank(ascending=False, method="first"),
                s[q.image_id[neg]].rank(ascending=True, method="first")]) - 1
man.assign(synth_rank=man.image_id.map(rk).astype(int)).to_csv(ROOT / "artifacts/manifests/synthetic_sd_lora_qc.csv", index=False)

# check: how much does quality predict the label inside each selection? (0.5 = no information)
rows = []
for name, path in [("random", SRC), ("matched", "artifacts/manifests/synthetic_sd_lora_qm.csv"),
                   ("counter", "artifacts/manifests/synthetic_sd_lora_qc.csv")]:
    m = read_manifest(path).merge(q.drop(columns="label"), on="image_id")
    for n in (125, 250):
        g = m[m.synth_rank < n]
        rows.append({"selection": name, "per_class": n, "n": len(g),
                     **{f"auroc_low_{c}": roc_auc_score(g.label, -g[c]) for c in ["hf_energy", "std"]}})
print(pd.DataFrame(rows).round(3).to_string())
