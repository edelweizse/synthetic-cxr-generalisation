"""Low-level image statistics per class: synthetic vs real Kermany vs RSNA (at the 224 px training resolution)."""
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

from cxr.data import load_set

torch.set_num_threads(4)
LAP = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)


def stats(tset, n=1500, seed=0):
    idx = np.sort(np.random.default_rng(seed).choice(len(tset), min(n, len(tset)), replace=False))
    x = torch.from_numpy(np.ascontiguousarray(tset.images[idx] if not hasattr(tset.images, "shape") else tset.images[idx]))
    x = x.float().div(255)
    if x.ndim == 3:
        x = x.unsqueeze(1)
    lap = F.conv2d(x, LAP)
    c = x[:, :, 32:-32, 32:-32]
    blur = F.avg_pool2d(x, 5, 1, 2)
    hf = (x - blur).abs().mean((1, 2, 3))
    df = tset.df.iloc[idx].reset_index(drop=True)
    return pd.DataFrame({"label": tset.labels.numpy()[idx], "lap_var": lap.var((1, 2, 3)).numpy() * 1e3,
                         "hf_energy": hf.numpy() * 1e2, "mean": c.mean((1, 2, 3)).numpy(), "std": c.std((1, 2, 3)).numpy(),
                         "view": df["view"].values if "view" in df else "-"})


rows = []
for name, man, where in [("synthetic", "artifacts/manifests/synthetic_sd_lora.csv", None),
                         ("kermany_train", "artifacts/manifests/kermany.csv", {"split": "train"}),
                         ("rsna", "artifacts/manifests/rsna.csv", {"split": "external"})]:
    s = stats(load_set(man, where, 224, "squash", device=None))
    s["set"] = name
    rows.append(s)
d = pd.concat(rows)
d.to_csv(ROOT / "reports/results/synth_probe/texture.csv", index=False)
print(d.groupby(["set", "view", "label"])[["lap_var", "hf_energy", "mean", "std"]].median().round(3).to_string())
