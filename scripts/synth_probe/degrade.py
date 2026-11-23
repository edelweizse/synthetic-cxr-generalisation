"""Causal test of the quality shortcut: degrade real NORMAL images (blur / lower contrast) and measure how much
each model's pneumonia probability rises. Same images, only quality changes."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch

from cxr.data import load_set
from cxr.engine import load_model, RUNS_DIR, sigmoid
from cxr.quality import contrast, gauss

torch.set_num_threads(4)
dev = torch.device("cpu")


DEGR = {"clean": lambda x: x, "blur1": lambda x: gauss(x, 1.0), "blur2": lambda x: gauss(x, 2.0),
        "contrast0.7": lambda x: contrast(x, 0.7), "blur1+contrast0.7": lambda x: contrast(gauss(x, 1.0), 0.7)}

sets = {}
a = load_set("artifacts/manifests/kermany.csv", {"split": "test"}, 224, "squash", device=None)
sets["A_test_normal"] = torch.from_numpy(np.ascontiguousarray(a.images[0:len(a)]))[torch.from_numpy(a.labels.numpy() == 0)]
b = load_set("artifacts/manifests/rsna.csv", {"split": "external"}, 224, "squash", device=None)
bl, bv = b.labels.numpy(), b.df["view"].to_numpy()
rng = np.random.default_rng(0)
for v in ("PA", "AP"):
    idx = np.sort(rng.choice(np.where((bl == 0) & (bv == v))[0], 300, replace=False))
    sets[f"B_{v}_normal"] = torch.from_numpy(np.ascontiguousarray(b.images[idx]))

rows = []
for arch in ["efficientnet_b0"]:
    for exp in ["E1", "R25", "E2", "E2_ERASE", "E3_sd_lora", "E4_sd_lora"]:
        for seed in (0, 1):
            rd = RUNS_DIR / exp / arch / f"seed{seed}"
            if not (rd / "best.pt").exists():
                continue
            model, _ = load_model(rd, dev)
            with torch.no_grad():
                for sname, X in sets.items():
                    X = X.float().div(255)
                    for dname, fn in DEGR.items():
                        p = np.concatenate([sigmoid(model(fn(X[i:i + 64]).contiguous(memory_format=torch.channels_last)).numpy())
                                            for i in range(0, len(X), 64)])
                        rows.append(dict(exp=exp, seed=seed, set=sname, degr=dname, mean_p=p.mean()))
            print("done", exp, seed, flush=True)
d = pd.DataFrame(rows)
d.to_csv(ROOT / "reports/results/synth_probe/degrade.csv", index=False)
t = d.groupby(["set", "exp", "degr"]).mean_p.mean().unstack()
t = t.sub(t["clean"], axis=0).drop(columns="clean").assign(clean_p=d[d.degr == "clean"].groupby(["set", "exp"]).mean_p.mean())
print(t.round(3).to_string())
