"""Does `harmonize` remove the quality shortcut? Shortcut AUROC of sharpness/contrast/noise statistics per mode."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.ndimage import laplace, uniform_filter, median_filter
from sklearn.metrics import roc_auc_score

from cxr.data import read_manifest, select, resolve
from cxr.imaging import load_gray, to_square

MODES = ["squash", "harmonize224", "harmonize160", "harmonize128", "harmonize96"]


def feats(x):
    x = x.astype(np.float32) / 255
    c = x[32:-32, 32:-32]
    hf = np.abs(x - uniform_filter(x, 5)).mean()
    noise = np.abs(x - median_filter(x, 3)).mean()
    return dict(lap_var=laplace(x).var() * 1e3, hf=hf * 1e2, noise=noise * 1e2, std=c.std(), mean=c.mean())


def job(args):
    path = args
    img = load_gray(resolve(path))
    return {m: feats(to_square(img, 224, m)) for m in MODES}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    parts = []
    for name, man, where in [("kermany", "artifacts/manifests/kermany.csv", {"split": "train"}),
                             ("synthetic", "artifacts/manifests/synthetic_sd_lora.csv", None),
                             ("rsna", "artifacts/manifests/rsna.csv", {"split": "external"})]:
        df = select(read_manifest(man), where)
        df = df[df.label.notna()]
        parts.append(df.iloc[np.sort(rng.choice(len(df), 800, replace=False))].assign(set=name))
    df = pd.concat(parts, ignore_index=True)
    with Pool(4) as pool:
        res = pool.map(job, df.image_path.tolist(), chunksize=8)
    rows = []
    for (_, r), f in zip(df.iterrows(), res):
        for m, d in f.items():
            rows.append({"set": r.set, "view": r.get("view") if isinstance(r.get("view"), str) else "-", "label": r.label, "mode": m, **d})
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "reports/results/synth_probe/harm_check.csv", index=False)
    out = []
    for (m, s, v), g in d.groupby(["mode", "set", "view"]):
        out.append({"mode": m, "set": s, "view": v, **{c: round(abs(roc_auc_score(g.label, -g[c]) - .5) + .5, 3)
                                                     for c in ["lap_var", "hf", "noise", "std", "mean"]}})
    print("max(AUROC, 1-AUROC) of each statistic for the label (0.5 = no shortcut)")
    print(pd.DataFrame(out).set_index(["mode", "set", "view"]).to_string())
