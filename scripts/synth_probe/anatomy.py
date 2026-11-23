"""Anatomy of synthetic vs real chest X-rays (docs/RESEARCH_PIVOT.md arm C): segment 14 structures with the torchxrayvision
ChestX-Det PSPNet and record, per structure, the mask area (fraction of the 512 px image) and centroid.

Scored sets (CPU, resumable; rows are appended to the CSV as batches finish):
  synthetic  sd_lora images in E3/E4 rank order, synth_rank < N_SYN per class (enough to refill 250 plausible ones per class)
  real       Kermany training images of the E1 subset (subset_rank < N_REAL per class): the reference for "plausible"
Output: reports/results/synth_probe/anatomy.csv. Plausibility thresholds and the filtered manifest: anatomy_select.py.
The segmenter is only used to score Kermany-like images, never RSNA, so its NIH-derived training data does not touch evaluation."""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv

from cxr.data import read_manifest, select
from cxr.experiments import A
from cxr.imaging import load_gray, to_square

torch.set_num_threads(int(os.environ.get("THREADS", 8)))
N_SYN, N_REAL, BS = 600, 300, 8
OUT = ROOT / "reports/results/synth_probe/anatomy.csv"


def prep(path):
    return to_square(load_gray(ROOT / path), 512, "squash")


syn = select(read_manifest("artifacts/manifests/synthetic_sd_lora.csv"), {"synth_rank": {"lt": N_SYN}}).assign(set="synthetic")
real = select(read_manifest(A), {"split": "train", "subset_rank": {"lt": N_REAL}}).assign(set="real",
                                                                                          synth_rank=lambda d: d.subset_rank)
todo = pd.concat([real, syn])[["image_id", "image_path", "label", "set", "synth_rank"]].sort_values(["synth_rank", "set"])
done = set(pd.read_csv(OUT).image_id) if OUT.exists() else set()
todo = todo[~todo.image_id.isin(done)]
print(f"{len(done)} done, {len(todo)} to score", flush=True)

model = xrv.baseline_models.chestx_det.PSPNet().eval()
names = [t.replace(" ", "_") for t in model.targets]
yy, xx = np.mgrid[0:512, 0:512] / 512.0
with Pool(4) as pool:
    for i in range(0, len(todo), BS):
        b = todo.iloc[i:i + BS]
        imgs = np.stack(pool.map(prep, b.image_path))
        x = torch.from_numpy(imgs).float().div(255).mul(2048).sub(1024).unsqueeze(1)
        with torch.no_grad():
            m = (torch.sigmoid(model(x)) > 0.5).numpy()  # [B, 14, 512, 512]
        area = m.mean((2, 3))
        cy = (m * yy).sum((2, 3)) / np.maximum(m.sum((2, 3)), 1)
        cx = (m * xx).sum((2, 3)) / np.maximum(m.sum((2, 3)), 1)
        rows = b[["image_id", "label", "set", "synth_rank"]].reset_index(drop=True)
        for k, n in enumerate(names):
            rows[f"area_{n}"], rows[f"cy_{n}"], rows[f"cx_{n}"] = area[:, k], cy[:, k], cx[:, k]
        rows.round(5).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        if i % (BS * 25) == 0:
            print(f"{i + len(b)}/{len(todo)}", flush=True)
print("anatomy done", flush=True)
