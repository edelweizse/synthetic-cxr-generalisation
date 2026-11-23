"""Generator evaluation: realism, class signal, memorisation of synthetic sets (docs/RESEARCH_PIVOT.md §11). CPU.

    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/generator/evaluate.py chexpert cx_sd cx_sd_v2 ...
    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/generator/evaluate.py kermany sd_lora
Each name is artifacts/manifests/synthetic_<name>.csv; the source's real held-out TEST split is scored too (the floor).
Reference = the source's real TRAIN split (what the generator was trained on).
  realism   KID x1000 (polynomial kernel, unbiased; ImageNet EfficientNet-B0 features) and precision / recall
            (k-NN manifolds, k = 5), both on 100 per class vs 100 per class real train, mean of 10 draws
  class     teacher = the source's E2_ERASE seed-0 models (EfficientNet-B0, DenseNet-121, ViT-S), mean p by prompt class,
            and the AUROC of the prompt label under the teacher
  memory    cosine similarity of each image to its nearest real train image; reference = real test -> train
            (different patients); n_above = images closer than the closest real test image ever gets
Output: reports/results/generator/eval_<source>.csv (or --out=<name>), nn_pairs_<name>.png (the 8 closest pairs).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from sklearn.metrics import roc_auc_score

from cxr.data import load_set
from cxr.engine import RUNS_DIR, load_model, predict_logits, sigmoid

torch.set_num_threads(6)
dev = torch.device("cpu")
MAN = ROOT / "artifacts/manifests"
OUT = ROOT / "reports/results/generator"
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = {"chexpert": ("chexpert.csv", "CX_E2_ERASE"), "kermany": ("kermany.csv", "E2_ERASE")}
TEACHERS = ["efficientnet_b0", "densenet121", "vit_small"]
N_PER_CLASS, DRAWS, K = 100, 10, 5

source, names = sys.argv[1], [n for n in sys.argv[2:] if not n.startswith("--out=")]
out_name = next((n[6:] for n in sys.argv[2:] if n.startswith("--out=")), f"eval_{source}")
real_csv, teacher_exp = SOURCES[source]
imnet = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0).eval()
MEAN = torch.tensor(imnet.pretrained_cfg["mean"]).view(1, 3, 1, 1)
STD = torch.tensor(imnet.pretrained_cfg["std"]).view(1, 3, 1, 1)


@torch.no_grad()
def features(ts):
    return torch.cat([imnet((x.expand(-1, 3, -1, -1) - MEAN) / STD) for x in ts.batches(64, dev)]).numpy()


def kid(a, b):
    d = a.shape[1]
    kxx, kyy, kxy = ((a @ a.T / d + 1) ** 3), ((b @ b.T / d + 1) ** 3), ((a @ b.T / d + 1) ** 3)
    m, n = len(a), len(b)
    return ((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (n * (n - 1)) - 2 * kxy.mean())


def prec_recall(real, fake):
    def radii(x):
        d = np.linalg.norm(x[:, None] - x[None], axis=2)
        return np.sort(d, 1)[:, K]
    rr, rf = radii(real), radii(fake)
    d = np.linalg.norm(fake[:, None] - real[None], axis=2)  # fake x real
    return (d <= rr[None]).any(1).mean(), (d.T <= rf[None]).any(1).mean()


def balanced_draw(y, rng):
    return np.concatenate([rng.choice(np.flatnonzero(y == c), N_PER_CLASS, replace=False) for c in (0, 1)])


trn = load_set(str(MAN / real_csv), {"split": "train"}, 224, "squash", device=None)
f_trn, y_trn = features(trn), trn.labels.numpy()
u_trn = f_trn / np.linalg.norm(f_trn, axis=1, keepdims=True)
teachers = [load_model(RUNS_DIR / teacher_exp / a / "seed0", dev)[0] for a in TEACHERS]

sets = {f"real test ({source})": load_set(str(MAN / real_csv), {"split": "test"}, 224, "squash", device=None)}
sets.update({n: load_set(str(MAN / f"synthetic_{n}.csv"), None, 224, "squash", device=None) for n in names})
rows, ref_max = [], None
for name, ts in sets.items():
    f, y = features(ts), ts.labels.numpy()
    rng = np.random.default_rng(0)
    k, pr = [], []
    for _ in range(DRAWS):
        i, j = balanced_draw(y, rng), balanced_draw(y_trn, rng)
        k.append(kid(f[i], f_trn[j]))
        pr.append(prec_recall(f_trn[j], f[i]))
    p = np.mean([sigmoid(predict_logits(m, ts, dev, batch_size=64, amp=False)) for m in teachers], axis=0)
    sim = (f / np.linalg.norm(f, axis=1, keepdims=True)) @ u_trn.T
    nn_sim, nn_idx = sim.max(1), sim.argmax(1)
    if ref_max is None:
        ref_max = nn_sim.max()
    rows.append(dict(set=name, n=len(y), kid_x1000=1000 * np.mean(k), kid_sd=1000 * np.std(k),
                     precision=np.mean([a for a, _ in pr]), recall=np.mean([b for _, b in pr]),
                     p_teacher_normal=p[y == 0].mean(), p_teacher_pneumonia=p[y == 1].mean(),
                     auroc_teacher=roc_auc_score(y, p), nn_sim_median=np.median(nn_sim),
                     nn_sim_p99=np.quantile(nn_sim, 0.99), nn_sim_max=nn_sim.max(),
                     n_above_real_max=int((nn_sim > ref_max).sum()), distinct_nn_share=len(np.unique(nn_idx)) / len(y)))
    print(pd.Series(rows[-1]).to_string(), flush=True)
    if not name.startswith("real"):
        top = np.argsort(-nn_sim)[:8]
        fig, ax = plt.subplots(2, 8, figsize=(16, 4.4))
        for c, t in enumerate(top):
            ax[0, c].imshow(ts.images[t:t + 1][0, 0], cmap="gray")
            ax[0, c].set_title(f"sim {nn_sim[t]:.3f}", fontsize=8)
            ax[1, c].imshow(trn.images[nn_idx[t]:nn_idx[t] + 1][0, 0], cmap="gray")
        for a in ax.ravel():
            a.axis("off")
        ax[0, 0].text(-20, 112, "synthetic", rotation=90, va="center")
        ax[1, 0].text(-20, 112, "nearest real", rotation=90, va="center")
        fig.tight_layout()
        fig.savefig(OUT / f"nn_pairs_{name}.png", dpi=80)
        plt.close(fig)
pd.DataFrame(rows).round(4).to_csv(OUT / f"{out_name}.csv", index=False)
print("generator eval done", flush=True)
