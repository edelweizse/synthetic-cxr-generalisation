"""Real-image mimics of the synthetic set (docs/RESEARCH_PIVOT.md §3): which property of the synthetic images does the damage?

Every manifest holds REAL Kermany training images of patients outside E1 (subset_rank >= 500, the pool R25/R50 draw from),
with a per-class `synth_rank` so they plug into the E3 (+250, rank < 125) / E4 (+500, rank < 250) slot like a synthetic
manifest. Each copies one property of the random sd_lora selection used by E3/E4:

  mimic_quality.csv  (a) shortcut strength: for each synthetic image (in rank order), the nearest unused real image of the
                     same class in (sharpness, contrast), each standardised within its own source. The real subset then has
                     the synthetic per-class quality profile, i.e. the amplified "soft / low-contrast = pneumonia" cue.
  mimic_noise.csv    (b) weak class signal: the R25/R50 images (same order), but 30% of the pneumonia slots hold a NORMAL
                     image labelled pneumonia, chosen quality-nearest to the pneumonia image it replaces, so the quality cue
                     is unchanged. A perfect scorer separates the added set at AUROC 0.85 (synthetic: 0.75-0.91, §2).
  mimic_both.csv     (a) + (b): mimic_quality with the same 30% of pneumonia slots replaced by quality-matched normals.

    python scripts/synth_probe/mimic_select.py          # 30% mislabelled: mimic_{quality,noise,both}.csv
    python scripts/synth_probe/mimic_select.py 0.5      # 50%: mimic_noise50.csv, mimic_both50.csv (quality is unchanged)
Stronger doses: a robust real-trained classifier gives synthetic pneumonia a mean p of only 0.28-0.54 (§2), so far more than
30% of it may show no pathology.

Writes reports/results/synth_probe/mimic_check[<pct>].csv: label predictiveness of low sharpness / contrast inside each subset
(compare with the synthetic random selection) and the share of mislabelled images."""
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

from cxr.data import load_set, read_manifest
from cxr.experiments import A, BASE_PER_CLASS

torch.set_num_threads(4)
SYN = "artifacts/manifests/synthetic_sd_lora.csv"
N_MAX = 250  # E4 size per class
FLIP = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3  # share of pneumonia slots holding a mislabelled normal
SUFFIX = "" if FLIP == 0.3 else str(round(FLIP * 100))
OUT = ROOT / "artifacts/manifests"


def quality(manifest, where):
    ts = load_set(manifest, where, 224, "squash", device=None)
    x = torch.from_numpy(np.ascontiguousarray(ts.images[:])).float().div(255)
    return ts.df.assign(hf_energy=(x - F.avg_pool2d(x, 5, 1, 2)).abs().mean((1, 2, 3)).numpy() * 1e2,
                        std=x[:, :, 32:-32, 32:-32].std((1, 2, 3)).numpy())


def zscore(df):
    return np.column_stack([(df[c] - df[c].mean()) / df[c].std() for c in ["hf_energy", "std"]])


def flipped(r):  # evenly spread: exactly round(FLIP * n) flipped slots in every prefix of length n
    return np.floor(FLIP * (r + 1)) > np.floor(FLIP * r)


def nearest(target_z, pool_z, used):
    d = ((pool_z - target_z) ** 2).sum(1)
    d[list(used)] = np.inf
    j = int(np.argmin(d)); used.add(j)
    return j


syn = quality(SYN, {"synth_rank": {"lt": N_MAX}})
real = quality(A, {"split": "train", "subset_rank": {"ge": BASE_PER_CLASS}})
syn_z, real_z = zscore(syn), zscore(real)
cls = {c: np.where(real.label == c)[0] for c in (0, 1)}
cls_z = {c: real_z[cls[c]] for c in (0, 1)}

rows = {"quality": [], "noise": [], "both": []}
used = {"quality": {0: set(), 1: set()}, "noise_normals": set(), "both_normals": set()}
# reserve R50's normals so (b)'s mislabelled normals are images R25/R50 never use
r50_normals = {k for k, i in enumerate(cls[0]) if real.subset_rank.iloc[i] < BASE_PER_CLASS + N_MAX}
used["noise_normals"] |= r50_normals
for c in (0, 1):
    s = syn[syn.label == c].sort_values("synth_rank")
    rr = real.iloc[cls[c]].sort_values("subset_rank")
    for r, (si, ri) in enumerate(zip(s.index, rr.index[:N_MAX])):
        flip = c == 1 and flipped(r)
        # (a): real image with the synthetic image's quality
        j = nearest(syn_z[syn.index.get_loc(si)], cls_z[c], used["quality"][c])
        rows["quality"].append((cls[c][j], c, r))
        # (a)+(b): mislabelled slots take a normal with that synthetic pneumonia image's quality
        if flip:
            j2 = nearest(syn_z[syn.index.get_loc(si)], cls_z[0], used["both_normals"] | used["quality"][0])
            used["both_normals"].add(j2); rows["both"].append((cls[0][j2], 1, r))
        else:
            rows["both"].append((cls[c][j], c, r))
        # (b): R25/R50 order; mislabelled slots take a normal with the replaced pneumonia image's quality
        k = real.index.get_loc(ri)
        if flip:
            j3 = nearest(real_z[k], cls_z[0], used["noise_normals"])
            rows["noise"].append((cls[0][j3], 1, r))
        else:
            rows["noise"].append((k, c, r))
# both's normal class must not reuse its mislabelled normals
assert not ({cls[0][j] for j in used["both_normals"]} & {i for i, c, _ in rows["both"] if c == 0})

man = read_manifest(A).set_index("image_id")
check = []
for name, sel in rows.items():
    if SUFFIX and name == "quality":  # does not depend on FLIP
        continue
    name += SUFFIX
    idx, lab, rank = map(np.array, zip(*sel))
    ids = real.image_id.to_numpy()[idx]
    assert len(set(ids)) == len(ids), name
    m = man.loc[ids].reset_index().assign(label=lab, synth_rank=rank, source=f"kermany:mimic_{name}")
    m.to_csv(OUT / f"mimic_{name}.csv", index=False)
    q = real.iloc[idx].assign(label=lab, synth_rank=rank, true=real.label.to_numpy()[idx])
    for n in (125, 250):
        g = q[q.synth_rank < n]
        check.append({"set": f"mimic_{name}", "per_class": n, "auroc_low_hf_energy": roc_auc_score(g.label, -g["hf_energy"]),
                      "auroc_low_std": roc_auc_score(g.label, -g["std"]), "mislabelled": (g.label != g.true).mean()})
for n in (125, 250):
    g = syn[syn.synth_rank < n]
    check.append({"set": "synthetic random", "per_class": n, "auroc_low_hf_energy": roc_auc_score(g.label, -g["hf_energy"]),
                  "auroc_low_std": roc_auc_score(g.label, -g["std"]), "mislabelled": np.nan})
    g = real[real.subset_rank < BASE_PER_CLASS + n]
    check.append({"set": "R25/R50 real", "per_class": n, "auroc_low_hf_energy": roc_auc_score(g.label, -g["hf_energy"]),
                  "auroc_low_std": roc_auc_score(g.label, -g["std"]), "mislabelled": 0.0})
check = pd.DataFrame(check).round(3)
check.to_csv(ROOT / f"reports/results/synth_probe/mimic_check{SUFFIX}.csv", index=False)
print(check.to_string())
