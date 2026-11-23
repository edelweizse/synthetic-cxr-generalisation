"""Manifests for the "can synthetic data be made to help?" arms (docs/RESEARCH_PIVOT.md §8).

  .venv/bin/python scripts/synth_fix/make_manifests.py
Teacher = mean probability of the robust real-trained E2_ERASE models (seed 0; EfficientNet-B0, DenseNet-121, ViT-S).
All manifests keep a per-class `synth_rank`, so E3 takes rank < 125 per class and E4 rank < 250, as for every other arm.
  synthetic_sd_lora_teach.csv     TEACH   the same sampled synthetic images as E3/E4, label = teacher probability (soft)
  synthetic_sd_lora_conf.csv      CONF    prompt labels; pneumonia ordered by teacher p (most convincing first), normals
                                          in their usual order minus the 51 the teacher calls pneumonia (p >= 0.5)
  synthetic_cf_lung_s10_src.csv   SRC_S10 counterfactual pairs labelled with the SOURCE image's real label: both edits of
  synthetic_cf_lung_s08_src.csv   SRC_S08 a pair share one label, so the generated lung texture is independent of the label
                                          (the "no-op edit" control)
  real_teach.csv                  RTEACH  the extra real images of R25/R50 (Kermany train, subset_rank 500..999), label =
                                          teacher probability: is TEACH just distillation?
Also writes reports/results/synth_fix/teacher_scores.csv."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch

from cxr.data import load_set
from cxr.engine import load_model, predict_logits, sigmoid, RUNS_DIR

MAN = ROOT / "artifacts/manifests"
OUT = ROOT / "reports/results/synth_fix"
OUT.mkdir(parents=True, exist_ok=True)
TEACHERS = ["efficientnet_b0", "densenet121", "vit_small"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

syn = pd.read_csv(MAN / "synthetic_sd_lora.csv")
ker = pd.read_csv(MAN / "kermany.csv")
extra = ker[(ker.split == "train") & ker.subset_rank.between(500, 999)].copy()
extra.to_csv(MAN / "_extra_real.csv", index=False)

sets = {"syn": load_set(str(MAN / "synthetic_sd_lora.csv"), None, 224, "squash", device=None),
        "real": load_set(str(MAN / "_extra_real.csv"), None, 224, "squash", device=None)}
scores = {k: pd.DataFrame({"image_id": s.df.image_id}) for k, s in sets.items()}
for arch in TEACHERS:
    model, _ = load_model(RUNS_DIR / "E2_ERASE" / arch / "seed0", dev)
    for k, s in sets.items():
        scores[k][arch] = sigmoid(predict_logits(model, s, dev, batch_size=64, amp=False))
    del model
    torch.cuda.empty_cache()
for k in scores:
    scores[k]["p_teacher"] = scores[k][TEACHERS].mean(1)
pd.concat([scores["syn"].assign(set="synthetic"), scores["real"].assign(set="extra_real")]).round(5).to_csv(
    OUT / "teacher_scores.csv", index=False)

p_syn = syn.image_id.map(scores["syn"].set_index("image_id").p_teacher)
p_real = extra.image_id.map(scores["real"].set_index("image_id").p_teacher)

# TEACH: same images and ranks as E3/E4, soft labels
syn.assign(label=p_syn.round(5)).to_csv(MAN / "synthetic_sd_lora_teach.csv", index=False)

# CONF: prompt labels, convincing images first
conf = syn.assign(p=p_syn)
pn = conf.label == 1
conf.loc[pn, "synth_rank"] = conf.loc[pn, "p"].rank(ascending=False, method="first").astype(int) - 1
keep = ~pn & (conf.p < 0.5)
conf.loc[keep, "synth_rank"] = conf.loc[keep, "synth_rank"].rank(method="first").astype(int) - 1
conf.loc[~pn & ~keep, "synth_rank"] = 10_000
conf.drop(columns="p").to_csv(MAN / "synthetic_sd_lora_conf.csv", index=False)

# SRC: counterfactual pairs labelled with the source image's label
for s in ("10", "08"):
    cf = pd.read_csv(MAN / f"synthetic_cf_lung_s{s}.csv")
    cf.assign(label=cf.source_label).to_csv(MAN / f"synthetic_cf_lung_s{s}_src.csv", index=False)

# RTEACH: extra real images with teacher labels, ranked like R25/R50
rt = extra.assign(label=p_real.round(5).to_numpy(), synth_rank=extra.subset_rank - 500, source="real:kermany_teach")
rt[["label", "image_id", "image_path", "source", "split", "patient_id", "synth_rank"]].to_csv(MAN / "real_teach.csv", index=False)

for name, p, y in (("synthetic", p_syn, syn.label), ("extra real", p_real, extra.label)):
    print(f"{name}: mean teacher p | normal {p[y == 0].mean():.3f}, pneumonia {p[y == 1].mean():.3f}")
c = conf[conf.synth_rank < 250]
print(f"CONF E4 subset: pneumonia p >= {c[c.label == 1].p.min():.3f}, normals p max {c[c.label == 0].p.max():.3f}")
