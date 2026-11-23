"""Final controls for the synthetic-only TEACH result (docs/RESEARCH_PIVOT.md §13.x "Diagnosis", 2026-11-12).

CX_SYN_ONLY_TEACH (2,000 first-generation cx_sd images, teacher labels, real val) reached RSNA 0.92, but without a
distillation control it does not show that the images are useful. Same teacher as cx_manifests.py (CX_E2_ERASE seed 0:
EfficientNet-B0, DenseNet-121, ViT-S, mean probability).

    .venv/bin/python scripts/generator/teach_final.py
  real_teach_cx_2k.csv           tag RTEACH   2,000 real CheXpert train images the teacher never saw (subset_rank 1000..1999,
                                              the teacher trained on ranks < 500), label = teacher p
  synthetic_cx_sd_v2c25_teach.csv tag TEACHV2 the final generator's 2,000 images (v2, CFG 2.5), label = teacher p
Also writes reports/results/generator/teacher_scores_final.csv.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd
import torch

from cxr.data import load_set
from cxr.engine import RUNS_DIR, load_model, predict_logits, sigmoid

MAN = ROOT / "artifacts/manifests"
OUT = ROOT / "reports/results/generator"
TEACHERS = ["efficientnet_b0", "densenet121", "vit_small"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

syn = pd.read_csv(MAN / "synthetic_cx_sd_v2c25.csv")
syn = syn[syn.synth_rank < 1000].copy()
cx = pd.read_csv(MAN / "chexpert.csv", dtype={"patient_id": str})
real = cx[(cx.split == "train") & cx.subset_rank.between(1000, 1999)].copy()
assert real.label.value_counts().min() == 1000, real.label.value_counts()
real.to_csv(MAN / "_real_cx_2k.csv", index=False)
syn.to_csv(MAN / "_syn_v2c25_2k.csv", index=False)

sets = {"syn": load_set(str(MAN / "_syn_v2c25_2k.csv"), None, 224, "squash", device=None),
        "real": load_set(str(MAN / "_real_cx_2k.csv"), None, 224, "squash", device=None)}
scores = {k: pd.DataFrame({"image_id": s.df.image_id}) for k, s in sets.items()}
for arch in TEACHERS:
    model, _ = load_model(RUNS_DIR / "CX_E2_ERASE" / arch / "seed0", dev)
    for k, s in sets.items():
        scores[k][arch] = sigmoid(predict_logits(model, s, dev, batch_size=64, amp=False))
    del model
    torch.cuda.empty_cache()
for k in scores:
    scores[k]["p_teacher"] = scores[k][TEACHERS].mean(1)
pd.concat([scores["syn"].assign(set="synthetic_v2c25"), scores["real"].assign(set="real_unseen_2k")]).round(5).to_csv(
    OUT / "teacher_scores_final.csv", index=False)

p_syn = syn.image_id.map(scores["syn"].set_index("image_id").p_teacher)
p_real = real.image_id.map(scores["real"].set_index("image_id").p_teacher)
syn.assign(label=p_syn.round(5)).to_csv(MAN / "synthetic_cx_sd_v2c25_teach.csv", index=False)
rt = real.assign(label=p_real.round(5).to_numpy(), synth_rank=real.subset_rank - 1000, source="real:chexpert_teach2k")
rt[["label", "image_id", "image_path", "source", "split", "patient_id", "synth_rank"]].to_csv(MAN / "real_teach_cx_2k.csv", index=False)

for name, p, y in (("synthetic v2c25", p_syn, syn.label), ("real unseen 2k", p_real, real.label)):
    print(f"{name}: mean teacher p | normal {p[y == 0].mean():.3f}, pneumonia {p[y == 1].mean():.3f}")
