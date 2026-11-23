"""Study 2 teacher-label manifests: the TEACH / RTEACH arms of docs/RESEARCH_PIVOT.md §8, repeated with CheXpert.

    .venv/bin/python scripts/generator/cx_manifests.py
Teacher = mean probability of CX_E2_ERASE seed 0 (EfficientNet-B0, DenseNet-121, ViT-S), as the Kermany teacher was E2_ERASE.
  synthetic_cx_sd_teach.csv  CX_TEACH   the CheXpert-LoRA samples (synthetic_cx_sd.csv, same ranks), label = teacher p
  real_teach_cx.csv          CX_RTEACH  the extra real CheXpert images of CX_R25/R50 (train subset_rank 500..999),
                                        label = teacher p: is CX_TEACH just distillation?
Also writes reports/results/synth_fix/teacher_scores_cx.csv.
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
OUT = ROOT / "reports/results/synth_fix"
OUT.mkdir(parents=True, exist_ok=True)
TEACHERS = ["efficientnet_b0", "densenet121", "vit_small"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

syn = pd.read_csv(MAN / "synthetic_cx_sd.csv")
cx = pd.read_csv(MAN / "chexpert.csv", dtype={"patient_id": str})
extra = cx[(cx.split == "train") & cx.subset_rank.between(500, 999)].copy()
extra.to_csv(MAN / "_extra_real_cx.csv", index=False)

sets = {"syn": load_set(str(MAN / "synthetic_cx_sd.csv"), None, 224, "squash", device=None),
        "real": load_set(str(MAN / "_extra_real_cx.csv"), None, 224, "squash", device=None)}
scores = {k: pd.DataFrame({"image_id": s.df.image_id}) for k, s in sets.items()}
for arch in TEACHERS:
    model, _ = load_model(RUNS_DIR / "CX_E2_ERASE" / arch / "seed0", dev)
    for k, s in sets.items():
        scores[k][arch] = sigmoid(predict_logits(model, s, dev, batch_size=64, amp=False))
    del model
    torch.cuda.empty_cache()
for k in scores:
    scores[k]["p_teacher"] = scores[k][TEACHERS].mean(1)
pd.concat([scores["syn"].assign(set="synthetic"), scores["real"].assign(set="extra_real")]).round(5).to_csv(
    OUT / "teacher_scores_cx.csv", index=False)

p_syn = syn.image_id.map(scores["syn"].set_index("image_id").p_teacher)
p_real = extra.image_id.map(scores["real"].set_index("image_id").p_teacher)
syn.assign(label=p_syn.round(5)).to_csv(MAN / "synthetic_cx_sd_teach.csv", index=False)
rt = extra.assign(label=p_real.round(5).to_numpy(), synth_rank=extra.subset_rank - 500, source="real:chexpert_teach")
rt[["label", "image_id", "image_path", "source", "split", "patient_id", "synth_rank"]].to_csv(MAN / "real_teach_cx.csv", index=False)

for name, p, y in (("synthetic", p_syn, syn.label), ("extra real", p_real, extra.label)):
    print(f"{name}: mean teacher p | normal {p[y == 0].mean():.3f}, pneumonia {p[y == 1].mean():.3f}")
