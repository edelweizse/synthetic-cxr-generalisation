"""Study 2 manifests (docs/RESEARCH_PIVOT.md §9): CheXpert Plus as the training source instead of Kermany.

    .venv/bin/python scripts/chexpert/make_manifest.py
Input: data/chexpert/selection.csv + images/ (scripts/chexpert/fetch.py). One frontal image per patient, so every split is
patient-level. Per class: 250 val, 750 test (internal CheXpert test), the rest train with a random `subset_rank`, as in
kermany.csv, so real(n) takes the same nested class-balanced subsets.
  chexpert.csv     natural view mix (pneumonia-type ~86% AP, normals ~65% AP: the view is itself a shortcut, as in RSNA)
  chexpert_vb.csv  same rows and splits; train `subset_rank` alternates AP / PA within each class, so every prefix up to
                   the scarcer view's count (~700 PA pneumonia) is view-balanced: the view shortcut is absent in training
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/chexpert"
MAN = ROOT / "artifacts/manifests"
N_VAL, N_TEST = 250, 750

sel = pd.read_csv(SRC / "selection.csv")
png = sel.path_to_image.map(lambda p: Path(p).with_suffix(".png").as_posix().removeprefix("train/"))
sel = sel.assign(image_path=["data/chexpert/images/" + p for p in png])
sel = sel[np.array([(ROOT / p).exists() for p in sel.image_path]) & sel.ap_pa.isin(["AP", "PA"])].reset_index(drop=True)

m = pd.DataFrame({"image_id": "C/" + sel.deid_patient_id, "image_path": sel.image_path, "label": sel.label,
                  "source": "chexpert", "patient_id": sel.deid_patient_id, "view": sel.ap_pa, "age": sel.age,
                  "sex": sel.sex})
rng = np.random.default_rng(0)
m["split"], m["subset_rank"] = "train", np.nan
for y, g in m.groupby("label"):
    idx = g.index.to_numpy()[rng.permutation(len(g))]
    m.loc[idx[:N_VAL], "split"] = "val"
    m.loc[idx[N_VAL:N_VAL + N_TEST], "split"] = "test"
    tr = idx[N_VAL + N_TEST:]
    m.loc[tr, "subset_rank"] = np.arange(len(tr))
m.to_csv(MAN / "chexpert.csv", index=False)

vb = m.copy()
tr = vb[vb.split == "train"]
for y, g in tr.groupby("label"):
    # k-th AP and k-th PA image (in random order) get ranks 2k and 2k+1; the leftover view comes after
    within = g.groupby("view").subset_rank.rank(method="first") - 1
    vb.loc[g.index, "subset_rank"] = (2 * within + (g.view == "PA")).rank(method="first") - 1
vb.to_csv(MAN / "chexpert_vb.csv", index=False)

print(m.groupby(["split", "label"]).size().unstack())
print("view share AP by split/label\n", m.assign(ap=m.view == "AP").groupby(["split", "label"]).ap.mean().unstack().round(2))
for name, d in (("natural", m), ("view-balanced", vb)):
    e1 = d[(d.split == "train") & (d.subset_rank < 500)]
    print(name, "E1 AP share by class", e1.assign(ap=e1.view == "AP").groupby("label").ap.mean().round(2).to_dict())
pa = tr[(tr.label == 1) & (tr.view == "PA")]
print("view-balanced prefix holds up to", 2 * len(pa), "pneumonia-type images")
