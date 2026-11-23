"""Training set for the CheXpert LoRA (docs/RESEARCH_PIVOT.md §9), in the layout train_lora.py reads.

    .venv/bin/python scripts/generator/prepare_chexpert.py
Every CheXpert TRAIN-split image of artifacts/manifests/chexpert.csv (val/test never reach the generator), squashed to
512 x 512 exactly like the classifier input (cxr.imaging.to_square), saved as RGB PNG. Caption = the class prompt the
Kermany samples were drawn with (scripts/synth_cf/cf_generate.PROMPT), so prompts and sampling settings stay identical
between study 1 and study 2; only the generator's training data changes.
Output: data/generator/chexpert/{NORMAL,PNEUMONIA}/<patient>.png + metadata.jsonl ({"file_name", "text"} per line).

    .venv/bin/python scripts/generator/prepare_chexpert.py rich   # + metadata_rich.jsonl (same images)
    .venv/bin/python scripts/generator/prepare_chexpert.py big    # data/generator/chexpert_big (fetch.py select-gen set)
    .venv/bin/python scripts/generator/prepare_chexpert.py big selection_obv.csv chexpert_obv   # obvious-positive set
    .venv/bin/python scripts/generator/prepare_chexpert.py view   # + metadata_view.jsonl (same images)
View captions (docs/RESEARCH_PIVOT.md §13, de-confounding): the class prompt + ", AP view" / ", PA view". In the training set the
view predicts the class (AP 65% of No Finding, 86% of opacity); with view in the prompt, sample.py --balance-view can draw both
classes 50/50 AP/PA so the synthetic set no longer carries that shortcut.
Big set (docs/RESEARCH_PIVOT.md §13, more data): the 14,769 + 14,769 images of data/chexpert/selection_gen.csv (CheXpert train
patients outside our val / test, several images per patient), same preprocessing and the same two class prompts.
Rich captions (generator v3, docs/RESEARCH_PIVOT.md §13): view, sex, age decade and every CheXbert finding = 1 of the image's
report (data/chexpert/selection.csv), e.g. "chest x-ray, AP view, female, age 60s, lung opacity, pleural effusion,
support devices". The class is no longer a fixed prompt; it is whatever findings the report lists.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from PIL import Image

from cxr.imaging import load_gray, to_square

OUT = ROOT / "data/generator/chexpert"
PROMPT = {0: "a frontal cxr_scan of normal lungs, clear lung fields, radiograph",
          1: "a frontal cxr_scan showing pneumonia, lung consolidation, radiograph"}  # = cf_generate.PROMPT


def one(args):
    src, dst = args
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp.png")
        Image.fromarray(to_square(load_gray(src), 512, "squash")).convert("RGB").save(tmp)
        tmp.rename(dst)


FINDINGS = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
            "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]


def rich_caption(r):
    found = [f.lower() for f in FINDINGS if r[f] == 1]
    if not r["label"]:
        found = ["no finding"] + [f for f in found if f == "support devices"]
    return f"chest x-ray, {r['view']} view, {r['sex'].lower()}, age {int(r['age']) // 10 * 10}s, " + ", ".join(found)


def rich():
    m = pd.read_csv(ROOT / "artifacts/manifests/chexpert.csv", dtype={"patient_id": str})
    m = m[m.split == "train"]
    sel = pd.read_csv(ROOT / "data/chexpert/selection.csv").drop(columns=["label", "age", "sex"])
    d = m.merge(sel, left_on="patient_id", right_on="deid_patient_id", how="left", validate="one_to_one")
    assert d[FINDINGS].notna().any(axis=1).all() or len(d) == len(m)
    with open(OUT / "metadata_rich.jsonl", "w") as f:
        for _, r in d.iterrows():
            rel = f"{'PNEUMONIA' if r['label'] else 'NORMAL'}/{r['patient_id']}.png"
            f.write(json.dumps({"file_name": rel, "text": rich_caption(r)}) + "\n")
    print("rich captions:", len(d), "e.g.", rich_caption(d.iloc[0]), "|", rich_caption(d[d.label == 1].iloc[0]))


def view():
    m = pd.read_csv(ROOT / "artifacts/manifests/chexpert.csv", dtype={"patient_id": str})
    m = m[m.split == "train"]
    with open(OUT / "metadata_view.jsonl", "w") as f:
        for y, p, v in zip(m.label, m.patient_id, m.view):
            f.write(json.dumps({"file_name": f"{'PNEUMONIA' if y else 'NORMAL'}/{p}.png", "text": f"{PROMPT[int(y)]}, {v} view"}) + "\n")
    print("view captions:", len(m), pd.crosstab(m.label, m.view).to_dict())


def big(selection="selection_gen.csv", name="chexpert_big"):
    sel = pd.read_csv(ROOT / "data/chexpert" / selection)
    src = [ROOT / "data/chexpert/images" / Path(p).with_suffix(".png").as_posix().removeprefix("train/") for p in sel.path_to_image]
    keep = [s.exists() for s in src]
    sel, src = sel[keep], [s for s, k in zip(src, keep) if k]
    rel = [f"{'PNEUMONIA' if y else 'NORMAL'}/{'_'.join(Path(p).with_suffix('').parts[-3:])}.png"
           for y, p in zip(sel.label, sel.path_to_image)]
    out = ROOT / "data/generator" / name
    with ProcessPoolExecutor(4) as ex:
        list(ex.map(one, [(s, out / r) for s, r in zip(src, rel)], chunksize=64))
    with open(out / "metadata.jsonl", "w") as f:
        for r, y in zip(rel, sel.label):
            f.write(json.dumps({"file_name": r, "text": PROMPT[int(y)]}) + "\n")
    print("big generator set:", len(rel), "images", sel.label.value_counts().to_dict(), f"({len(keep) - len(rel)} not on disk)")


def main():
    m = pd.read_csv(ROOT / "artifacts/manifests/chexpert.csv", dtype={"patient_id": str})
    m = m[m.split == "train"]
    rel = [f"{'PNEUMONIA' if y else 'NORMAL'}/{p}.png" for y, p in zip(m.label, m.patient_id)]
    with ProcessPoolExecutor(8) as ex:
        list(ex.map(one, [(ROOT / s, OUT / r) for s, r in zip(m.image_path, rel)], chunksize=64))
    with open(OUT / "metadata.jsonl", "w") as f:
        for r, y in zip(rel, m.label):
            f.write(json.dumps({"file_name": r, "text": PROMPT[int(y)]}) + "\n")
    print("generator training set:", len(rel), "images", m.label.value_counts().to_dict())


if __name__ == "__main__":
    {"rich": rich, "view": view, "big": big}.get(sys.argv[1] if sys.argv[1:] else "", main)(*sys.argv[2:])
