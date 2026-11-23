"""Register a folder of synthetic chest X-rays as a training manifest for E3/E4.

Accepted layouts (PNG/JPEG, any size; grayscale or RGB):
  <dir>/normal/*.png  and  <dir>/pneumonia/*.png        (folder names case-insensitive)
  <dir>/metadata.csv with columns: file_name,label   (label 0/1 or normal/pneumonia; extra columns kept)

    python scripts/prepare_synthetic.py data/synthetic/lora_v1 --name lora_v1
    -> artifacts/manifests/synthetic_lora_v1.csv
    python scripts/run.py --exp E3 E4 --arch <arch> --synthetic artifacts/manifests/synthetic_lora_v1.csv --tag lora_v1

Each class gets a random `synth_rank` (seed 0), so E3 uses rank < 125 and E4 rank < 250 per class
(25% / 50% of the 1,000-image real subset), and E3's images are a subset of E4's.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from PIL import Image

from cxr import PROJECT_ROOT

EXTS = {".png", ".jpg", ".jpeg"}
LABELS = {"normal": 0, "pneumonia": 1, "0": 0, "1": 1}


def scan(root: Path) -> pd.DataFrame:
    meta = root / "metadata.csv"
    if meta.exists():
        df = pd.read_csv(meta)
        df["path"] = [root / f for f in df.pop("file_name")]
        df["label"] = df["label"].astype(str).str.lower().map(LABELS)
        return df
    rows = []
    for sub in root.iterdir():
        if sub.is_dir() and sub.name.lower() in ("normal", "pneumonia"):
            rows += [{"path": p, "label": LABELS[sub.name.lower()]} for p in sorted(sub.rglob("*")) if p.suffix.lower() in EXTS]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.folder).resolve()
    name = args.name or root.name
    df = scan(root)
    if df.empty or df["label"].isna().any():
        sys.exit(f"no usable images / unknown labels under {root}")
    df["image_id"] = [f"S/{name}/{p.parent.name}/{p.stem}" for p in df.path]
    df["image_path"] = [str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p) for p in df.path]
    df["source"], df["split"] = f"synthetic:{name}", "train"
    df["patient_id"] = df["image_id"]
    rng = np.random.default_rng(args.seed)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    df["synth_rank"] = df.groupby("label").cumcount()
    sizes = [Image.open(p).size for p in df.path.iloc[:50]]
    out = PROJECT_ROOT / "artifacts/manifests" / f"synthetic_{name}.csv"
    df.drop(columns="path").to_csv(out, index=False)
    counts = df.label.value_counts().to_dict()
    print(f"{len(df)} images {counts} -> {out}\nexample sizes: {sorted(set(sizes))[:5]}")
    for need, exp in [(125, "E3"), (250, "E4")]:
        ok = all(counts.get(c, 0) >= need for c in (0, 1))
        print(f"  {exp}: needs {need}/class -> {'OK' if ok else 'NOT ENOUGH'}")


if __name__ == "__main__":
    main()
