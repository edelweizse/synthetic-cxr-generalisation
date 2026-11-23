"""PLACEHOLDER synthetic data, used only to test the E3/E4 plumbing before the real generator output exists.

Each image is a within-class blend of two real images from the 1k training subset plus noise and blur.
This is not generative augmentation, and its results must not be reported as E3/E4 findings.

    python scripts/make_placeholder_synthetic.py && python scripts/prepare_synthetic.py data/synthetic/placeholder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageFilter

from cxr import PROJECT_ROOT
from cxr.data import read_manifest, select
from cxr.imaging import load_gray, to_square

OUT = PROJECT_ROOT / "data/synthetic/placeholder"
PER_CLASS = 250


def main():
    rng = np.random.default_rng(0)
    df = select(read_manifest("artifacts/manifests/kermany.csv"), {"split": "train", "subset_1k": True})
    for label, name in [(0, "normal"), (1, "pneumonia")]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
        paths = df[df.label == label].image_path.tolist()
        for k in range(PER_CLASS):
            a, b = rng.choice(len(paths), 2, replace=False)
            lam = rng.uniform(0.3, 0.7)
            x = lam * to_square(load_gray(PROJECT_ROOT / paths[a]), 512).astype(np.float32) \
                + (1 - lam) * to_square(load_gray(PROJECT_ROOT / paths[b]), 512).astype(np.float32)
            x = np.clip(x + rng.normal(0, 4, x.shape), 0, 255).astype(np.uint8)
            Image.fromarray(x).filter(ImageFilter.GaussianBlur(1.0)).save(OUT / name / f"ph_{k:04d}.png")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
