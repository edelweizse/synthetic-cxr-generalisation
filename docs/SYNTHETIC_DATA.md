# Synthetic data hand-off (generator → classifier)

This is the contract between the generative component (LoRA image generator) and the classifier experiments E3/E4.

## 1. What the generator may train on

Only Dataset A images with `split == "train"` in `artifacts/manifests/kermany.csv`.
**Never** use `val`, `test` or `excluded` rows, and never use any Dataset B (RSNA) image.

Two legitimate settings. Pick one and record it, because it changes how E3/E4 should be interpreted:

| Setting | Generator training images | What E3/E4 then measure |
|---|---|---|
| **strict** (recommended) | `split == "train" & subset_1k == True` (the same 1,000 images the classifier sees) | pure augmentation: can synthetic samples add value beyond the real images they were derived from? |
| **transfer** | all `split == "train"` rows (4.1k images) | the generator also carries information from ~3k real images the classifier never sees. Compare against the real-data controls R25/R50 |

```python
import pandas as pd
a = pd.read_csv("artifacts/manifests/kermany.csv")
strict = a[(a.split == "train") & a.subset_1k][["image_path", "label", "orig_label"]]
```

## 2. What to deliver

* At least **250 images per class** (E4 needs 250 normal + 250 pneumonia; E3 uses a 125/class subset of these).
  More is fine: extra images are simply not used.
* PNG, grayscale (RGB is converted), ideally ≥ 512 px on the short side. Full field of view, the same framing as real CXRs.
* Folder layout, either of:
  * `data/synthetic/<run_name>/normal/*.png` and `data/synthetic/<run_name>/pneumonia/*.png`
  * `data/synthetic/<run_name>/metadata.csv` with `file_name,label[,prompt,seed,...]`
* A short note: base model, LoRA rank/steps, prompts, sampler/guidance, setting (strict/transfer), and any filtering applied.

## 3. How they are used

```bash
python scripts/prepare_synthetic.py data/synthetic/<run_name> --name <run_name>
python scripts/run.py --exp E3 E4 --arch <arch> --seeds 0 1 2 \
       --synthetic artifacts/manifests/synthetic_<run_name>.csv --tag <run_name>
python scripts/aggregate.py
```

Synthetic images go through exactly the same decoding, resizing (224 px, full-FOV squash) and normalisation as real images.
Every run keeps the same step budget, early-stopping rule, hyper-parameters, validation set and threshold rule as E1, so the training data is the only difference.

`data/synthetic/placeholder/` is a pipeline test (within-class blends of real images), not generative data. Its runs are tagged `PLACEHOLDER` and excluded from all reported results.
