"""Experiment registry: what each experiment trains on, and how runs are configured.

Every experiment shares the validation set, evaluation sets, step budget, early-stopping rule and
the per-architecture hyper-parameters found by Optuna on E1 (A-validation only), so that the
*training data* is the only thing that changes between E1-E4 and the real-data controls.

Three different ways of changing what the model sees (kept apart in names and docs):
  augmentation   random transforms of the SAME real images, redrawn every step, training only
                 (E2 = conventional, E2_ERASE/ACQ/ROBUST/MIX, the ERASE_* prefix)
  synthetic data NEW images from the generative model added to the training set (E3, E4, the *_SYN* runs)
  preprocessing  one fixed transform of EVERY image, training and test (`resize_mode`; tag HARM)
"""

import json
from pathlib import Path

from . import PROJECT_ROOT
from .engine import DEFAULTS, RUNS_DIR

A = "artifacts/manifests/kermany.csv"
B = "artifacts/manifests/rsna.csv"
SYNTH = "artifacts/manifests/synthetic.csv"
# study 2 (docs/RESEARCH_PIVOT.md §9): the same experiments with CheXpert as the source; A is swapped for the source's
# manifest in train AND val. chexpert_vb has view-balanced training ranks (no AP/PA shortcut in any real(n) subset).
SOURCES = {"kermany": A, "chexpert": "artifacts/manifests/chexpert.csv", "chexpert_vb": "artifacts/manifests/chexpert_vb.csv"}
SOURCE_PREFIX = {"kermany": "", "chexpert": "CX_", "chexpert_vb": "CXVB_"}
HPO_DIR = PROJECT_ROOT / "configs" / "hpo"

VAL = dict(manifest=A, where={"split": "val"})
EVAL_SETS = {
    "A_test": dict(manifest=A, where={"split": "test"}, label_cols=["label"]),
    # primary external task: Lung Opacity vs Normal; secondary: Lung Opacity vs all other studies
    "B_external": dict(manifest=B, where={"split": "external"}, label_cols=["label", "label_any"]),
    # CheXpert internal test (study 2); for Kermany-trained runs a second external set
    "C_test": dict(manifest=SOURCES["chexpert"], where={"split": "test"}, label_cols=["label"]),
}

# shared budget / stopping rule for every run (HPO trials included)
BUDGET = dict(max_steps=1500, eval_every=50, early_stopping_patience=8, min_delta=1e-4)

BASE_PER_CLASS = 500  # E1 real subset: 500 normal + 500 pneumonia patients


def real(per_class: int) -> dict:
    """Class-balanced real-A subset (nested: smaller subsets are contained in larger ones)."""
    return dict(manifest=A, where={"split": "train", "subset_rank": {"lt": per_class}}, tag=f"real_{2 * per_class}")


def synthetic(per_class: int, manifest: str = SYNTH) -> dict:
    return dict(manifest=manifest, where={"synth_rank": {"lt": per_class}}, tag=f"synthetic_{2 * per_class}")


FULL_A = dict(manifest=A, where={"split": "train"}, tag="real_full")

EXPERIMENTS = {
    # core design
    "E1": dict(train=[real(BASE_PER_CLASS)], augment="none",
               doc="real A (1k balanced) - baseline"),
    "E2": dict(train=[real(BASE_PER_CLASS)], augment="conventional",
               doc="real A + conventional augmentation"),
    "E3": dict(train=[real(BASE_PER_CLASS), synthetic(BASE_PER_CLASS // 4)], augment="none",
               doc="real A + 25% synthetic A"),
    "E4": dict(train=[real(BASE_PER_CLASS), synthetic(BASE_PER_CLASS // 2)], augment="none",
               doc="real A + 50% synthetic A"),
    # matched real-data controls for E3/E4: is a synthetic image worth a real one?
    "R25": dict(train=[real(BASE_PER_CLASS * 5 // 4)], augment="none", doc="real A, 25% more real images"),
    "R50": dict(train=[real(BASE_PER_CLASS * 3 // 2)], augment="none", doc="real A, 50% more real images"),
    # real-data scaling curve
    "N250": dict(train=[real(125)], augment="none", doc="real A, 250 images"),
    "N500": dict(train=[real(250)], augment="none", doc="real A, 500 images"),
    "N2000": dict(train=[real(1000)], augment="none", doc="real A, 2000 images (max balanced)"),
    "FULL": dict(train=[FULL_A], augment="none", doc="all real A training images (imbalanced, ~72% pneumonia)"),
    "FULL_AUG": dict(train=[FULL_A], augment="conventional", doc="all real A + conventional augmentation"),
    # anti-overfitting variants, all built on E2 (same 1k images, same step budget)
    "E2_ACQ": dict(train=[real(BASE_PER_CLASS)], augment="acquisition",
                   doc="E2 + acquisition jitter (blur, noise, down/up-sampling)"),
    "E2_ERASE": dict(train=[real(BASE_PER_CLASS)], augment="erase",
                     doc="E2 + random erasing and border masking"),
    "E2_ROBUST": dict(train=[real(BASE_PER_CLASS)], augment="robust", doc="E2 + acquisition jitter + erasing"),
    "E2_MIX": dict(train=[real(BASE_PER_CLASS)], augment="conventional", overrides={"mixup_alpha": 0.4},
                   doc="E2 + mixup (alpha 0.4)"),
    "LP": dict(train=[real(BASE_PER_CLASS)], augment="conventional",
               overrides={"freeze_backbone": True, "lr": 5e-3, "head_lr_mult": 1.0, "weight_decay": 1e-4,
                          "schedule": "cosine", "warmup_steps": 50},
               doc="linear probe: frozen ImageNet backbone, head only, + conventional augmentation"),
    # E3/E4 and their real-data controls rebuilt on the strongest augmentation (E2_ERASE, selected on B-dev).
    # Name = <augmentation>_<what is added>: AUGMENTATION transforms the same images at every step,
    # SYN = extra synthetic images from the generator, REAL = extra real images.
    "ERASE_SYN250": dict(train=[real(BASE_PER_CLASS), synthetic(BASE_PER_CLASS // 4)], augment="erase",
                         doc="E2_ERASE augmentation + 250 synthetic images (E3 on the strong baseline)"),
    "ERASE_SYN500": dict(train=[real(BASE_PER_CLASS), synthetic(BASE_PER_CLASS // 2)], augment="erase",
                         doc="E2_ERASE augmentation + 500 synthetic images (E4 on the strong baseline)"),
    "ERASE_REAL250": dict(train=[real(BASE_PER_CLASS * 5 // 4)], augment="erase",
                          doc="E2_ERASE augmentation + 250 more real images (R25 on the strong baseline)"),
    "ERASE_REAL500": dict(train=[real(BASE_PER_CLASS * 3 // 2)], augment="erase",
                          doc="E2_ERASE augmentation + 500 more real images (R50 on the strong baseline)"),
    # dose-response: does a LOT more synthetic data help? (1,000 synthetic pneumonia images cap balanced sets at 2,000)
    "SYN1000": dict(train=[real(BASE_PER_CLASS), synthetic(BASE_PER_CLASS)], augment="none",
                    doc="real A (1k) + 1,000 synthetic images (100%)"),
    "SYN2000": dict(train=[real(BASE_PER_CLASS), synthetic(2 * BASE_PER_CLASS)], augment="none",
                    doc="real A (1k) + 2,000 synthetic images (200%)"),
    "SYN_ONLY": dict(train=[synthetic(2 * BASE_PER_CLASS)], augment="none",
                     doc="2,000 synthetic images only, no real training images"),
    # scaling (docs/RESEARCH_PIVOT.md §12): does more synthetic data close the gap to real? and the scarce-real use case
    "SYN_ONLY5K": dict(train=[synthetic(2500)], augment="none", doc="5,000 synthetic images only"),
    "SYN_ONLY10K": dict(train=[synthetic(5000)], augment="none", doc="10,000 synthetic images only"),
    "N250_SYN2000": dict(train=[real(125), synthetic(1000)], augment="none", doc="250 real + 2,000 synthetic images"),
}


def hpo_params(arch: str, variant: str | None = None) -> dict:
    """Optuna's best parameters; `variant` selects a search re-run under other preprocessing (configs/hpo/<arch>_<variant>.json)."""
    path = HPO_DIR / (f"{arch}_{variant}.json" if variant else f"{arch}.json")
    if variant and not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())["params"] if path.exists() else {}


def make_cfg(exp: str, arch: str, overrides: dict | None = None, synthetic_manifest: str | None = None,
             hpo_variant: str | None = None, source: str = "kermany") -> dict:
    """`source` swaps the real training/validation manifest; HPO parameters stay the Kermany ones (not re-tuned)."""
    spec = dict(EXPERIMENTS[exp])
    src = SOURCES[source]
    train = [{**s, "manifest": src} if s["manifest"] == A else dict(s) for s in spec["train"]]
    if synthetic_manifest:
        for s in train:
            if s["manifest"] == SYNTH:
                s["manifest"] = synthetic_manifest
    cfg = {**DEFAULTS, **BUDGET, **hpo_params(arch, hpo_variant), "arch": arch, "name": f"{exp}/{arch}",
           "experiment": exp, "train": train, "val": {**VAL, "manifest": src}, "source": source, "augment": spec["augment"], **spec.get("overrides", {})}
    return {**cfg, **(overrides or {})}


def run_dir(exp: str, arch: str, seed: int) -> Path:
    return RUNS_DIR / exp / arch / f"seed{seed}"
