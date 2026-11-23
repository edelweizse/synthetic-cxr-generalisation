from common import SETUP, Builder, out_path

b = Builder()
b.md(r"""
# 06 · Overfitting: to the images, or to the hospital?

"Overfitting" means two different things here, and they need different remedies:

| | Overfitting to the **training images** | Overfitting to the **source dataset** |
|---|---|---|
| symptom | training loss ≪ validation loss | validation loss ≪ Kermany-test / RSNA loss |
| visible on Kermany validation? | yes | **no**: validation shares the source's quirks |
| controlled by | early stopping, weight decay, dropout | variation the model cannot exploit: augmentation, multi-source data |
| status after E1 | contained (early stopping, val AUROC 0.99) | **severe**: loss 10–17× higher out of distribution |

This notebook first measures both kinds for every configuration, then tests five interventions built on E2. Each targets a specific mechanism identified in notebooks 01 and 05:

| Variant | What is added to E2 | Hypothesis it tests |
|---|---|---|
| **E2_ACQ** | random blur (σ 0.3–1.5 px), sensor-like noise, down/up-sampling to 35–70% resolution | Kermany's classes differ in resolution and JPEG quality. Randomising sharpness and noise should weaken that shortcut and improve RSNA |
| **E2_ERASE** | random erasing of a 2–12% rectangle + random masking of a 3–12% border frame | models attend to the image periphery (markers, collimation, shoulders). Occluding it should push evidence into the lungs |
| **E2_ROBUST** | ACQ + ERASE | are the two complementary? |
| **E2_MIX** | mixup (α = 0.4): train on convex combinations of image pairs and their labels | out-of-distribution failure is *over-confident*. Mixup is known to regularise confidence and improve calibration |
| **LP** | frozen ImageNet backbone, only the linear head is trained (+ E2 augmentation) | if most of the damage is the network *re-learning* Kermany-specific features, a frozen generic representation should transfer better, at the price of in-domain fit |

Rules are unchanged: the same 1,000 images, step budget and early stopping, Kermany validation for checkpoint and threshold, three seeds. LP uses its own fixed head-only optimiser settings (lr 5·10⁻³, cosine), since a single linear layer trains very differently from a full network. Architectures: all seven from notebook 03.

External numbers: RSNA is split by patient into **RSNA-dev** (30%, used only to choose between configurations) and **RSNA-test** (70%,
reporting). The ladder and calibration use all labelled RSNA images; comparisons between variants are reported on RSNA-test.
""")
b.code(SETUP + r'''
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, log_loss
from cxr.analysis import collect_runs, mean_sd, compare, strata_runs, ece, EVAL_LABELS, STRATA_LABELS, ROLE_LABELS
from cxr.engine import load_model, predict_logits, get_device, sigmoid, RUNS_DIR
from cxr.data import load_set, read_manifest, ensure_cache
from cxr.explain import gradcam

EXPS = ["E1", "E2", "E2_ACQ", "E2_ERASE", "E2_ROBUST", "E2_MIX", "LP"]
ARCHS = ["efficientnet_b0", "densenet121", "vit_small", "resnet18", "resnet50", "convnext_tiny", "swin_tiny"]
A = "artifacts/manifests/kermany.csv"
ALL_LABELS = {**EVAL_LABELS, **STRATA_LABELS, **ROLE_LABELS}
DELTA_KEYS = ["A_test:label", "B_external:label@b_role=test", "B_external:label@view=PA", "B_external:label@view=AP"]
runs = collect_runs()
runs = runs[runs.experiment.isin(EXPS) & runs.arch.isin(ARCHS)]
print(runs.drop_duplicates(["experiment", "arch", "seed"]).groupby(["arch", "experiment"]).size().unstack())
device = get_device()
''')

b.md(r"""
## 1. The loss ladder: train → validation → Kermany test → RSNA

For every run: BCE and AUROC on its own 1,000 training images (no augmentation), Kermany validation, Kermany test and RSNA (opacity vs normal), all at the selected checkpoint. A healthy model has a short ladder. Large jumps between validation and the test sets are dataset-level overfitting that validation cannot see.
""")
b.code(r'''
cache = RES / "overfitting_ladder.csv"
done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["experiment", "arch", "seed"])
have = set(map(tuple, done[["experiment", "arch", "seed"]].astype(str).values))
train_set = load_set(A, {"split": "train", "subset_rank": {"lt": 500}}, 224, "squash", device=None)
rows = []
for (exp, arch, seed), _ in runs.groupby(["experiment", "arch", "seed"]):
    if (exp, arch, str(seed)) in have:
        continue
    rd = RUNS_DIR / exp / arch / f"seed{seed}"
    model, _ = load_model(rd, device)
    y = train_set.labels.numpy(); p = sigmoid(predict_logits(model, train_set, device, 256))
    r = {"experiment": exp, "arch": arch, "seed": seed, "train_bce": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)), "train_auroc": roc_auc_score(y, p)}
    for name, f, col in [("val", "preds_val.csv", "label"), ("testA", "preds_A_test.csv", "label"), ("B", "preds_B_external.csv", "label")]:
        d = pd.read_csv(rd / f); d = d[d[col].notna()]
        r[f"{name}_bce"] = log_loss(d[col], np.clip(d.prob, 1e-7, 1 - 1e-7)); r[f"{name}_auroc"] = roc_auc_score(d[col], d.prob)
    rows.append(r)
    pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(cache, index=False)  # resumable: saved after every run
    del model; torch.cuda.empty_cache()
ladder = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
ladder["seed"] = ladder.seed.astype(int)
ladder.to_csv(cache, index=False)
lad = ladder.groupby(["arch", "experiment"])[[c for c in ladder if c.endswith(("_bce", "_auroc"))]].mean()
lad["RSNA/val loss ratio"] = lad.B_bce / lad.val_bce
display(lad.reindex(pd.MultiIndex.from_product([ARCHS, EXPS])).dropna(how="all").round(3))
''')
b.code(r'''
ncol = 4
fig, axes = plt.subplots(-(-len(ARCHS) // ncol), ncol, figsize=(5 * ncol, 4 * -(-len(ARCHS) // ncol)), sharey=True, squeeze=False)
axes = axes.ravel()
for ax in axes[len(ARCHS):]:
    ax.axis("off")
axes = axes[:len(ARCHS)]
stages = ["train_bce", "val_bce", "testA_bce", "B_bce"]
for ax, arch in zip(axes, ARCHS):
    for exp in EXPS:
        if (arch, exp) in lad.index:
            ax.plot(range(4), lad.loc[(arch, exp), stages], marker="o", label=exp)
    ax.set_xticks(range(4), ["train", "Kermany val", "Kermany test", "RSNA"]); ax.set_yscale("log"); ax.set_title(arch)
axes[0].set_ylabel("BCE (log scale)"); axes[-1].legend(fontsize=8)
fig.suptitle("Loss ladder: how fast does confidence break down away from the training images?")
plt.tight_layout(); plt.savefig(FIG / "overfitting_ladder.png", bbox_inches="tight"); plt.show()
''')
b.md(r"""
**How to read it.** The step from *train* to *Kermany val* is classic overfitting to the training images. The steps from *Kermany val* to *Kermany test* and *RSNA* are overfitting to the dataset. Note the log scale: one grid line is a factor of 10.
""")

b.md("## 2. Discrimination: AUROC on every test view")
b.code(r'''
strata = strata_runs(runs, {**STRATA_LABELS, **ROLE_LABELS})
tabs = []
for key, lab in EVAL_LABELS.items():
    tabs.append(mean_sd(runs[runs["eval"] == key], ["arch", "experiment"], ["auroc"]).drop(columns="seeds").rename(columns={"auroc": lab}))
for key, lab in {**STRATA_LABELS, **ROLE_LABELS}.items():
    tabs.append(mean_sd(strata[strata["eval"] == key], ["arch", "experiment"], ["auroc"]).drop(columns="seeds").rename(columns={"auroc": lab}))
auc_table = pd.concat(tabs, axis=1).reindex(pd.MultiIndex.from_product([ARCHS, EXPS])).dropna(how="all")
auc_table.to_csv(RES / "overfitting_variants_auroc.csv")
display(auc_table)
''')
b.code(r'''
rows = [compare("E2", e, arch, key, n_boot=500) for arch in ARCHS for e in EXPS if e not in ("E2",) for key in DELTA_KEYS]
deltas = pd.DataFrame([r for r in rows if r])
deltas.to_csv(RES / "overfitting_variants_vs_e2.csv", index=False)
fig, axes = plt.subplots(len(ARCHS), len(DELTA_KEYS), figsize=(20, 3.3 * len(ARCHS)), squeeze=False, sharex="col")
order = [e for e in EXPS if e != "E2"]
for i, arch in enumerate(ARCHS):
    for j, key in enumerate(DELTA_KEYS):
        x = deltas[(deltas.arch == arch) & (deltas["eval"] == key)].set_index("to").reindex(order).dropna(subset=["delta_ensemble"])
        for yy, (dlt, lo, hi, ps) in enumerate(x[["delta_ensemble", "ci_lo", "ci_hi", "delta_per_seed_mean"]].to_numpy()):
            c = "tab:green" if lo > 0 and ps > 0.01 else "tab:red" if hi < 0 and ps < -0.01 else "tab:gray"
            axes[i, j].errorbar([dlt], [yy], xerr=[[dlt - lo], [hi - dlt]], fmt="o", color=c, capsize=3)
            axes[i, j].plot([ps], [yy], "x", color="k", ms=5)
        axes[i, j].axvline(0, color="k", lw=0.8); axes[i, j].set_yticks(range(len(x)), x.index)
        axes[i, j].set_title(f"{arch} · {ALL_LABELS[key]}", fontsize=8)
for ax in axes[-1]: ax.set_xlabel("Δ AUROC vs E2 (dot: seed-ensemble ± 95% CI; ×: mean per-seed Δ)")
plt.tight_layout(); plt.savefig(FIG / "overfitting_variants_delta.png", bbox_inches="tight"); plt.show()
''')
b.md(r"""
Colour code: green or red only when the bootstrap CI excludes 0 **and** the mean per-seed difference exceeds 0.01 in the same direction. With 12k RSNA-test images, the CI alone is almost always "significant", so seed variability is the binding uncertainty (see notebook 04).
""")

b.md("## 3. Calibration on RSNA")
b.code(r'''
rows = []
for (exp, arch, seed), _ in runs.groupby(["experiment", "arch", "seed"]):
    d = pd.read_csv(RUNS_DIR / exp / arch / f"seed{seed}" / "preds_B_external.csv", usecols=["label", "prob"]).dropna()
    rows.append({"experiment": exp, "arch": arch, "seed": seed, "ece_RSNA": ece(d.label, d.prob), "mean_score_RSNA": d.prob.mean()})
cal = pd.DataFrame(rows)
display(mean_sd(cal, ["arch", "experiment"], ["ece_RSNA", "mean_score_RSNA"]).reindex(pd.MultiIndex.from_product([ARCHS, EXPS])).dropna())
print("RSNA prevalence (opacity vs normal):", round(pd.read_csv(RUNS_DIR / "E1" / ARCHS[0] / "seed0" / "preds_B_external.csv").label.mean(), 3))
''')

b.md(r"""
## 4. Mechanism checks
### 4a. Does acquisition jitter weaken the image-size association? (target of E2_ACQ)
Spearman ρ between score and original image width among *normal* Kermany-test images (cf. notebook 05). The closer to 0, the less the score tracks the acquisition property.
""")
b.code(r'''
a_man = read_manifest(A).set_index("image_id")
rows = []
for (exp, arch, seed), _ in runs.groupby(["experiment", "arch", "seed"]):
    d = pd.read_csv(RUNS_DIR / exp / arch / f"seed{seed}" / "preds_A_test.csv")
    d = d[d.label == 0]
    rows.append({"experiment": exp, "arch": arch, "seed": seed,
                 "rho_width_normals": spearmanr(d.prob, d.width).statistic,
                 "rho_filesize_normals": spearmanr(d.prob, d.file_bytes).statistic})
probe = pd.DataFrame(rows)
display(mean_sd(probe, ["arch", "experiment"], ["rho_width_normals", "rho_filesize_normals"]).reindex(pd.MultiIndex.from_product([ARCHS, EXPS])).dropna())
''')
b.md(r"""
### 4b. Does erasing move attribution off the periphery? (target of E2_ERASE)
The share of Grad-CAM mass in the outer 15% frame (uniform = 0.51), from 200 random RSNA images and 200 random Kermany-test images per configuration (seed 0).
""")
b.code(r'''
def border_share(rd, manifest, where, n=200):
    ts = load_set(manifest, where, 224, "squash", device=None, require_label=False)
    idx = np.random.default_rng(0).choice(len(ts), min(n, len(ts)), replace=False)
    imgs = torch.from_numpy(np.stack([ts.images[int(i):int(i) + 1][0] for i in sorted(idx)])).float().div(255)
    model, _ = load_model(rd, device); out = []
    for k in range(0, len(imgs), 10):
        cam = gradcam(model, imgs[k:k + 10].to(device))[0]; m = int(224 * 0.15); torch.cuda.empty_cache()
        out.extend(1 - cam[:, m:-m, m:-m].sum((1, 2)) / (cam.sum((1, 2)) + 1e-8))
    del model; torch.cuda.empty_cache()
    return float(np.mean(out))
rows = []
for arch in ARCHS:
    for exp in EXPS:
        rd = RUNS_DIR / exp / arch / "seed0"
        if (rd / "best.pt").exists():
            rows.append({"arch": arch, "experiment": exp,
                         "border_share_Kermany": border_share(rd, A, {"split": "test"}),
                         "border_share_RSNA": border_share(rd, "artifacts/manifests/rsna.csv", {"split": "external"})})
border = pd.DataFrame(rows).set_index(["arch", "experiment"])
border.to_csv(RES / "overfitting_border_share.csv")
display(border.round(3))
''')
b.md("### 4c. Seed stability")
b.code(r'''
stab = runs[runs["eval"].isin(["A_test:label", "B_external:label"])].groupby(["arch", "experiment", "eval"]).auroc.std().unstack()
stab.columns = [f"SD across seeds · {EVAL_LABELS[c]}" for c in stab.columns]
display(stab.reindex(pd.MultiIndex.from_product([ARCHS, EXPS])).dropna().round(3))
''')
b.md(r"""
## Insights

Seven architectures × seven configurations × three seeds. External numbers are **RSNA-test**; Δ is the seed-ensemble difference vs E2.

**1. The overfitting that matters is to the dataset, not to the images.** Every configuration fits its 1,000 training images almost perfectly (train BCE 0.002–0.13, train AUROC ≥0.995), and Kermany-validation BCE stays at 0.04–0.15: early stopping contains classic overfitting. The damage happens out of distribution. For E1 the loss is 0.40–0.75 on the Kermany test set and 0.58–1.36 on RSNA, **9–17× the validation loss** (ResNet-18 and Swin-T at the low end, the other five at 16.7–17.5), a gap that Kermany validation cannot see. E2 already cuts the RSNA loss by roughly a quarter to a half.

**2. E2_ERASE is the best variant on RSNA-dev for six of seven architectures** (RSNA-test 0.905–0.926), and never worse than E2 (Δ −0.000 to +0.033). The exception is **Swin-T, where acquisition jitter (E2_ACQ) is best** (0.911 vs 0.905). Gains over E2 by the ±0.02 rule: ViT-S +0.033, ConvNeXt-T +0.023, Swin-T +0.019 (both transformers and the modern CNN); ResNet-50 +0.016, DenseNet-121 +0.011, ResNet-18 +0.003, EfficientNet-B0 ±0. Part of the transformer gain is stability: their E2 runs are the least stable (seed SD on RSNA: ViT-S 0.062, Swin-T 0.029), erasing brings it to ≤0.008.

**3. Erasing does move attention inward, for every architecture except EfficientNet-B0.** Share of Grad-CAM mass in the outer 15% frame on RSNA, E2 → E2_ERASE: ConvNeXt-T 0.44 → 0.32, DenseNet-121 0.59 → 0.40, ResNet-50 0.46 → 0.39, ResNet-18 0.41 → 0.37, ViT-S 0.51 → 0.46, Swin-T 0.43 → 0.40, but EfficientNet-B0 0.46 → 0.59 (E2_ROBUST 0.76). So the hypothesised mechanism holds for 6/7 models, and EfficientNet-B0, the one model whose border share rises, is also the only one with no AUROC gain from erasing. (One seed, 200 images per set; Grad-CAM is coarse.)

**4. Acquisition jitter (E2_ACQ) helps modern architectures and hurts classic CNNs.** Δ vs E2: Swin-T +0.030, ConvNeXt-T +0.015, ResNet-50 +0.008, ViT-S −0.002, ResNet-18 −0.007, EfficientNet-B0 −0.010, DenseNet-121 −0.021. It never does what it was designed for: the association between score and image width among healthy Kermany-test images is unchanged (ρ −0.49 to −0.57, vs −0.49 to −0.61 for E2). Combining it with erasing (E2_ROBUST) is never better than the better of the two alone.

**5. Mixup fixes confidence, not ranking.** E2_MIX has the best RSNA calibration of all variants **for all seven architectures** (ECE 0.06–0.10 vs 0.11–0.19 for E2) and the lowest RSNA loss. Its AUROC effect is small and mixed (Δ −0.022 to +0.009), and for the two ViT-family models it costs Kermany-test AUROC (ViT-S 0.884, Swin-T 0.853).

**6. A frozen ImageNet backbone transfers worst, except for Swin-T.** LP is 0.07–0.15 below E2 for six architectures (RSNA-test 0.78–0.84) while the Kermany test stays high (0.94–0.96). Swin-T is the outlier: frozen features reach 0.875, only −0.020 below its own E2 (per-seed +0.005), the best LP by far. Generic natural-image features do not avoid the problem in general: they are worse at radiographs, so the network has to learn chest-X-ray features, and the question is which invariances it learns them under.

**7. Nothing removes the width association.** ρ(score, width) among healthy Kermany-test images is −0.36 to −0.65 for every variant of every architecture, usually *more* negative after augmentation than for E1. Either it is a genuine age effect (larger images ↔ older children with clearly normal lungs), or it is a cue none of these augmentations reaches. Kermany does not record age, so this cannot be resolved here; notebook 07 §9 shows the analogous age effect inside RSNA.

**Consequences.**
* The strong conventional baseline is **E2_ERASE**, chosen on RSNA-dev (6/7 architectures). Synthetic data (E3/E4) is compared with it, not only with E1 or E2.
* For the demo, an E2_ERASE model is the most defensible choice: the best and most stable configuration externally.
* What helps on RSNA is **variation and occlusion at training time**, not more of the same source or generic features. Which variation helps most depends on the architecture.
""")
b.write(out_path("06_overfitting.ipynb"))
