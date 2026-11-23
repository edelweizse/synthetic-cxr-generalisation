# Evaluating synthetic chest X-ray augmentation for cross-dataset pneumonia classification
## Technical report: prediction component


## 1. Problem and motivation

Deep-learning pneumonia classifiers trained on one hospital's radiographs often lose much of their accuracy on images from another hospital. Patient population, X-ray equipment, acquisition protocol, labelling criteria and dataset-specific artefacts all differ between sites. Generative models (here: LoRA-adapted diffusion, built by the generative component) can produce unlimited labelled "new" images, so a natural hypothesis is that synthetic training data improves robustness.

**Research question.** Does generative augmentation improve a pneumonia classifier's *external* performance, compared with no augmentation, conventional augmentation, and the same amount of additional *real* data?

**Measurable objectives**
1. A reproducible baseline classifier trained on real images from Kermany (E1).
2. Internal (Kermany test) and external (RSNA) performance with uncertainty estimates.
3. Conventional augmentation (E2) vs generative augmentation (E3/E4).
4. Effect of the amount of synthetic data (25% vs 50%).
5. Synthetic vs additional real data (R25/R50, scaling curve).
6. Conditions under which augmentation helps, hurts, or has no measurable effect.

**Primary endpoint:** Δ view-balanced AUROC on RSNA-test (Lung Opacity vs Normal) relative to E1. View-balanced AUROC is the mean of the AUROC within AP films and the AUROC within PA films. In RSNA the view is confounded with the label: 76% of AP films but 17% of PA films show opacity, and the view alone reaches a pooled AUROC of 0.80. Pooled AUROC therefore partly rewards recognising the view; it is kept as a secondary column. Other secondary endpoints: Kermany test AUROC, sensitivity and specificity at a source-chosen threshold, and the Kermany→RSNA AUROC drop.

## 2. Data

| | Kermany (Guangzhou) | RSNA Pneumonia Challenge (NIH subset, adjudicated) |
|---|---|---|
| Population | children 1–5 y, one hospital | mostly adults (median 49 y), multiple NIH sites |
| Images | 5,856 JPEG, variable size | 29,684 labelled DICOM studies, 1024×1024 |
| Labels | clinical: normal / bacterial / viral pneumonia | radiological consensus: Lung Opacity / Normal / No Lung Opacity–Not Normal |
| Role | train / validation / internal test | external test only |

**Preparation (notebook 01)**
* **Patient-level grouping.** Patient IDs are parsed from file names. Near-duplicates (normalised 32×32 thumbnails, cosine similarity > 0.99) are merged into the patient's group. Perceptual hashes were tried first and rejected: on chest X-rays thousands of different patients collide.
* **Leakage in the official split.** 170 pneumonia patient IDs occur in both the official train and test folders. IDs are global, not per pathogen: 808 training patients have both bacterial and viral images. Their 457 training images are excluded, and Kermany test remains the official 624-image test set.
* **Validation set.** The official val folder has 16 images, too few to use. Instead the pooled train+val is split 6:1 with patient-grouped, label-stratified folds (682 validation images).
* **Training subsets.** One random image per patient, then a random per-class rank (`subset_rank`). The first *k* per class form nested, balanced subsets: E1 = 500/class, R25 = 625/class, R50 = 750/class.
* **Acquisition shortcut.** Normal images in Kermany are much larger (median width 1654 vs 1120–1264 px) and much larger files. A logistic model on width alone reaches AUROC 0.95 on Kermany test, without looking at a single pixel. Letterboxing would encode this in the padding, and center-cropping cuts the lateral lung fields of wide images. All images are therefore resized with their full field of view to 224×224. RSNA is already square, so it is not distorted.
* **Label harmonisation.** RSNA's *Lung Opacity* is the closest analogue to Kermany's pneumonia, and RSNA's *Normal* to Kermany's normal. *No Lung Opacity / Not Normal* has no counterpart in Kermany, so it is excluded from the primary task and treated as negative in a secondary analysis. Kermany labels are clinical diagnoses while RSNA labels are radiological opacity findings; this mismatch is part of the measured shift.

### 2.1 Why validation AUROC is ≈0.99 from the first evaluation: an acquisition shortcut in Kermany

Every classifier, of every architecture, reaches Kermany validation AUROC ≈0.99 after only 50 optimizer steps. The reason is that the Kermany classes can be told apart **without looking at the anatomy**. A logistic regression on four file properties (log width, log height, aspect ratio, log file size), fitted on the 1,000 E1 training images, gives:

| Classifier (no pixels used) | Kermany validation AUROC | Kermany test AUROC |
|---|---|---|
| Logistic regression on width, height, aspect ratio, file size | **0.998** | 0.905 |
| Image width alone | 0.856 | 0.950 |

Normal images in Kermany are systematically larger (median 1654×1323 px vs 1120×752 px for bacterial pneumonia) and stored at much higher JPEG quality (median 518 kB vs 65 kB). The likely cause is that the classes were gathered from different archive queries, time periods or export settings, as is common in retrospectively assembled clinical datasets. The dataset was published to demonstrate transfer learning (Kermany et al., *Cell* 2018), not as a shortcut-free benchmark. The widely used Kaggle repackaging adds further issues: a 16-image validation folder and patient IDs shared between train and test.

Resizing every image to 224 px removes the size itself but not its traces (sharpness, noise, compression texture). Two findings show that the CNNs pick these up:
* Scores on Kermany test correlate with the original image width *within* each true class (notebook 05, shortcut probe).
* Without augmentation, Grad-CAM mass concentrates on the image border rather than the lungs (notebook 05).

This is the same failure mode as earlier pneumonia and COVID-19 X-ray models that learned to recognise the hospital or the dataset instead of the disease (Zech et al., *PLoS Medicine* 2018; DeGrave et al., *Nature Machine Intelligence* 2021).

Consequences for this study:
1. **Validation AUROC cannot rank checkpoints or hyper-parameters** (everything is ≈0.99). Validation BCE is used instead, because it still separates runs (≈0.04–0.25).
2. **Kermany validation, Kermany test and RSNA measure different things.** Validation shares the shortcut with training. The official test set was assembled differently (the metadata-only model drops to 0.905). RSNA does not have the shortcut at all: every image is a 1024×1024 DICOM. The Kermany-val → Kermany test → RSNA decline is therefore partly a measurement of shortcut reliance, not only of population shift.
3. **Model selection on Kermany may favour shortcut users**, because Kermany validation contains the same cue. We accept this rather than select on RSNA, which would stop RSNA from being an external test, and report it as a limitation. It also gives a concrete criterion for useful augmentation: it should make the model rely less on acquisition cues, which is exactly what external performance on RSNA rewards.

## 3. Method

**Models.** ImageNet-pretrained timm backbones: ResNet-18/50, DenseNet-121, EfficientNet-B0, ConvNeXt-Tiny, ViT-S/16 and Swin-T, each with a single-logit head. Grayscale input is replicated to 3 channels with the backbone's normalisation. CXR-pretrained weights (e.g. torchxrayvision) were deliberately not used: they were trained on NIH/RSNA, i.e. on our external domain.

**Training.**
* Loss and optimiser: BCE with logits; AdamW with no decay on norms and biases.
* Precision: bf16 autocast.
* Data: the whole training set lives on the GPU as uint8, and augmentation runs on the GPU.
* Budget and stopping: a fixed maximum of 1,500 optimizer steps, with validation every 50 steps. Early stopping after 8 validations without improvement in validation BCE. Cosine or reduce-on-plateau LR (chosen by HPO).
* Checkpoints: the checkpoint with the lowest validation BCE is kept. Validation AUROC on Kermany saturates (>0.99) and does not separate checkpoints, so BCE (a proper scoring rule) is used instead.

**Hyper-parameter optimisation.** One Optuna study per architecture on the E1 setting.
* Sampler and pruner: TPE (multivariate) with median pruning.
* Search space: learning rate, weight decay, batch size, LR schedule, warm-up, head-LR multiplier, label smoothing, dropout and stochastic depth.
* Objective: Kermany validation BCE.
* The best parameters are then frozen for every experiment of that architecture.

**Decision threshold.** Youden's J on Kermany validation, frozen for Kermany test and RSNA.

**Architecture selection for the data experiments.** A rule fixed in advance: the best CNN and the best transformer by mean E1 Kermany validation BCE. RSNA is never used for any choice.

**External split.** RSNA is split once, by patient (hash of the patient ID), into RSNA-dev (30%) and RSNA-test (70%, n ≈ 11,800 for the primary task). RSNA-dev was used only to choose between analysis variants (e.g. pooled vs view-balanced metric); every reported external number is on RSNA-test. No model, checkpoint or hyper-parameter is chosen on RSNA.

**Statistics.**
* Every configuration runs with 3 seeds (seeds change head initialisation, batch order and augmentation draws).
* Per-run 95% CIs come from a patient-clustered bootstrap (1,000 resamples).
* Comparisons use Δ AUROC with a paired, patient-clustered bootstrap on seed-averaged scores (3-seed ensembles), plus the mean ± SD of per-seed differences. For the view-balanced metric the bootstrap recomputes both within-view AUROCs in every resample.
* The bootstrap CI covers patient sampling only, not seed variance. Per-seed differences have SD 0.01–0.10, so |Δ| below about 0.03–0.04 is treated as inconclusive even when its CI excludes 0.

**Engineering.**
* Resumable runs (`last.pt` with model/optimiser/RNG/early-stopping state) and a resumable Optuna study (SQLite).
* An idempotent pipeline, with MLflow tracking of parameters, curves, metrics and artefacts.
* Deterministic cuDNN/cuBLAS; environment and git commit recorded per run; pinned requirements.

## 4. Results

All numbers are mean ± SD over 3 seeds unless stated otherwise. "RSNA" means the primary external task (Lung Opacity vs Normal, n = 16,896) unless stated otherwise. Sensitivity and specificity are at the frozen Kermany validation threshold. Δ values are the mean of per-seed differences; where a CI is given, it belongs to the Δ of the 3-seed ensembles (paired, patient-clustered bootstrap, 2,000 resamples). Source tables: `reports/results/*.csv`; figures: `reports/figures/`.

### 4.1 Hyper-parameter search
25 Optuna trials per architecture (175 in total, about 3 GPU-hours on an RTX 3060 Ti); median pruning stopped 4–12 trials per study early.

| Architecture | best Kermany-val BCE | best lr | batch | schedule |
|---|---|---|---|---|
| EfficientNet-B0 | **0.034** | 8.8·10⁻⁴ | 64 | plateau |
| DenseNet-121 | 0.038 | 6.0·10⁻⁴ | 16 | plateau |
| ResNet-50 | 0.046 | 8.4·10⁻⁴ | 32 | plateau |
| ConvNeXt-T | 0.056 | 2.5·10⁻⁴ | 64 | cosine |
| ResNet-18 | 0.065 | 6.5·10⁻⁴ | 64 | plateau |
| ViT-S/16 | 0.066 | 2.8·10⁻⁵ | 32 | cosine |
| Swin-T | 0.069 | 1.0·10⁻⁴ | 32 | plateau |

Learning rate dominated the parameter importances. The transformers prefer learning rates 10–30× lower than the CNNs. Best validation AUROC was 0.994–0.9995 for every architecture, which is why BCE is the objective (§2.1). Figures: `hpo_history.png`, `hpo_importance.png`, `hpo_lr_batch.png`.

### 4.2 E1 baseline across architectures: internal vs external

RSNA-test: view-balanced AUROC (headline), within-view AUROC and pooled AUROC (`headline_summary.csv`).

| Architecture | Kermany-val BCE | Kermany test AUROC | **RSNA-test balanced** | AP only | PA only | pooled |
|---|---|---|---|---|---|---|
| ResNet-18 | 0.063 ± 0.005 | 0.945 ± 0.005 | 0.771 ± 0.045 | 0.740 | 0.802 | 0.877 ± 0.032 |
| ResNet-50 | 0.055 ± 0.008 | 0.947 ± 0.004 | 0.747 ± 0.024 | 0.714 | 0.779 | 0.866 ± 0.021 |
| DenseNet-121 | 0.044 ± 0.006 | 0.932 ± 0.020 | **0.787 ± 0.019** | 0.763 | 0.811 | 0.883 ± 0.017 |
| EfficientNet-B0 | **0.044 ± 0.014** | 0.937 ± 0.005 | 0.736 ± 0.005 | 0.701 | 0.771 | 0.860 ± 0.004 |
| ConvNeXt-T | 0.060 ± 0.004 | 0.867 ± 0.098 | 0.713 ± 0.037 | 0.667 | 0.759 | 0.839 ± 0.035 |
| ViT-S/16 | 0.079 ± 0.012 | 0.866 ± 0.075 | 0.692 ± 0.019 | 0.668 | 0.716 | 0.824 ± 0.017 |
| Swin-T | 0.081 ± 0.013 | 0.878 ± 0.040 | 0.698 ± 0.016 | 0.629 | 0.767 | 0.838 ± 0.012 |

* **Classic CNNs are the strongest and most stable at 1,000 images.** The transformers and ConvNeXt are worse on RSNA and unstable on Kermany test (single seeds down to 0.77–0.80).
* **Every model loses 0.15–0.20 AUROC from Kermany test to RSNA (view-balanced)**; pooled, the loss looks like only 0.03–0.08. The models also over-call pneumonia on RSNA (all of RSNA, frozen threshold): sensitivity 0.84–0.91 but specificity 0.56–0.74. Counting RSNA's "not normal" studies as negatives, specificity falls to 0.38–0.53. The models learned "abnormal vs normal child", not "pneumonia vs everything else".
* **Pooled RSNA AUROC is inflated by view, by 0.10–0.14 for E1.** AP films carry 4.5× more opacity than PA films (prevalence 0.755 vs 0.167 in RSNA-test) and look different, so recognising the view alone scores 0.80 pooled AUROC. Among normal films only, every E1 model scores AP above PA (AUROC 0.84–0.92): a healthy bedside patient looks sicker to the model than a PA patient with pneumonia. This is why the view-balanced AUROC is the headline. The ranking of architectures is the same under both metrics.
* **Selection.** The pre-registered rule (best CNN + best transformer by Kermany-val BCE) picked **EfficientNet-B0** and **ViT-S** for the data experiments. In hindsight DenseNet-121 and ResNet-18 transfer better in E1; that is the honest cost of not selecting on RSNA.

### 4.3 Conventional augmentation (E2)

RSNA-test, Δ vs E1 (`headline_vs_ref.csv`): view-balanced per-seed mean ± SD, 3-seed ensemble [95% CI], within-view per-seed mean, pooled ensemble.

| Architecture | E2 balanced | Δ balanced (per-seed) | **Δ balanced ensemble [95% CI]** | Δ AP only | Δ PA only | Δ pooled | Δ Kermany test |
|---|---|---|---|---|---|---|---|
| ResNet-18 | 0.818 ± 0.019 | +0.047 ± 0.052 | +0.042 [+0.037, +0.048] | +0.051 | +0.043 | +0.021 | −0.005 (n.s.) |
| ResNet-50 | 0.825 ± 0.036 | +0.078 ± 0.045 | +0.070 [+0.062, +0.077] | +0.113 | +0.044 | +0.032 | +0.021 |
| DenseNet-121 | **0.836 ± 0.025** | +0.049 ± 0.043 | +0.051 [+0.044, +0.057] | +0.052 | +0.046 | +0.024 | +0.015 (n.s.) |
| EfficientNet-B0 | 0.833 ± 0.024 | +0.097 ± 0.029 | **+0.100** [+0.093, +0.108] | +0.125 | +0.069 | +0.051 | −0.006 (n.s.) |
| ConvNeXt-T | 0.808 ± 0.015 | +0.095 ± 0.047 | +0.086 [+0.079, +0.093] | +0.136 | +0.053 | +0.044 | +0.087 |
| ViT-S/16 | 0.775 ± 0.086 | +0.083 ± 0.103 | +0.088 [+0.080, +0.096] | +0.115 | +0.051 | +0.052 | +0.059 |
| Swin-T | 0.745 ± 0.060 | +0.047 ± 0.075 | +0.058 [+0.047, +0.067] | +0.043 | +0.051 | +0.029 | +0.007 (n.s.) |

* **E2 improves RSNA for all 7 architectures, and every CI excludes 0.** The view-balanced gain (+0.04 to +0.10) is about twice the pooled gain (+0.02 to +0.05). The gain is not better view recognition: it appears within both views (PA +0.04…+0.07, AP +0.04…+0.14), largest on AP films.
* **Kermany test is mixed** (−0.006 to +0.087); significant gains only where augmentation stabilises seeds that otherwise collapse (ConvNeXt, ViT, ResNet-50).
* **In-domain validation would have rejected E2.** E2's Kermany-val BCE is worse than E1's for 5 of 7 architectures (EfficientNet-B0 0.044 → 0.058). Within one recipe, lower validation BCE does go with better RSNA (Spearman ρ = −0.51 for E1, −0.63 for E2); across recipes, it does not (ρ = −0.28, p = 0.07).
* **Calibration improves:** RSNA ECE for EfficientNet-B0 0.248 → 0.161, ViT-S 0.284 → 0.107.
* **Seed ensembles** help: the 3-seed E2 ensembles reach 0.840–0.869 view-balanced (EfficientNet-B0 0.869, DenseNet-121 0.861), above the per-seed means (0.745–0.836).

### 4.4 More real data: controls and scaling

RSNA-test view-balanced AUROC (mean ± SD over seeds), within-view AUROC (AP / PA) and pooled AUROC; Kermany test AUROC for reference.

| Training set (n images) | EfficientNet-B0 balanced | AP / PA | pooled | ViT-S balanced | AP / PA | pooled | Kermany test (Eff / ViT) |
|---|---|---|---|---|---|---|---|
| N250 (250) | 0.724 ± 0.010 | 0.706 / 0.742 | 0.844 | 0.633 ± 0.056 | 0.621 / 0.644 | 0.754 | 0.944 / 0.913 |
| N500 (500) | 0.731 ± 0.017 | 0.681 / 0.781 | 0.864 | 0.675 ± 0.085 | 0.679 / 0.671 | 0.794 | 0.939 / 0.903 |
| **E1** (1,000) | 0.736 ± 0.005 | 0.701 / 0.771 | 0.860 | 0.692 ± 0.019 | 0.668 / 0.716 | 0.824 | 0.937 / 0.866 |
| R25 (1,250) | 0.763 ± 0.031 | 0.758 / 0.768 | 0.864 | 0.733 ± 0.053 | 0.712 / 0.754 | 0.853 | 0.888 / 0.849 |
| R50 (1,500) | 0.763 ± 0.046 | 0.760 / 0.765 | 0.866 | 0.747 ± 0.028 | 0.746 / 0.749 | 0.863 | 0.885 / 0.885 |
| N2000 (2,000) | 0.768 ± 0.014 | 0.768 / 0.769 | 0.867 | 0.725 ± 0.052 | 0.705 / 0.746 | 0.848 | 0.887 / 0.740 |
| FULL (4,093) | 0.785 ± 0.005 | 0.789 / 0.780 | 0.878 | 0.782 ± 0.020 | 0.774 / 0.790 | 0.888 | 0.891 / 0.855 |
| **E2** (1,000 + aug.) | 0.833 ± 0.024 | 0.826 / 0.840 | 0.915 | 0.775 ± 0.086 | 0.783 / 0.767 | 0.870 | 0.932 / 0.926 |
| FULL_AUG (4,093 + aug.) | 0.836 ± 0.016 | 0.823 / 0.850 | 0.916 | 0.839 ± 0.012 | 0.855 / 0.824 | 0.915 | 0.955 / 0.899 |

* **For EfficientNet-B0, augmentation on 1,000 images beats 4× more real data** (E2 +0.097 vs FULL +0.049 view-balanced), and real data on top of augmentation adds almost nothing (FULL_AUG 0.836 vs E2 0.833). The bottleneck is the source's lack of variation more than its size.
* **ViT-S is data-hungry; data and augmentation are complementary.** It gains about +0.035 view-balanced AUROC per doubling of real data (about +0.015 for EfficientNet-B0), and FULL_AUG is its best configuration here (0.839).
* **The bar for synthetic data (matched real controls):** +25% / +50% real images give EfficientNet-B0 +0.027 / +0.027 (per-seed mean; ensemble +0.027 / +0.017, CIs above 0) and ViT-S +0.041 / +0.055. Under the pooled metric the EfficientNet-B0 gain was invisible (+0.004 / +0.005), because it lands on AP films. E3/E4 must also be compared with E2 (+0.10).
* **More real data helps EfficientNet-B0 on AP films only** (AP +0.06…+0.09, PA flat), while augmentation helps both views.
* **Kermany test often drops with more Kermany data** (EfficientNet-B0 R50 −0.053, N2000 −0.051; ViT-S N2000 −0.18). This is consistent with stronger reliance on acquisition cues that hold only partly in the official test set (§2.1), but it is a hypothesis, not a demonstrated mechanism.
* **The frozen threshold is fragile.** Kermany-val Youden thresholds range 0.50–0.94, giving RSNA operating points from 0.70 / 0.88 (EfficientNet FULL) to 0.97 / 0.16 (ViT N250) sensitivity / specificity. E2 and FULL_AUG are the most balanced.

### 4.5 Generative augmentation (E3/E4)

**Synthetic data.** The generative component's Stable Diffusion 1.5 + LoRA (r = 16), with one fixed prompt per class, a shared negative prompt, CFG 4.5 and 30 steps (4,000 normal + 1,000 pneumonia images, 512 px). **E3** adds 250 synthetic images (125 per class) to E1's 1,000 real images, **E4** adds 500. Both are trained without augmentation, so their baseline is E1, and their matched real controls are R25/R50 (the same number of extra *real* Kermany images). `ERASE_SYN250/500` repeat this on top of E2_ERASE augmentation, with `ERASE_REAL250/500` as the real controls. Full log: `docs/FINDINGS_SYNTHETIC.md`.

**Main result.** RSNA-test, Δ view-balanced AUROC of the 3-seed ensembles [95% CI] (`view_compare.csv`):

| Architecture | + 250 real (R25) | + 500 real (R50) | **+ 250 synthetic (E3)** | **+ 500 synthetic (E4)** | E2_ERASE + 250 real | **E2_ERASE + 250 synthetic** |
|---|---|---|---|---|---|---|
| EfficientNet-B0 | +0.027 | +0.017 | **−0.075** [−0.085, −0.066] | **−0.103** [−0.112, −0.093] | −0.016 | **−0.145** [−0.155, −0.134] |
| ResNet-18 | −0.021 | +0.003 | **−0.081** [−0.089, −0.074] | **−0.148** [−0.156, −0.139] | +0.008 | **−0.053** [−0.060, −0.047] |
| ResNet-50 | +0.002 | +0.018 | **−0.063** [−0.072, −0.055] | **−0.082** [−0.091, −0.073] | −0.016 | **−0.079** [−0.086, −0.072] |
| DenseNet-121 | +0.030 | +0.028 | **−0.038** [−0.046, −0.031] | **−0.019** [−0.027, −0.011] | +0.005 | **−0.043** [−0.049, −0.035] |
| ConvNeXt-T | +0.015 | −0.017 | **−0.041** [−0.049, −0.034] | **−0.088** [−0.097, −0.079] | +0.012 | **−0.030** [−0.035, −0.026] |
| ViT-S/16 | +0.035 | +0.041 | **−0.051** [−0.058, −0.044] | **−0.090** [−0.098, −0.081] | −0.003 | **−0.118** [−0.125, −0.111] |
| Swin-T | +0.051 | −0.056 | **−0.104** [−0.114, −0.094] | **−0.054** [−0.063, −0.045] | +0.007 | **−0.031** [−0.037, −0.024] |

* **Synthetic images hurt in every comparison: 14 of 14 without augmentation (−0.019 to −0.148) and 7 of 7 with augmentation (−0.030 to −0.145)**, all with the CI below 0. The same number of extra real images changes little (−0.056 to +0.051, mostly positive). Against those real controls, synthetic data costs −0.047 to −0.156 (all except Swin-T E4, whose R50 control is itself weak).
* **Augmentation does not protect against the synthetic images.** For EfficientNet-B0 and ViT-S the penalty is larger on top of E2_ERASE (−0.145, −0.118) than without augmentation (−0.075, −0.051).
* **The damage lands on AP films** (EfficientNet-B0: AP −0.10 / −0.11 without augmentation, −0.25 with it; PA −0.05 / −0.10). Pooled AUROC shows only a third to a half of the effect (EfficientNet-B0 E3 pooled −0.044).
* **More synthetic data does not fix it.** From 250 to 2,000 synthetic images (8×) the EfficientNet-B0 penalty stays between −0.063 and −0.103, with no trend. A model trained on 2,000 synthetic images *only* reaches 0.88 on Kermany test but is at chance on RSNA PA films (0.50; Δ balanced −0.248). The generator encodes Kermany's appearance, not a transferable notion of pneumonia.

**What the synthetic images are like** (`docs/FINDINGS_SYNTHETIC.md` §2–§6, §9):
1. *They copy, and amplify, Kermany's image-quality shortcut.* Low sharpness predicts pneumonia in Kermany (AUROC 0.87) and in the synthetic set (0.86); low contrast 0.75 in Kermany, 0.89 in the synthetic set. Within RSNA views both cues are nearly useless (0.50–0.60). On real RSNA films, synthetic training doubles the weight the model gives to blur relative to opacity (E1 0.36 → E4 0.91 on AP films).
2. *Their pathology is weak.* A robust real-trained classifier (E2_ERASE) gives synthetic pneumonia a mean probability of 0.28–0.54, against 0.96–0.98 for real pneumonia.
3. *They form a domain of their own.* No synthetic image is a near-copy of a training image, but a linear probe separates real from synthetic perfectly (AUROC 1.000), and 5,000 synthetic images map to only 658 distinct nearest real images.

**Can the penalty be removed?** EfficientNet-B0 unless noted, Δ view-balanced vs the matching baseline:

| Intervention | Penalty of adding synthetic images |
|---|---|
| none (E1 → E3 / E4) | −0.075 / −0.103 |
| quality-aware selection: quality cue removed or reversed inside the synthetic set (E1 → E3_QC / E4_QM) | −0.044 / −0.042 |
| histogram-equalised preprocessing, re-tuned (E1_HARMT → E4_HARMT) | −0.080 |
| quality-consistency loss (qcons; E1 + qcons → E3 + qcons), 7 architectures | −0.005 to −0.050 |
| erasing augmentation + qcons (→ ERASE_SYN250 + qcons) | −0.035 |

* **Removing the quality cue halves the penalty but never turns it positive.** The best configurations only approach the baseline.
* **The quality-consistency loss is a useful by-product.** It penalises changes of the logit under random blur and contrast reduction. On real data alone it gains +0.066 for EfficientNet-B0 and +0.075 for ViT-S (view-balanced), but it stays below conventional augmentation (E2 −0.034 vs E1 + qcons) and adds nothing on top of it.

**Which property causes the penalty?** (`docs/RESEARCH_PIVOT.md`; EfficientNet-B0, ResNet-18, DenseNet-121.) Each arm adds *real* Kermany images that copy one property of the synthetic set, or filters the synthetic set by anatomy:
* *Amplified shortcut* (MIMIC_Q: real images with the synthetic per-class sharpness/contrast profile), *weak labels* (MIMIC_N: 30% of the pneumonia slots hold quality-matched normals) and both together reproduce only 8–16% of the penalty (median over 6 architecture × size cells).
* *Anatomy:* dropping the 20% anatomically most atypical synthetic images (torchxrayvision segmentation of 14 structures) leaves the penalty unchanged (99% of it remains), and putting the worst first does not make it larger.
* *Stronger label noise* (50% / 70% of pneumonia slots mislabelled, i.e. 25% / 35% of all added labels): Δ vs E1 between −0.056 and +0.013, not monotonic in the dose (70% is no worse than 50%). For EfficientNet-B0 and ResNet-18 this is 0–60% of the synthetic penalty (median about 25%): even a third of the added labels wrong costs at most about 0.05, while synthetic images cost 0.075–0.15. For DenseNet-121, whose synthetic penalty is the smallest (−0.038 / −0.019), noise reproduces all of it.

So neither the shortcut, nor weak labels, nor gross anatomy, alone or the first two combined, explains the large penalties, which real images of the same quality and label noise do not produce. What remains is the synthetic *domain* itself: images that are perfectly separable from real ones and whose class differences are not those of real disease.

**Counterfactual pairs** (`docs/RESEARCH_PIVOT.md` §7) test this directly. A real Kermany image is inpainted inside a lung mask twice with the same generator, seed and mask, once with the normal and once with the pneumonia prompt; pixels outside the mask are pasted back. The two images of a pair therefore share hospital, patient, anatomy and acquisition, and differ only in generated lung content. Strength 1.0 regenerates the lungs completely, strength 0.8 keeps some of the original. E3 / E4 sizes, 3 seeds, Δ view-balanced vs E1:

| Arch, size | + real | + sampled synthetic | + CF pairs, strength 1.0 | + CF pairs, strength 0.8 |
|---|---|---|---|---|
| EfficientNet-B0, +250 / +500 | +0.027 / +0.017 | −0.075 / −0.103 | −0.076 / −0.032 | −0.102 / −0.162 |
| ResNet-18, +250 / +500 | −0.021 / +0.003 | −0.081 / −0.148 | −0.028 / −0.051 | −0.096 / −0.139 |
| DenseNet-121, +250 / +500 | +0.030 / +0.028 | −0.038 / −0.019 | −0.125 / −0.115 | −0.117 / −0.150 |

(Bootstrap CIs ±0.01; `reports/results/synth_probe/cf_compare.csv`.)
* **Counterfactual pairs hurt in all 12 cells.** Removing every difference except the lungs lowers the penalty for ResNet-18 and EfficientNet-B0 +500, leaves it unchanged for EfficientNet-B0 +250, and triples it for DenseNet-121.
* **The weaker edit (0.8) hurts most.** Its two images are nearly identical, yet carry opposite labels. Robust real-trained classifiers (E2_ERASE) rank the pneumonia edit above the normal edit in 80–84% of pairs at strength 1.0 but only 72–75% at 0.8, with mean pneumonia scores of 0.08–0.22. The edit carries little that a real-trained model recognises as pneumonia.
* **Interpretation.** Mislabelled real images are *inconsistent* noise: nothing systematic separates them, so models largely ignore them. A counterfactual pair offers a *consistent* class difference that consists mostly of generator-specific lung texture. The model learns that texture as "pneumonia", and it does not exist in RSNA. When the hospital shortcut is removed, the generator's texture takes its place.
* **Open control.** A "no-op" edit (both images inpainted with the same class prompt) would separate the inpainting texture itself from the weak class signal. It has not been run.

The working conclusion is that synthetic data hurts **whenever the only consistent difference between its classes is a feature of the generator rather than of the disease**. How strongly a robust real-trained classifier separates the synthetic classes does *not* predict the harm on its own: it is 0.55–0.60 within the counterfactual set, but 0.88 for the sampled synthetic images (E2_ERASE ensemble), which hurt about as much. Part of that 0.88 is the quality shortcut, which the robust classifiers still use.

**Making the synthetic images help** (`docs/RESEARCH_PIVOT.md` §8). If the penalty comes from the *labels* (the prompt) rather than the images, relabelling should remove it. TEACH keeps the same sampled synthetic images as E3/E4 but replaces the prompt label with the soft probability of a robust real-trained teacher (mean of the E2_ERASE EfficientNet-B0, DenseNet-121 and ViT-S models). RTEACH applies the same teacher labels to the extra real images of R25/R50, to separate the effect of the images from plain distillation. CTX keeps the real lungs of E1's own images and lets the generator redraw only the surroundings with one class-neutral prompt. Δ view-balanced vs E1, 3 seeds, +250 / +500:

| Arch | + real | + real, teacher labels | + synthetic, prompt labels | **+ synthetic, teacher labels** | CTX |
|---|---|---|---|---|---|
| EfficientNet-B0 | +0.027 / +0.017 | +0.040 / +0.037 | −0.075 / −0.103 | **+0.038 / +0.027** | +0.011 / +0.011 |
| ResNet-18 | −0.021 / +0.003 | −0.023 / −0.016 | −0.081 / −0.148 | **−0.032 / −0.019** | −0.032 / −0.050 |
| DenseNet-121 | +0.030 / +0.028 | +0.014 / +0.026 | −0.038 / −0.019 | **+0.053 / +0.062** | +0.036 / +0.003 |

(`reports/results/synth_fix/fix_compare.csv`.)
* **Teacher labels remove the synthetic penalty in all 6 cells** (+0.08 to +0.13 over prompt labels). With them, the synthetic images are worth about as much as the same number of real images, and for DenseNet-121 more (+0.053 / +0.062 vs +0.030 / +0.028).
* For EfficientNet-B0 and ResNet-18, teacher-labelled synthetic ≈ teacher-labelled real: the gain is distillation from the teacher, and the synthetic images serve as an unlabelled pool. Only DenseNet-121 gains beyond RTEACH (+0.039 / +0.036), at the edge of seed noise.
* The teacher gives the average synthetic "pneumonia" image a probability of 0.37. Hard prompt labels force the classifier to separate the prompts by whatever differs, which is generator texture. Soft labels that follow the visible evidence remove that incentive. This confirms the conclusion above from the constructive side.
* CTX (generative context randomisation) is inconclusive: small gains for EfficientNet-B0 and DenseNet-121 come from PA films; ResNet-18 loses on AP films.

### 4.6 Error analysis
Notebook 05, on E1/E2 × EfficientNet-B0/ViT-S.

1. **AP/portable films → false positives.** This is the largest single error source on RSNA. For E1, specificity on normal AP films is about 0.24: 3 of 4 normal AP films are flagged. E2 ViT raises it to 0.60. Grad-CAM shows diffuse thorax-wide heat plus the upper border and "PORTABLE"/"AP" text markers: a global "sick-patient film" response.
2. **Focal adult consolidations on clean PA films → false negatives.** Grad-CAM often finds the opacity, yet the score stays near 0.01: the clean PA appearance overrides local evidence. Pneumonia in Kermany is mostly diffuse.
3. **Other diseases → called pneumonia.** "No Lung Opacity / Not Normal" studies get a median score of 0.98.
4. **Over-confidence and threshold shift.** RSNA-optimal thresholds are 0.95–0.97 for three of four models but 0.26 for E2 ViT, so no fixed correction works. Local recalibration on a small labelled sample would fix part of the frozen-threshold error, but not the lost AUROC.
5. **Residual association with acquisition properties.** Among normal Kermany test images, scores fall with original image width (Spearman ρ −0.39 to −0.61, p < 10⁻⁴), also after augmentation. Width is confounded with patient age, so this is association, not proof.

Subgroups: sex differences are small (≈0.02–0.03 for E1, none after augmentation). The pediatric part of RSNA (<18 y, n = 982) is *not* easier (E1 EfficientNet-B0 0.815 vs 0.86–0.87 for adults 18–60): the shift is hospital, acquisition and labelling more than age. The elderly (>60 y) have the lowest specificity (0.51–0.61 for E1).

Grad-CAM border share (outer 15% frame; uniform = 0.51): EfficientNet-B0 34–41%; ViT-S E1 60% (Kermany test) / 51% (RSNA), lowered by E2 to 53% / 47%.

### 4.7 Targeted anti-overfitting variants
Notebook 06. Five variants built on E2, each aimed at a mechanism from §2.1 and §4.6 (definitions in notebook 06): **E2_ACQ** (blur, sensor noise, down/up-sampling), **E2_ERASE** (random erasing + random border masking), **E2_ROBUST** (both), **E2_MIX** (mixup α = 0.4), **LP** (frozen ImageNet backbone, linear head only). Same 1,000 images, budget, early stopping and selection rules; EfficientNet-B0, ViT-S and DenseNet-121; 3 seeds.

**Two kinds of overfitting.** Every configuration fits its training images almost perfectly (train BCE 0.005–0.04, train AUROC ≈1.000), while Kermany validation BCE stays low (0.04–0.09). Early stopping therefore contains classic overfitting. The damage is out of distribution: for E1, BCE rises to 0.59–0.70 on Kermany test and 0.76–1.36 on RSNA, i.e. **9–17× the validation loss**, which Kermany validation cannot see. E2 cuts the Kermany test / RSNA loss by 25–50% (`overfitting_ladder.png`).

RSNA-test view-balanced AUROC (mean ± SD over 3 seeds; all seven architectures, `headline_summary.csv`):

| Variant | EfficientNet-B0 | DenseNet-121 | ViT-S/16 | ResNet-18 | ResNet-50 | ConvNeXt-T | Swin-T |
|---|---|---|---|---|---|---|---|
| E1 | 0.736 ± 0.005 | 0.787 ± 0.019 | 0.692 ± 0.019 | 0.771 ± 0.045 | 0.747 ± 0.024 | 0.713 ± 0.037 | 0.698 ± 0.016 |
| E2 | 0.833 ± 0.024 | 0.836 ± 0.025 | 0.775 ± 0.086 | 0.818 ± 0.019 | 0.825 ± 0.036 | 0.808 ± 0.015 | 0.745 ± 0.060 |
| E2_ACQ | 0.826 ± 0.018 | 0.788 ± 0.066 | 0.789 ± 0.016 | 0.822 ± 0.026 | 0.849 ± 0.007 | **0.844 ± 0.004** | **0.828 ± 0.009** |
| **E2_ERASE** | **0.851 ± 0.003** | **0.858 ± 0.013** | **0.849 ± 0.015** | **0.826 ± 0.011** | **0.863 ± 0.024** | 0.841 ± 0.024 | 0.813 ± 0.015 |
| E2_ROBUST | 0.822 ± 0.031 | 0.797 ± 0.052 | 0.828 ± 0.022 | 0.799 ± 0.046 | 0.833 ± 0.021 | 0.840 ± 0.036 | 0.809 ± 0.020 |
| E2_MIX | 0.805 ± 0.014 | 0.855 ± 0.005 | 0.789 ± 0.015 | 0.793 ± 0.012 | 0.801 ± 0.029 | 0.814 ± 0.038 | 0.768 ± 0.052 |
| LP | 0.654 ± 0.004 | 0.768 ± 0.007 | 0.677 ± 0.014 | 0.710 ± 0.008 | 0.692 ± 0.009 | 0.682 ± 0.011 | 0.759 ± 0.007 |

Within-view AUROC for E2_ERASE is even across views (EfficientNet-B0 / DenseNet-121 / ViT-S: AP 0.855 / 0.859 / 0.851, PA 0.847 / 0.858 / 0.848); for E1 it is AP 0.701 / 0.763 / 0.668, PA 0.771 / 0.811 / 0.716.

* **E2_ERASE is the best variant for five of seven architectures** and among the most stable (seed SD 0.003–0.024). It brings EfficientNet-B0, DenseNet-121 and ViT-S to ≈0.85 view-balanced and makes them equally good on AP and PA films. Its per-seed gain over E2 is +0.008 (ResNet-18) to +0.075 (ViT-S); part of the ViT gain is stability (one E2 ViT seed stopped at step 100). The two modern architectures, ConvNeXt-T and Swin-T, do best with acquisition augmentation (E2_ACQ), Swin-T clearly (+0.015 over E2_ERASE). Erasing raises Kermany test for EfficientNet-B0, DenseNet-121 and ViT-S (per-seed +0.02…+0.03). E2_ERASE also had the best RSNA-dev AUROC for six of seven architectures, so choosing it as the reference did not depend on RSNA-test.
* **The hypothesised mechanism holds only partly.** Erasing was meant to move Grad-CAM mass from the periphery into the lungs. That happens for DenseNet-121 (border share on RSNA 0.59 → 0.40) and slightly for ViT-S (0.51 → 0.46), but for EfficientNet-B0 it *rises* (0.46 → 0.59). Erasing improves transfer without consistently changing where the model looks; reliance on several pieces of evidence rather than one is a more plausible explanation (seed 0, 200 images per set; Grad-CAM is coarse).
* **E2_ACQ does not deliver what it was designed for.** It does not weaken the association between score and original image width among normal Kermany test images (ρ −0.49 to −0.57 vs −0.49 to −0.61 for E2), and its effect on RSNA (view-balanced, vs E2) depends on the architecture: −0.048 (DenseNet-121), −0.007 (EfficientNet-B0), +0.004 (ResNet-18), +0.014 (ViT-S), +0.024 (ResNet-50), +0.036 (ConvNeXt-T), +0.083 (Swin-T). It helps the modern architectures and hurts DenseNet-121. **ACQ and ERASE are not complementary:** E2_ROBUST is below E2_ERASE for all seven architectures (by 0.001–0.061). No variant removes the width association (ρ −0.42 to −0.62 throughout), which leaves open whether it is a shortcut or an age effect (Kermany records no age).
* **Mixup fixes confidence more than ranking.** It has the best RSNA calibration for every architecture (ECE 0.06–0.10 vs 0.11–0.19 for E2) and the lowest RSNA loss (BCE 0.42–0.45 vs 0.51–0.65). Its AUROC effect is architecture-specific: DenseNet-121 0.855 view-balanced (≈E2_ERASE; best AP-only, 0.869), EfficientNet-B0 −0.028 vs E2, ViT-S +0.014 with a Kermany test drop.
* **A frozen ImageNet backbone transfers worst** (RSNA view-balanced 0.65–0.77, −0.07 to −0.18 vs E2 for six of seven architectures; AP-only down to 0.60). Swin-T is the exception: its frozen features reach 0.759, above its unstable E2 (0.745) though well below E2_ACQ, even though its Kermany test AUROC is high (0.957–0.960). The hypothesis that generic features avoid Kermany-specific overfitting is refuted: the network must learn radiograph-specific features, and the question is only which invariances it learns them under.

**Consequence.** E2_ERASE is the strongest conventional baseline and the reference that E3/E4 must beat; it is also the model exported for the demo (selected by Kermany validation like every other run).

## 5. Discussion

**What limits external performance is the lack of variation in the source more than its size.** On 1,000 images, conventional augmentation gains +0.05 to +0.10 view-balanced RSNA AUROC for all seven architectures (§4.3), while four times as much real data gains +0.05 for EfficientNet-B0 and adds almost nothing once augmentation is used (§4.4). Targeted occlusion (E2_ERASE) brings three architectures from different families to the same ≈0.85 view-balanced AUROC with a small seed spread (§4.7). The interventions that help all make the source *harder to exploit* rather than *larger*.

**Source-domain validation is a misleading guide to generalisation.** Kermany validation AUROC is ≈0.99 for every model (§2.1). Validation BCE ranks models trained with the same recipe sensibly, but it prefers E1 over E2 for 5 of 7 architectures, the opposite of the external result (§4.3). Even the pre-registered architecture rule picked an E1 model that transfers worse than two of its competitors (§4.2). We kept RSNA out of every decision anyway: selecting on RSNA would make the reported external numbers optimistic, and a real deployment would not have RSNA either. The practical lesson is that a training *recipe* has to be validated on a second source; validation on the first source can only choose checkpoints and seeds.

**What the remaining errors are.** At ≈0.92, the leading errors are AP/portable normals scored as pneumonia, focal adult consolidations on PA films that are missed, and other diseases called pneumonia (§4.6). None of these can be fixed by more images like Kermany's. Kermany has no AP films of adults, no focal adult consolidations and no "other disease" class. The useful content of extra training data, real or synthetic, is defined by these gaps.

**Synthetic augmentation clears none of the bars.** The study set three bars that generative augmentation had to clear:
1. **More of the same real data** (R25/R50): +0.02 to +0.03 for EfficientNet-B0, +0.04 for ViT-S.
2. **Conventional augmentation** (E2): +0.10 for EfficientNet-B0.
3. **Targeted occlusion** (E2_ERASE): ≈0.85 view-balanced for EfficientNet-B0, DenseNet-121 and ViT-S.

The synthetic images fail even the lowest bar: they lower external AUROC in all 21 comparisons (§4.5), whereas the same number of real images is neutral or helps. A LoRA-adapted generator trained on Kermany learns Kermany's distribution, including its quality shortcut, which it amplifies. But the shortcut is not the whole explanation. Real images that copy the synthetic quality profile, or carry the same or much stronger label noise, are about as harmless as random real images. Removing the anatomically worst images changes nothing. The remaining suspect is the synthetic domain itself. A classifier can tell synthetic from real perfectly, and when the two classes are synthetic, the easiest features for separating them need not be features of real disease. A generator would have to add variation that Kermany lacks (acquisition, positioning, adult pathology) *and* put its class differences where real disease differs; this generator does neither.

**Uncertainty and what the numbers do not say.**
* With about 11,800 RSNA-test images, bootstrap CIs are tight (±0.004–0.010 for the view-balanced metric), so almost every difference is "significant". Seed-to-seed variation (per-seed Δ SD 0.01–0.10) is the binding uncertainty, so we treat |Δ| < 0.03–0.04 as inconclusive.
* Pooled RSNA AUROC partly rewards recognising AP films (view alone: 0.80). Hence the view-balanced headline. The two metrics rank all configurations almost identically (Spearman 0.985), no conclusion changes sign, and most effects are larger under the balanced metric.
* Kermany test behaves erratically (it often falls with more training data) and is not a clean "internal" estimate, because the official test set was assembled differently from train/validation.
* Mechanistic claims are weaker than performance claims. Erasing works, but Grad-CAM shows it does not consistently move attention into the lungs. The association between score and image width survives every intervention and cannot be separated from patient age.

**Answers to the objectives** (§1):
1. A reproducible baseline: done (7 architectures × 3 seeds, HPO, MLflow).
2. Internal and external performance with uncertainty: done (per-run and paired bootstrap CIs plus seed SD; view-stratified RSNA).
3. Conventional vs generative augmentation: conventional augmentation helps every architecture (+0.05 to +0.15 view-balanced); generative augmentation hurts every architecture (§4.5).
4. 25% vs 50% synthetic data: both hurt. 50% hurts more for five of seven architectures, and 250–2,000 synthetic images give a similar penalty.
5. Synthetic vs real data: the same number of real images is neutral to positive, synthetic images cost −0.05 to −0.16 against them.
6. When augmentation helps: conventional augmentation helps externally always, and most for AP films, unstable architectures (ViT, ConvNeXt) and calibration. Internally, results are mixed. Generic features (linear probe) do not help, and acquisition jitter helps only the modern architectures. Synthetic augmentation did not help under any condition tested (seven architectures, 250–2,000 images, with or without augmentation, quality-aware selection, harmonised preprocessing, four training losses).

## 6. Responsible AI, risks and limitations

**Intended use.** A research instrument for measuring the value of training data. Not a diagnostic device. The demo shows a non-dismissable warning, and outputs are labelled as model scores, not probabilities or diagnoses.

**Risks and mitigations**

| Risk | Where it shows up here | Mitigation |
|---|---|---|
| Domain shift / silent failure on other hospitals | large AUROC drop Kermany→RSNA; threshold miscalibrated on RSNA | external validation reported as the primary result; frozen threshold; demo states the training population and shows the external metrics |
| Shortcut learning | image size/format predicts the label in Kermany; markers and borders | full-FOV resizing; shortcut probe; Grad-CAM border-mass analysis; error galleries |
| Population bias | trained only on Chinese children aged 1–5; RSNA is mostly US adults; AP/PA confounding in RSNA | subgroup analysis by view, sex and age; results reported per subgroup; no claims beyond the tested populations |
| Label noise / definition mismatch | Kermany: clinical diagnosis; RSNA: radiological opacity; known label noise in Kermany | harmonised primary task; secondary analysis with "not normal" studies |
| Synthetic data artefacts | synthetic images amplify the source's quality shortcut, carry weak pathology, some have broken anatomy; adding them lowers external AUROC everywhere (§4.5) | E3/E4 compared against matched real controls; anatomy screening by segmentation; quality-cue audit; recommendation not to use this generator for training; placeholder runs excluded from reporting |
| Data leakage | patient overlap in the official Kermany split | patient-grouped splits, duplicate detection, exclusion of shared patients |
| Privacy | medical images | only public, de-identified datasets; no uploads are stored by the demo beyond a local temporary file |
| Over-trust / misuse | a score might be read as a diagnosis | research-only labelling, no treatment recommendation, explicit limitations in the UI |

**Limitations.**
* There is a single source dataset, and the external set differs in age, so effects of population and acquisition shift cannot be separated.
* The 224 px resolution may lose subtle findings.
* Three seeds give limited power for small effects.
* Optuna was run once per architecture (seed 0).
* The pediatric subset of RSNA is small.
* Grad-CAM is a coarse and sometimes unfaithful attribution method.
* The demo's RSNA example images were chosen using the models' RSNA predictions, to illustrate typical cases (including an AP normal that E1 flags and E2_ERASE does not). They are illustrations, not evidence.

## 7. Reproducibility

See README. Every number in this report can be regenerated with `scripts/pipeline.sh`, `scripts/pipeline_stage2.sh`, `scripts/pipeline_stage3.sh` and notebooks 01–06 (notebook sources: `notebooks/_build/`). Run configurations, environments and metrics are in MLflow and in `artifacts/runs/*/*/seed*/run.json`.
