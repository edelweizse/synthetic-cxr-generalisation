# Findings: SD-LoRA synthetic data and the image-quality shortcut

Running log of what we found when the teammate's synthetic images arrived (2026-10-12). Feeds report §4.5 and §5.
Numbers are EfficientNet-B0 unless stated. **Preliminary** marks results with fewer than 3 seeds or a single architecture.
Visual version of every finding: **`notebooks/07_synthetic.ipynb`** (figures `reports/figures/synth_*.png`).

### Naming: augmentation vs synthetic data vs preprocessing

| Term | What it is | Applied to | Runs |
|---|---|---|---|
| **Augmentation** | random transformations of the *same real images* (flip, rotation, brightness/contrast, erasing…), drawn fresh at every training step. No new images | training only | **E2** (conventional), **E2_ERASE** (+ random erasing), E2_ACQ, E2_ROBUST, E2_MIX |
| **Synthetic data** | *new images* from the teammate's generator (Stable Diffusion + LoRA), added to the training set as extra examples | training only | **E3** (+250 synthetic), **E4** (+500 synthetic); tag `sd_lora` |
| **Preprocessing** | a fixed transformation of *every* image, training **and** test | everything | default `squash`; runs tagged **HARM** use `harmonize224` (§5) |

Reference runs: **E1** = the 1,000 real images only (baseline); **R25 / R50** = +250 / +500 *real* images (the fair control for E3/E4).

## 0. The data

| | |
|---|---|
| Location | `data/synthetic/sd_lora/` (images + `manifest.csv`), `data/synthetic/sd_lora_adapter/` (LoRA weights) |
| Generator | Stable Diffusion 1.5 + LoRA (r = 16, alpha = 16, dropout 0.05, UNet attention projections), PEFT 0.21.2 |
| Sampling | one fixed prompt per class, shared negative prompt, 30 steps, CFG 4.5, one seed per image |
| Images | 4,000 NORMAL + 1,000 PNEUMONIA, 512 × 512 RGB PNG (both classes square) |
| Training manifest | `artifacts/manifests/synthetic_sd_lora.csv` (`scripts/prepare_synthetic.py`) |
| **Unknown** | strict vs transfer generator training set, LoRA steps, any filtering (`docs/SYNTHETIC_DATA.md` §2). Asked the teammate |

## 1. Synthetic images hurt external performance, mostly on AP films

E3 = real 1k + 250 synthetic, E4 = real 1k + 500 synthetic, no augmentation, so the same-setting baseline is E1.
R25/R50 are the matched real-data controls.

EfficientNet-B0, **RSNA-test** (the 70% of RSNA patients used for reporting, §7), mean over seeds:

| Run | Seeds | RSNA-test AUROC (pooled) | AP only | PA only | Mean p, healthy AP | Mean p, healthy PA |
|---|---|---|---|---|---|---|
| E1 | 3 | 0.860 ± 0.004 | 0.701 | 0.771 | 0.83 | 0.32 |
| R25 | 3 | 0.863 ± 0.028 | 0.758 | 0.768 | 0.71 | 0.25 |
| R50 | 3 | 0.866 ± 0.032 | 0.760 | 0.765 | 0.60 | 0.20 |
| E3 | 3 | 0.820 ± 0.008 | **0.619** | 0.733 | **0.90** | 0.40 |
| E4 | 3 | 0.789 ± 0.064 | **0.612** | 0.688 | **0.94** | 0.47 |
| *E2_ERASE (augmentation, for reference)* | 3 | *0.921 ± 0.002* | *0.855* | *0.847* | *0.73* | *0.14* |

Adding synthetic images lowers RSNA AUROC (E4's larger drop comes mostly from one weak seed, 0.72; §8 shows the penalty does not grow with the number of images). The same number of extra *real* images
(R25/R50) does not, and on AP films it helps. The drop is largest on AP films, where the models score healthy patients as
pneumonia (mean p 0.90–0.94). Strong augmentation on the same 1k real images beats every data-adding variant.

**All seven architectures** (`scripts/synth_probe/synth_archs.py` → `synth_archs.csv`; RSNA-test, seed ensemble, Δ vs E1 [95% CI]):

| Architecture | E1 | +250 synthetic (E3) | +500 synthetic (E4) | E3 / E4 on AP films |
|---|---|---|---|---|
| EfficientNet-B0 | 0.880 | −0.044 [−0.049, −0.040] | −0.065 [−0.069, −0.060] | −0.100 / −0.110 |
| ResNet-18 | 0.896 | −0.033 [−0.036, −0.030] | −0.067 [−0.071, −0.063] | −0.106 / **−0.200** |
| Swin-T | 0.867 | −0.042 [−0.046, −0.037] | −0.029 [−0.033, −0.025] | **−0.150** / −0.040 |
| ResNet-50 | 0.885 | −0.030 [−0.034, −0.026] | −0.038 [−0.042, −0.034] | −0.058 / −0.062 |
| ConvNeXt-T | 0.865 | −0.017 [−0.020, −0.013] | −0.042 [−0.046, −0.038] | −0.048 / −0.092 |
| DenseNet-121 | 0.902 | −0.015 [−0.018, −0.012] | −0.010 [−0.014, −0.007] | −0.041 / +0.003 |
| ViT-S | 0.842 | −0.009 [−0.013, −0.005] | −0.023 [−0.027, −0.019] | −0.047 / −0.113 |

**14 of 14 comparisons are negative with the CI below zero.** The size varies (−0.009 to −0.067) but the direction never does;
DenseNet-121 is the most resistant, EfficientNet-B0 and ResNet-18 the most affected. On PA films every one of the 14 is −0.034 or worse.

## 2. The labels carry some pneumonia signal, but much of it is image quality

Classifiers trained only on real images (seed 0) scoring the synthetic set (`label_signal.csv`):

| Model | AUROC on synthetic labels | Mean p, synthetic pneumonia | Mean p, real pneumonia (Kermany val) |
|---|---|---|---|
| EfficientNet E1 / E2 / E2_ERASE | 0.904 / 0.902 / 0.913 | 0.80 / 0.51 / 0.54 | 0.98 / 0.97 / 0.98 |
| DenseNet E1 / E2 / E2_ERASE | 0.858 / 0.867 / 0.749 | 0.79 / 0.76 / 0.28 | 0.98 / 0.98 / 0.98 |
| ViT-S E1 / E2 / E2_ERASE | 0.862 / 0.861 / 0.833 | 0.52 / 0.36 / 0.29 | 0.97 / 0.96 / 0.96 |

The generator learned something class-related (AUROC 0.75–0.91). But the more robust the classifier (E2_ERASE), the less
pneumonia it sees in the synthetic pneumonia images: mean p 0.28–0.54 against 0.96–0.98 for real pneumonia.
Synthetic "pneumonia" is mostly borderline, and part of what makes it look like pneumonia to a naive model is the quality cue below.

## 3. The synthetic images copy the Kermany image-quality shortcut, and amplify contrast

Label predictiveness of simple image statistics at the training resolution (224 px squash), shown as the AUROC of a
*low* value predicting pneumonia (`texture.csv`, 1,500 images per set):

| Set | Sharpness (high-frequency energy) | Laplacian variance | Contrast (pixel SD) |
|---|---|---|---|
| Kermany train (real) | 0.865 | 0.799 | 0.749 |
| **Synthetic** | **0.861** | 0.776 | **0.889** |
| RSNA AP | 0.555 | 0.530 | 0.629 |
| RSNA PA | 0.560 | 0.468 | 0.504 |

In Kermany, pneumonia images are softer and lower-contrast than normals, and that alone predicts the label. The generator
reproduces the softness cue almost exactly and makes the contrast cue stronger. In RSNA these cues are weak. Every synthetic
image is therefore another training example of "soft / low-contrast = pneumonia". Synthetic images are also sharper overall
than any real set (Laplacian variance 5–7 against 2–4.7).

This joins the geometric shortcut found earlier (width alone predicts the Kermany label, AUROC 0.95; pneumonia aspect 1.52 vs 1.23).

## 4. Causal test: degrading healthy images raises the pneumonia score

Same real **normal** images, only quality changed; the table shows the rise in mean pneumonia probability (2 seeds, `degrade.csv`):

| Model | Kermany, blur σ=1 | Kermany, blur σ=2 | RSNA PA, blur σ=1 | RSNA PA, blur σ=2 | RSNA PA, contrast × 0.7 | RSNA PA, clean p | RSNA AP, clean p |
|---|---|---|---|---|---|---|---|
| E1 | +0.33 | +0.50 | +0.38 | +0.61 | +0.21 | 0.28 | 0.76 |
| R25 | +0.32 | +0.47 | +0.46 | +0.67 | +0.26 | 0.30 | 0.73 |
| E2 | +0.09 | +0.56 | +0.32 | +0.68 | +0.06 | 0.14 | 0.54 |
| E2_ERASE | **+0.02** | +0.58 | **+0.22** | **+0.80** | +0.07 | 0.13 | 0.66 |
| E3 | +0.33 | +0.57 | +0.37 | +0.57 | **+0.41** | 0.42 | **0.92** |
| E4 | +0.37 | +0.51 | +0.43 | +0.55 | **+0.37** | 0.44 | **0.91** |

* "Blurry = pneumonia" is learned from the **real** Kermany data: E1 and R25 react as strongly as E3/E4.
* Augmentation only protects against **mild** blur. E2 and E2_ERASE barely react at σ = 1, which is the range of quality
  differences between real scanners. At σ = 2 every model pushes healthy images to p ≈ 0.9, and E2_ERASE rises the most
  (partly because it starts lowest, 0.13 on RSNA PA). So augmentation makes the shortcut less sensitive; it does not remove it.
  That is still a plausible mechanism for why augmented models transfer best (consistent with notebook 06: erasing helps as a
  regulariser, not by moving attention off the border).
* Synthetic training adds a **contrast** shortcut: E3/E4 roughly double their response to reduced contrast.
* E3/E4 already score healthy AP films at 0.92 without any degradation, which is the AP collapse in §1. Their small deltas
  there are a ceiling effect, not robustness.

## 5. Harmonising image quality: what preprocessing can and cannot remove

`cxr/imaging.py` mode `harmonize<F>`: squash → area-downsample to a resolution floor F → global histogram equalisation →
resize to 224. Shortcut AUROC (max(AUROC, 1 − AUROC)) of each statistic after the transform (`harm_check.csv`, 800 images per set):

| Set | Sharpness (hf), squash | Sharpness (hf), equalised (F = 224) | Contrast, squash | Contrast, equalised |
|---|---|---|---|---|
| Kermany | 0.876 | 0.767 | 0.749 | 0.657 |
| Synthetic | 0.893 | 0.704 | 0.881 | 0.570 |
| RSNA AP | 0.540 | 0.527 | 0.541 | 0.678 |

* **Equalisation is the part that works.** It removes most of the synthetic contrast cue. Low-pass filtering does not help:
  floors of 160/128/96 px leave the sharpness cue at least as strong (0.78–0.80 on Kermany).
* **Part of the Kermany sharpness cue is not image quality.** It survives downsampling to 96 px, far below both classes'
  native resolution, so it is content: Kermany pneumonia patients are probably younger and smaller. No filter removes that.
* Side effect: after equalisation, mean intensity predicts opacity in RSNA (0.80 AP, 0.71 PA). Real opacity becomes more
  visible, which may help transfer on its own.
* Deblurring was not tried on purpose: it hallucinates detail and amplifies noise, which would add a new artifact.

**Harmonised training** (`scripts/pipeline_harm.sh`; `harmonize224` applied to training *and* test images; EfficientNet-B0,
RSNA-test, 3 seeds each except standard E4 (2); hyper-parameters are the ones Optuna tuned for `squash`; see the re-tuned correction below).
AUROC of the 3-seed ensemble, Δ with paired patient-clustered bootstrap 95% CI (`compare_btest.csv`):

| Comparison | Standard preprocessing | Harmonised preprocessing |
|---|---|---|
| E1 → E3 (+250 synthetic) | −0.044 [−0.049, −0.040] | −0.010 [−0.015, −0.005] |
| E1 → E4 (+500 synthetic) | −0.065 [−0.069, −0.060] | +0.002 [−0.004, +0.007] |
| E1 → E3, AP films only | −0.100 | −0.066 |

| Baseline | Standard → harmonised (pooled) | AP films |
|---|---|---|
| E1 | 0.880 → 0.866 (−0.014) | 0.747 → 0.763 |
| E2 | 0.930 → 0.916 (−0.014) | 0.881 → 0.835 |
| E2_ERASE | 0.930 → 0.929 (−0.001) | 0.884 → 0.883 |

* **With harmonised images, synthetic data stops hurting overall** (−0.010 and +0.002, against −0.044 / −0.065). This supports
  the shortcut explanation of §1–§4: most of the damage came from the generator's contrast and quality profile, which equalisation removes.
* **It still does not help**, and on AP films synthetic data still costs 0.05–0.07.
* **Harmonising does not improve any baseline** (E1 and E2 −0.014, E2_ERASE ±0) with the squash-tuned hyper-parameters.

**Correction after re-tuning (HARMT).** Optuna re-run under `harmonize224` (`pipeline_harmt.sh`, 25 trials, same space and
Kermany-validation objective; best lr 4.2·10⁻⁴ vs 8.8·10⁻⁴ for squash). RSNA-test, ensemble Δ [95% CI] (`compare_harmt.csv`):

| Comparison | Δ pooled | Δ AP films |
|---|---|---|
| E1 → E1_HARMT (re-tuned harmonised baseline) | **+0.012** [+0.008, +0.016] | +0.028 |
| E1_HARMT → E3_HARMT (+250 synthetic) | **−0.030** [−0.033, −0.027] | −0.101 |
| E1_HARMT → E4_HARMT (+500 synthetic) | **−0.038** [−0.042, −0.035] | −0.101 |
| E2_ERASE → E2_ERASE_HARMT | −0.018 [−0.021, −0.014] | −0.067 |

* **The "harmonising removes the synthetic penalty" result above was largely an artefact of a weak baseline.** With parameters
  tuned for harmonised images, E1 itself improves (0.880 → 0.891), and synthetic data costs −0.030 / −0.038 again, about two thirds
  of the original penalty. Equalisation weakens the synthetic images' harm but does not remove it.
* Harmonised preprocessing helps the un-augmented baseline slightly (+0.012) but hurts the augmented one (E2_ERASE −0.018; note its
  parameters were tuned on E1, without augmentation).

## 6. No memorisation, but a separate domain

Nearest real training image of every synthetic image in ImageNet-EfficientNet-B0 feature space (cosine similarity; `probe.py`, `nn_imagenet.csv`):

| Nearest-neighbour similarity | Median | 99th pct | Max |
|---|---|---|---|
| synthetic → nearest real train image | 0.837 | 0.890 | 0.904 |
| real val → nearest real train image (different patients) | 0.902 | 0.945 | 0.961 |
| real train → nearest real train image (other near-duplicate group) | 0.904 | 0.945 | n/a |
| synthetic → nearest other synthetic image | 0.894 | 0.925 | 0.932 |

* **No near-copies:** no synthetic image is as close to a training image as real images of *different* patients typically are.
* **A domain of its own:** synthetic images resemble each other more than any real image. A linear probe on the same generic features
  separates real from synthetic **perfectly (AUROC 1.000**, 5-fold CV; 0.960 in the CXR-tuned feature space). Within each source the
  classes are about equally separable (normal vs pneumonia: synthetic 0.979, real 0.984).
* **Training set unknown:** nearest neighbours fall in the classifier's 1k subset at chance level (25.3% vs 24.4% base rate; normals
  27% vs 43%), so there is no sign of the *strict* setting. That does not prove *transfer*; it still needs the teammate's answer.
* The 5,000 synthetic images map to only 658 distinct nearest real images, and one real image is the nearest neighbour of 404
  synthetic ones: limited diversity, or a "prototype" image the generator gravitates towards.

## 7. Model selection on RSNA-dev, and the ERASE series

**Split.** RSNA patients are split by a hash of the patient ID (`cxr/analysis.py: b_role`): 30% **RSNA-dev**, used only to choose
between variants, and 70% **RSNA-test**, used for every number reported. The split is computed from the saved predictions, so no
model was re-evaluated. Both halves have the same make-up (opacity 41.5% / 42.3%, AP share 42% / 43.5%).

**Choice.** E2_ERASE has the highest RSNA-dev AUROC of the anti-overfitting variants for every architecture with complete runs:

| Architecture | E1 | E2 | E2_ACQ | E2_ERASE | E2_MIX | E2_ROBUST |
|---|---|---|---|---|---|---|
| EfficientNet-B0 | 0.858 | 0.917 | 0.906 | **0.919** | 0.894 | 0.909 |
| ViT-S | 0.822 | 0.862 | 0.883 | **0.916** | 0.887 | 0.906 |
| DenseNet-121 | 0.883 | 0.915 | 0.884 | **0.922** | 0.920 | 0.892 |
| ResNet-18 | 0.873 | 0.904 | 0.901 | **0.910** | 0.888 | 0.897 |
| ResNet-50 | 0.865 | 0.903 | 0.913 | **0.922** | (running) | (1 seed) |

RSNA-test numbers are within 0.01 of RSNA-dev everywhere, so the choice is not overfitted to the dev half.

**ERASE series** (`scripts/pipeline_erase.sh`; EfficientNet-B0, ViT-S, DenseNet-121 × 3 seeds). E3/E4 and their real
controls re-run on top of E2_ERASE, so the question becomes: *does synthetic data add anything to a strong augmented
baseline?* The names say what is added: `ERASE_SYN250/500` = E2_ERASE augmentation + 250/500 **synthetic** images,
`ERASE_REAL250/500` = E2_ERASE augmentation + 250/500 more **real** images. E1–E4 keep their names.

RSNA-test, 3-seed ensemble AUROC, Δ vs E2_ERASE with 95% CI (`erase_compare.csv`, `scripts/synth_probe/erase_compare.py`):

| Architecture | E2_ERASE | + 250 real | + 500 real | + 250 synthetic | + 500 synthetic | AP films: + 250 / + 500 synthetic |
|---|---|---|---|---|---|---|
| EfficientNet-B0 | 0.930 | −0.004 | +0.010 | **−0.055** [−0.059, −0.050] | **−0.060** [−0.063, −0.056] | **−0.25 / −0.20** |
| ViT-S | 0.927 | −0.002 | +0.006 | **−0.056** [−0.060, −0.053] | **−0.057** [−0.061, −0.054] | −0.15 / −0.15 |
| DenseNet-121 | 0.936 | +0.002 | −0.004 | −0.017 [−0.020, −0.014] | **−0.047** [−0.050, −0.043] | −0.08 / −0.16 |

* **On a strong augmented baseline, synthetic images hurt for all three architectures**; 500 hurt more than 250 only for DenseNet-121. The same number
  of extra *real* images changes nothing (|Δ| ≤ 0.01): the augmented model has saturated what more Kermany-like data can give.
* **Unexpected: augmentation does not shield the model from the synthetic images; the penalty is as large or larger** (EfficientNet-B0
  −0.055 with augmentation vs −0.044 without). On AP films synthetic data takes EfficientNet-B0 from 0.884 to 0.632, *below the
  unaugmented E1* (0.747): it undoes the entire augmentation gain there. Erasing removes local cues, which leaves the global
  quality/contrast cue of the synthetic images (§3) as an even easier shortcut.
* Synthetic vs the same number of real images (ERASE_REAL → ERASE_SYN): −0.050 / −0.069 (EfficientNet-B0), −0.055 / −0.064 (ViT-S),
  −0.019 / −0.043 (DenseNet-121), all CIs below 0. Adding synthetic images is worse than adding nothing.

**Harmonised preprocessing does not rescue the augmented case either; it makes it worse.** Same ERASE comparison with
`harmonize224` (squash-tuned parameters, tag HARM), EfficientNet-B0, RSNA-test, Δ vs E2_ERASE_HARM (0.929):

| Added to E2_ERASE_HARM | Δ pooled [95% CI] | AP films |
|---|---|---|
| +250 synthetic (ERASE_SYN250_HARM) | **−0.065** [−0.069, −0.060] | 0.883 → 0.660 (−0.223) |
| +500 synthetic (ERASE_SYN500_HARM) | **−0.109** [−0.114, −0.103] | 0.883 → 0.634 (−0.249) |

Unlike E3/E4 without augmentation (§5), equalising quality does not weaken the penalty here; it is larger than with squash (−0.055 /
−0.060). With erasing removing local cues and harmonising removing global quality cues, whatever remains distinctive about the
synthetic images is still a stronger shortcut than real pathology.

## 8. Dose-response: does a lot more synthetic data help?

E1's 1,000 real images plus 250 … 2,000 synthetic images (2,000 = all 1,000 synthetic pneumonia images + 1,000 normals, the
largest balanced set), and synthetic images only. No augmentation, EfficientNet-B0, RSNA-test, 3-seed ensemble AUROC, Δ vs E1
(0.880) with 95% CI (`dose_response.csv`; `SYN1000`, `SYN2000`, `SYN_ONLY` in `cxr/experiments.py`):

| Synthetic images added | Seeds | RSNA-test | Δ vs E1 | AP films | PA films |
|---|---|---|---|---|---|
| 250 (E3) | 3 | 0.835 | −0.044 [−0.049, −0.040] | 0.647 | 0.741 |
| 500 (E4) | 3 | 0.815 | −0.065 [−0.069, −0.060] | 0.637 | 0.695 |
| 1,000 (SYN1000) | 3 | 0.837 | −0.042 [−0.047, −0.038] | 0.677 | 0.736 |
| 2,000 (SYN2000) | 3 | 0.826 | −0.053 [−0.058, −0.049] | 0.652 | 0.720 |
| synthetic only, 2,000 (SYN_ONLY) | 3 | **0.659** | −0.220 | 0.546 | **0.496** |
| *E1 (0 synthetic)* | 3 | *0.880* | | *0.747* | *0.791* |

* **More synthetic data does not fix it, and does not make it much worse either.** From 250 to 2,000 images (8×) the penalty stays at
  about −0.04 to −0.065, with no trend (E4, the largest, has one weak seed). The harm is a fixed bias introduced by the synthetic images, not a
  dilution effect: "generate as many as you want" buys nothing here.
* **Synthetic images alone teach almost nothing that transfers.** A model trained only on 2,000 synthetic images reaches 0.88 on the
  Kermany test set (seeds 0.85–0.91) but **0.50 on RSNA PA films, i.e. chance**, and 0.55 on AP films; its pooled 0.66 is mostly
  view recognition. What the generator encodes is Kermany's appearance, not a transferable notion of pneumonia.

**In-domain check (Kermany test, 624 images, ensemble Δ vs E1 0.961):** +250/+500 synthetic −0.043 / −0.055, +1,000/+2,000
−0.037 / −0.020, synthetic only −0.046, but **+500 real images also −0.060** [−0.087, −0.037] (+250 real −0.010). With augmentation
nothing moves (|Δ| ≤ 0.01). The Kermany test set is too small and too different from its own training split to attribute in-domain
changes to synthetic data; it is not used as evidence either way.

## 9. Natural experiment: the shortcut is visible in real RSNA films, no degradation needed

§4 *induced* blur on healthy images. Here nothing is changed: every labelled RSNA film's own sharpness (high-frequency energy),
contrast (central-crop SD), brightness and the patient's age are measured and related to each run's seed-averaged logit
(EfficientNet-B0, RSNA-test, `scripts/synth_probe/natural_quality.py` → `rsna_quality_scores.csv`; analysis →
`rsna_quality_dependence.csv`, `rsna_quality_ratio.csv`, `rsna_auroc_by_sharpness.csv`).

**How valid is the cue externally?** Within RSNA-test, low sharpness predicts opacity only weakly (AUROC 0.57 AP, 0.55 PA; contrast
0.60 / 0.50), against 0.87 in Kermany. So a model that leans on quality is leaning on something nearly useless here.

**Quality weight relative to opacity weight.** Per view, the standardised logit is regressed on label, sharpness, contrast,
brightness and age; the table gives −β(sharpness) / β(label), i.e. how many "units of opacity" one SD of blur is worth
(95% bootstrap CI, 300 resamples):

| Run | AP films | PA films |
|---|---|---|
| E1 (baseline) | 0.36 [0.32, 0.41] | 0.39 [0.35, 0.43] |
| R50 (+500 real) | 0.33 [0.30, 0.36] | 0.45 [0.41, 0.49] |
| E2_ERASE (augmentation) | **0.14** [0.12, 0.16] | **0.20** [0.18, 0.22] |
| E3 (+250 synthetic) | 0.56 [0.47, 0.65] | 0.66 [0.60, 0.72] |
| E4 (+500 synthetic) | **0.91** [0.79, 1.06] | **0.90** [0.81, 1.01] |
| SYN1000 / SYN2000 | 0.86 / 0.84 | 0.67 / 0.46 (contrast 0.56 / **0.72**) |
| ERASE_SYN250 / 500 | 0.58 / 0.80 | 0.35 / 0.59 |
| E1_HARMT / E3_HARMT / E4_HARMT (equalised, re-tuned) | 0.34 / 0.52 / 0.78 | 0.33 / 0.45 / 0.67 |

* **In real external films, synthetic training doubles the weight of blur relative to opacity** (E1 0.36–0.39 → E4 0.90–0.91): one SD of
  blur moves an E4 model as much as the presence of an opacity does. In parallel the label coefficient itself shrinks (AP: E1 0.78 →
  E4 0.46): the model learns less pathology *and* more quality.
* **Augmentation works mainly by learning more pathology, not by ignoring quality.** E2_ERASE's sharpness coefficient is only a
  little smaller than E1's (AP −0.18 vs −0.28), but its label coefficient is much larger (1.27 vs 0.78), which halves the ratio. Adding
  synthetic images on top of augmentation brings the ratio back up to E1 levels or beyond (ERASE_SYN500: 0.80 AP).
* **Equalised preprocessing does not remove sharpness reliance** (E1_HARMT 0.34 ≈ E1 0.36) and the synthetic increase survives it
  (E4_HARMT 0.78). Consistent with §5: equalisation acts on contrast, the sharpness cue is partly content.
* The contrast weight grows with synthetic data on PA films (E1 0.22 → E4 0.65, SYN2000 0.72; E2_ERASE 0.06), matching §3–§4 (the
  generator amplifies the contrast cue).
* LP (frozen ImageNet features) has no quality dependence at all (ρ +0.09 / +0.13), but also little opacity signal: the quality
  reliance is learned during fine-tuning on Kermany.

**Where the synthetic penalty lands.** AUROC within sharpness tertiles of RSNA-test (blurriest third / middle / sharpest third):

| Run | AP blurry | AP mid | AP sharp | PA blurry | PA mid | PA sharp |
|---|---|---|---|---|---|---|
| E1 | 0.721 | 0.741 | 0.794 | 0.771 | 0.792 | 0.806 |
| E2_ERASE | 0.856 | 0.885 | 0.907 | 0.870 | 0.855 | 0.855 |
| E3 (+250 synthetic) | **0.540** | 0.609 | 0.756 | 0.674 | 0.758 | 0.781 |
| E4 (+500 synthetic) | 0.619 | 0.651 | 0.725 | 0.651 | 0.710 | 0.737 |
| SYN2000 | 0.568 | 0.633 | 0.712 | 0.676 | 0.728 | 0.735 |
| ERASE_SYN500 | 0.658 | 0.659 | 0.752 | 0.798 | 0.773 | 0.796 |

* **The synthetic penalty is concentrated in the lowest-quality films**: E1 → E3 costs −0.18 on the blurriest AP third, −0.04 on the
  sharpest. E3 is *at chance-level discrimination (0.54) on blurry AP films*. These are typically portable bedside films of the sickest
  patients, exactly where a pneumonia classifier would be used. Pooled AUROC hides this.
* Opacity prevalence differs little across AP tertiles (0.82 / 0.72 / 0.73) and is low and flat in PA (0.20 / 0.15 / 0.14), so this is
  not a prevalence artefact.

**Age.** Among healthy RSNA films, every fine-tuned model scores older patients higher (ρ(score, age) +0.14 to +0.41), synthetic or
not; age itself barely predicts opacity (AUROC 0.55). A model trained on young children (Kermany) reads adult ageing as disease. This
is a separate, real-data shortcut, not created by the synthetic images.

## 10. Quality-aware selection of synthetic images

If the harm comes from the synthetic images' quality profile, choosing *which* synthetic images to add should change it.
`scripts/synth_probe/quality_select.py` computes sharpness and contrast for all 5,000 synthetic images and writes two re-ranked
manifests (same images, new `synth_rank`):

* **matched** (`synthetic_sd_lora_qm.csv`): greedy nearest-neighbour (pneumonia, normal) pairs in standardised (sharpness, contrast);
  inside the E3/E4 subsets quality carries no label information (AUROC of low sharpness / low contrast 0.500 / 0.500; random
  selection 0.878 / 0.895).
* **counter** (`synthetic_sd_lora_qc.csv`): sharpest, most contrasty pneumonia and softest, flattest normals first, so the cue points
  the other way (AUROC 0.01–0.11).

**Do the selected images still carry pneumonia?** Real-trained classifiers scoring the synthetic images (`label_signal_by_selection.csv`):
random subsets separate at 0.74–0.93, quality-matched still at **0.69–0.88**, counter at 0.65–0.83. So the synthetic classes differ in
more than sharpness and contrast, and the matched subset keeps most of the label signal. E3/E4 with each selection: `scripts/queue_quality.sh`, tags `sd_lora_QM`, `sd_lora_QC`.

**Result** (EfficientNet-B0, RSNA-test, Δ vs E1 0.880, seed ensemble [95% CI]; per-seed pooled AUROC):

| Selection | +250 (E3) | seeds | +500 (E4) | seeds |
|---|---|---|---|---|
| random (original) | −0.044 [−0.049, −0.040] | 0.821 0.827 0.812 | −0.065 [−0.069, −0.060] | 0.718 0.806 0.843 |
| quality-matched | −0.054 [−0.058, −0.050] | 0.781 0.829 0.816 | **−0.014** [−0.018, −0.010] | 0.858 0.856 0.844 |
| quality-counter | **−0.012** [−0.015, −0.009] | 0.847 0.862 0.867 | −0.029 [−0.033, −0.024] | 0.818 0.842 0.848 |

Quality weight relative to opacity weight on real RSNA films (§9 method; AP / PA):

| | sharpness | contrast |
|---|---|---|
| E1 | 0.36 / 0.39 | 0.15 / 0.22 |
| E3 / E4 random | 0.56 / 0.66, 0.91 / 0.90 | 0.03 / **0.46**, 0.10 / **0.65** |
| E3 / E4 matched | 1.19 / 0.76, 0.45 / 0.36 | −0.19 / 0.01, 0.07 / 0.27 |
| E3 / E4 counter | 0.51 / 0.52, 0.56 / 0.35 | 0.17 / 0.29, 0.06 / 0.16 |

* **Choosing the synthetic images by quality removes most of the penalty in three of four cases** (best: E3-counter −0.012, per-seed
  mean 0.859 = E1's 0.860; E4-matched −0.014), and it reliably removes the extra **contrast** reliance on PA films (0.46 / 0.65 → 0.01–0.29).
  This is the most direct evidence so far that the harm comes from the quality profile of the synthetic images, not from synthetic
  images as such.
* **It still never helps.** The best selections only reach E1. Sharpness reliance is not reduced consistently (E3-matched 1.19 on AP,
  with one weak seed 0.781), so the selection fixes the cue the generator *amplifies* (contrast) but not the one it *inherits* from
  Kermany (sharpness), which is also present in the 1,000 real images it is added to.
* Seed spread is large (up to 0.05 within a configuration); the ordering of matched vs counter is not stable across sizes. The robust
  statement is "quality-aware selection ≈ neutral, random selection harmful".

## 11. Anti-overfitting variants across all seven architectures (notebook 06)

RSNA-test, Δ vs E2 (seed ensemble): E2_ERASE −0.000…+0.033 for all seven, best on RSNA-dev for six; **Swin-T's best is E2_ACQ**
(0.911 vs 0.905). E2_ACQ helps modern architectures and hurts classic CNNs (Swin-T +0.030, ConvNeXt-T +0.015, ResNet-50 +0.008 …
DenseNet-121 −0.021). Mixup has the best calibration for all seven. LP is 0.07–0.15 worse except Swin-T (−0.020; frozen Swin
features reach 0.875). Erasing moves Grad-CAM mass away from the image border for 6/7 architectures; EfficientNet-B0, the exception,
is also the one without an AUROC gain from erasing.

## 12. Robust training losses

Can the training objective, rather than the data, make the model ignore the shortcut? Four losses in `cxr/engine.py` (`LOSSES`,
`scripts/run.py --loss`), each on E1 (real only) and E3 (+250 synthetic), EfficientNet-B0, 3 seeds, BCE-tuned hyper-parameters;
validation, checkpointing and early stopping stay plain BCE:

| Loss | Idea | Targets |
|---|---|---|
| `focal` (γ = 2) | down-weight examples already classified correctly | easy shortcut examples |
| `gce` (q = 0.7) | generalised cross-entropy: bounded, robust to label noise | synthetic labels are partly noise (§2) |
| `qdro` (η = 0.01) | online group DRO over label × softer/sharper half of the training images | the groups where "soft = pneumonia" fails |
| `qcons` (λ = 1) | BCE + squared logit change under random blur (σ 0.5–2) and contrast reduction (×0.6–1) | score should not depend on quality (§4, §9) |

**Results** (RSNA-test, seed-ensemble ΔAUROC [95% CI], `compare_losses.csv`; quality ratio = −β_hf / β_label of §9,
`loss_quality_ratio.csv`):

| Loss | E1 → E1 + loss (pooled / AP / PA) | E3 → E3 + loss (pooled / AP) | synthetic penalty under the loss, E1+loss → E3+loss | sharpness ratio AP / PA, E1+loss | E3+loss |
|---|---|---|---|---|---|
| BCE (reference) | — (0.880 / 0.747 / 0.791) | — (0.835 / 0.647) | −0.044 (§1) | 0.36 / 0.39 | 0.56 / 0.66 |
| **qcons** | **+0.031** [0.028, 0.035] / **+0.116** / +0.016 | **+0.050** / **+0.146** | −0.025 [−0.028, −0.022] | **0.12** / 0.32 | 0.36 / 0.47 |
| qdro | +0.009 / +0.018 / +0.010 | +0.036 / +0.083 | −0.016 [−0.020, −0.013] | 0.31 / 0.27 | 0.50 / 0.50 |
| focal | +0.007 / +0.053 / −0.004 | +0.021 / +0.077 | −0.030 [−0.035, −0.026] | 0.33 / 0.54 | **1.07** / 0.67 |
| gce | −0.019 / −0.049 / −0.007 | −0.051 / −0.076 | −0.075 | 0.32 / 0.45 | 0.52 / 0.57 |

* **The quality-consistency loss (qcons) is the only one that works.** On real data alone it gains +0.031 (+0.116 on AP films),
  cuts the sharpness reliance on AP from 0.36 to 0.12 (the level of E2_ERASE, 0.14) and removes the contrast reliance
  (0.15 → 0.03). It is a loss-level form of augmentation: it still trails conventional augmentation (E2 / E2_ERASE 0.930 vs
  0.911, −0.020), which also covers geometry and erasing.
* **No loss makes synthetic images help.** Adding the 250 synthetic images still costs −0.016 to −0.075 under every loss. qcons
  shrinks the penalty from −0.044 to −0.025, and E3 + qcons (0.886) edges past plain E1 (+0.006 [0.003, 0.010]). That gain
  comes from the loss, not the data: it stays below E1 + qcons (0.911), and on PA it is still below plain E1 (−0.015).
* **qdro and focal give small gains** of +0.007 to +0.009 pooled, larger on AP. qdro is seed-unstable (seed 1 is worst on both
  E1 and E3). Focal loss with synthetic data raises the AP sharpness ratio to 1.07, the highest of any run. Down-weighting easy
  examples hands more weight to the hard real AP films that the shortcut explains.
* **gce (q = 0.7) breaks training.** With synthetic data one seed stays at chance (RSNA 0.50) and the ensemble loses 0.095
  against E1. The bounded loss under-fits with BCE-tuned hyper-parameters; it was not re-tuned.
* **qcons on top of augmentation adds nothing**: E2_ERASE → E2_ERASE + qcons Δ 0.000 [−0.002, 0.002] (AP +0.001, PA +0.002).
  Erasing plus intensity augmentation already covers the invariance the loss enforces, so the two are substitutes, not complements.
* **With augmentation, qcons blunts the synthetic penalty but does not remove it.** Adding 250 synthetic images costs
  −0.055 with erasing augmentation alone and −0.016 [−0.018, −0.015] with erasing + qcons (AP −0.252 → −0.044). qcons recovers
  +0.208 on AP for the synthetic run, yet the run stays below real data with augmentation (0.914 vs 0.930). This is the
  strongest synthetic configuration found, and it is still worse than leaving the synthetic images out.
* **qcons generalises across architectures, in proportion to how shortcut-prone the baseline is.** E1 → E1 + qcons on RSNA-test:
  ViT-S **+0.040** [0.036, 0.045] (AP +0.107), EfficientNet-B0 +0.031, DenseNet-121 +0.005 [0.003, 0.008] (AP +0.029). DenseNet-121, the
  architecture least hurt by synthetic data (§1), has the least to gain. For all three it stays below conventional augmentation (vs E2:
  −0.011 to −0.020; vs E2_ERASE: −0.019 to −0.044).
* **Swin-T** (largest AP penalty from synthetic data, −0.150): E1 → E1 + qcons pooled only +0.006 [0.002, 0.010], but **AP +0.079**
  [0.063, 0.093], PA −0.005, i.e. ≈ +0.037 view-balanced. The pooled number hides the gain because it lands on AP films (§13).
  Swin-T is the one architecture where qcons falls far short of augmentation (vs E2_ERASE −0.041, vs E2_ACQ −0.052), consistent with
  Swin-T being the one that benefits most from acquisition augmentation (§11). View-balanced gain across the four architectures:
  ViT-S +0.075, EfficientNet-B0 +0.066, Swin-T +0.037, DenseNet-121 +0.018.
* **Swin-T with synthetic data:** qcons recovers most of its record AP penalty. E3 → E3 + qcons +0.039 [0.035, 0.044], **AP +0.181**
  (0.532 → 0.713), PA +0.032. The synthetic penalty under qcons shrinks from −0.042 to −0.009 [−0.012, −0.005] pooled (AP −0.150 →
  −0.049, PA −0.058 → −0.021; ≈ −0.104 → −0.035 view-balanced), but stays negative: as for EfficientNet-B0, the loss limits the
  damage and the synthetic images still do not help.
* **λ is not critical.** E1 → E1 + qcons: λ 0.3 +0.023 [0.020, 0.026], λ 1 +0.031, λ 3 +0.025 [0.022, 0.029]. The AP sharpness
  weight falls monotonically with λ (0.20 / 0.12 / 0.09), but the AUROC gain plateaus by λ ≈ 1: once the quality dependence is
  small, the remaining error is not quality.
* With augmentation + synthetic data, qcons halves the sharpness weight (ERASE_SYN250 0.58 / 0.35 → 0.25 / 0.29 AP / PA), yet the
  synthetic penalty persists (above). So the remaining penalty is not, or not only, the sharpness cue.
* Hyper-parameters are BCE-tuned for every loss, so all gains are conservative.

## 13. The view confound: AP vs PA in RSNA

**Hypothesis.** In RSNA, view is confounded with the label (normal films are mostly PA, opacity films mostly AP). Pooled AUROC would
then partly reward recognising the view, and since AP films are also blurrier and flatter, the Kermany quality shortcut would be
*rewarded* by the pooled metric. Check: `scripts/synth_probe/view_confound.py` (`scripts/queue_view.sh`) on every run on disk →
`view_facts.csv`, `view_metrics.csv`, `view_compare.csv`, `view_selection.csv`. Within-view and **view-balanced** AUROC (mean of AP
and PA AUROC), RSNA-test, seed ensembles, patient-clustered bootstrap.

**Confirmed, and stronger than expected.**

| RSNA-test | n | opacity | normal | prevalence |
|---|---|---|---|---|
| AP films | 5,164 | 3,898 | 1,266 | **0.755** |
| PA films | 6,698 | 1,117 | 5,581 | **0.167** |

* **View alone scores 0.796 pooled AUROC.** E1 EfficientNet-B0 (0.880 pooled) is only 0.08 above a view detector. Its view-balanced
  AUROC is 0.769; pooled AUROC is inflated by 0.05–0.19 across all runs (E1: 0.09–0.13).
* **The models are, to a large extent, view detectors.** Among *normal* films alone, E1 scores AP above PA with AUROC 0.84–0.92
  (all seven architectures). Decomposing the pooled AUROC by pair type: an AP opacity is ranked above a PA normal 94–98% of the
  time, but a **PA opacity is ranked above an AP normal only 27–37% of the time**. A healthy bedside patient looks sicker to the
  model than a PA patient with pneumonia.
* **Quality and view are tangled, as hypothesised.** Low sharpness predicts opacity at 0.68 pooled but only 0.57 / 0.55 within AP / PA;
  it predicts *AP vs PA* at 0.75. Low contrast: 0.72 pooled, 0.60 / 0.50 within view, 0.83 for view. So most of the pooled label
  signal of the Kermany quality cue in RSNA is the view, and the pooled metric partly rewards the shortcut.

**Consequence: pooled AUROC under-states every effect in this study** (Δ pooled → Δ view-balanced [95% CI]):

| Comparison (EfficientNet-B0 unless noted) | pooled | AP | PA | **balanced** |
|---|---|---|---|---|
| E1 → E3 (+250 synthetic) | −0.044 | −0.100 | −0.050 | **−0.075** [−0.085, −0.066] |
| E1 → E4 (+500 synthetic) | −0.065 | −0.110 | −0.095 | **−0.103** [−0.112, −0.093] |
| R25 → E3 / R50 → E4 (synthetic vs same number of real) | −0.053 / −0.065 | | | **−0.102 / −0.120** |
| E1 → R25 / R50 (+ real) | +0.009 / −0.000 | +0.056 / +0.045 | −0.002 / −0.010 | **+0.027 / +0.017** |
| E2_ERASE → ERASE_SYN250 / 500 | −0.055 / −0.060 | −0.252 / −0.195 | −0.038 / −0.072 | **−0.145 / −0.134** |
| E2_ERASE → ERASE_REAL500 | +0.010 | +0.020 | +0.023 | +0.022 |
| E1 → E2_ERASE (all 7 architectures) | +0.024…+0.085 | | | **+0.049…+0.150** |
| E1 → E3_QC / E4_QM (best quality-aware selection, §10) | −0.012 / −0.014 | | | **−0.044 / −0.042** |
| E1 → E1 + qcons | +0.031 | +0.116 | +0.016 | **+0.066** [0.059, 0.074] |
| E1 → E3 + qcons | +0.006 | | | +0.016 [0.008, 0.024] |
| E1 → E3 / E4, 7 architectures | −0.009…−0.067 | | | **−0.019…−0.148** (14/14 < 0) |

* **No conclusion flips; most get sharper.** Ranking of all 121 run × architecture configurations: Spearman(pooled, balanced) 0.985.
  Of 48 key comparisons, one changes sign significantly, and it strengthens the story: **+500 real images help** under the balanced
  metric (E1 → R50 +0.017 [0.010, 0.024], pooled −0.000). Real data helps, synthetic data of the same amount hurts by 0.10–0.12.
* **Two earlier statements need softening.** (1) Quality-aware selection (§10) "removes most of the penalty" holds pooled but not
  balanced: it roughly halves it (E3 −0.075 → −0.044, E4 −0.103 → −0.042). (2) The synthetic penalty *with augmentation* is the
  largest of all (−0.145), not smaller than without.
* **Selection on RSNA-dev is unaffected**: Spearman(dev pooled, dev balanced) 0.978, identical top-10 order (ERASE_REAL500, E2,
  E2_ERASE, …). The pre-registered choices stand.
* **Precision is fine.** Each view's AUROC has bootstrap SD ≈ 0.007–0.008 (pooled 0.003); AP rests on 1,266 normal films, enough for
  every effect above. AP normals are an unusual population (bedside patients sick with something else), which is closer to clinical use.
* Recommendation for the report: make **view-balanced AUROC the headline external metric**, with AP / PA separately, and pooled as a
  secondary column.

## Pending

* **Possible pivot** (idea only, see `docs/RESEARCH_PIVOT.md`): separate *why* synthetic images hurt (amplified shortcut, weak
  class signal, synthetic domain / anatomy) with real-image mimic arms and an anatomy filter.

* From the teammate: strict vs transfer generator training set, LoRA steps, filtering.

## Reproduce

All probes run on the GPU box on CPU (they don't compete with training), outputs in `reports/results/synth_probe/`:

```bash
python scripts/synth_probe/probe.py 6      # §2 (label signal) and provenance
python scripts/synth_probe/texture.py      # §3
python scripts/synth_probe/degrade.py      # §4 (needs the E3/E4_sd_lora checkpoints)
python scripts/synth_probe/harm_check.py   # §5
python scripts/synth_probe/natural_quality.py && python scripts/synth_probe/loss_compare.py   # §9, §12
python scripts/synth_probe/view_confound.py   # §13
```
