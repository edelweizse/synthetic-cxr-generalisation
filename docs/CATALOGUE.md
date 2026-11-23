# Experiment catalogue

Everything that was run in this project, in one place: what each experiment family tested, why, what came out, and where
the code and numbers live. State as of 2026-11-12. Narrative and reasoning live in the older docs, and this file points into
them: `docs/report.md` (study 1 report), `docs/FINDINGS_SYNTHETIC.md` (study 1 synthetic-data findings §1–13),
`docs/RESEARCH_PIVOT.md` (mechanism arms, study 2, generator rounds §1–13), `docs/EXPERIMENTS.md` (study 1 inventory
as of 10-13), `HANDOFF.md` (session log).

The run tables at the end (§12) are **generated** from the run directories and MLflow by `scripts/catalogue.py`:
- `reports/results/catalogue_runs.csv` has one row per run: config, MLflow metrics, RSNA-test AP / PA / balanced / pooled, CheXpert test, Kermany test, git commit and start time.
- `reports/results/catalogue_summary.csv` has one row per group × architecture, with mean / SD over seeds.

## 0. Common protocol

| | |
|---|---|
| Task | binary: pneumonia / lung opacity (1) vs normal (0), frontal chest X-ray, 224 px, full field of view squashed to square |
| Sources (training) | **Kermany** (study 1): paediatric, one hospital in Guangzhou, 5,856 images. **CheXpert** (study 2): adult, Stanford, 11,969 frontal images downloaded (one per patient; positive = Lung Opacity / Pneumonia / Consolidation, negative = No Finding) |
| External test | **RSNA** Pneumonia Challenge (adults, NIH): Lung Opacity vs Normal. Split by patient hash into RSNA-dev (30 %, used once, to pick the ERASE base) and **RSNA-test** (70 %, reporting only) |
| Headline metric | **view-balanced RSNA-test AUROC** = mean of AUROC on AP films and on PA films. In RSNA, view alone predicts the label (76 % of AP films vs 17 % of PA films show opacity, so view alone scores 0.80 pooled) |
| Other metrics | pooled RSNA AUROC, Kermany test, CheXpert test (`C_test`, 750 + 750), sensitivity / specificity at a Youden threshold fixed on source validation, patient bootstrap CIs |
| Training | timm ImageNet backbones; HPO parameters from Optuna on Kermany E1 (not re-tuned for CheXpert); 1,500-step budget, eval every 50, early stopping on source-validation BCE (patience 8); bf16; 3 seeds (0, 1, 2) |
| Model selection | checkpoint and threshold on source validation only (BCE, because Kermany validation AUROC is ≈0.99 for everything) |
| Noise rule | bootstrap CIs are tight (±0.01), seed SD is 0.01–0.10, so **|Δ| < 0.03–0.04 is treated as inconclusive** |
| Comparisons in the docs | usually **Δ of the 3-seed ensemble** vs a reference. The tables in §12 are **per-seed means**, which run a little lower than ensembles (e.g. Kermany E1 EfficientNet-B0: per-seed mean 0.736 vs ensemble 0.770) |
| Tracking | MLflow `mlflow.db`: experiment `cxr-pneumonia` (1,073 finished classifier runs + 2 interrupted), `cxr-hpo` (200 Optuna trials). Runs in `artifacts/runs/<group>/<arch>/seed<k>/` (`run.json`, `metrics.json`, `preds_*.csv`, `best.pt`) |
| Hardware | one RTX 3060 Ti (8 GB); about 2–5 min per classifier run; LoRA training 45 min (v2) to about 4 h (big) |

Run-group naming: `<prefix><experiment>[_<tag>]`. The prefix is empty for Kermany, `CX_` for CheXpert and `CXVB_` for
CheXpert with view-balanced training subsets. The experiment is defined in `cxr/experiments.py`. The tag names a synthetic
set, preprocessing (`HARM`, `HARMT`) or a loss (`L…`). For SYN_ONLY a trailing `V` means synthetic validation (no real
image anywhere in the classifier pipeline).

## 1. Study 1 setup: Kermany data and the acquisition shortcut

| What | Result | Where |
|---|---|---|
| Patient-level splits, removal of the 170-patient train/test leak in Kermany, near-duplicate grouping | Kermany test kept as the official 624 images; 682 validation images | notebook 01, report §2 |
| Metadata-only classifier (width, height, aspect ratio, file size) | **AUROC 0.998 on Kermany validation**, 0.905 on Kermany test; width alone 0.95. This explains why validation AUROC is ≈0.99 from step 50 | report §2.1 |
| View confound in RSNA | view alone 0.796 pooled; the headline was switched to view-balanced | FINDINGS §13 |

## 2. Study 1: baselines, hyper-parameters, conventional augmentation

| Group | What | Archs × seeds | Result (view-balanced RSNA-test) | Where |
|---|---|---|---|---|
| HPO (`cxr-hpo`) | Optuna TPE + median pruning, 25 trials per arch, objective = Kermany val BCE | 7 archs | `configs/hpo/*.json`; picked EfficientNet-B0 + ViT-S as the main archs | notebook 02 |
| `E1` | 1,000 real Kermany images (500 / class), no augmentation | 7 × 3 | 0.69–0.79 (Kermany test 0.87–0.95) | notebook 03, report §4.2 |
| `E2` | E1 + conventional augmentation (±10° rotation, shift, scale, brightness / contrast / gamma) | 7 × 3 | +0.05 to +0.10 over E1 for every arch | report §4.3 |
| `E2_ERASE` | E1 + random erasing / border occlusion | 7 × 3 | **best Kermany recipe: 0.81–0.86**; became the teacher and the base of the ERASE series | notebook 06, report §4.7 |
| `E2_ACQ`, `E2_ROBUST`, `E2_MIX`, `LP` | acquisition jitter; jitter + erasing; mixup 0.4; linear probe (frozen backbone) | 7 × 3 | ACQ / ROBUST ≈ E2; MIX gives the best calibration; **LP is worst** (generic ImageNet features don't transfer) | notebook 06 |
| `E1_HARM`, `E2_HARM`, `E2_ERASE_HARM`, `*_HARMT` | histogram-equalised preprocessing (`harmonize224`), untuned / re-tuned HPO | EffNet × 3 | equalisation does not help any baseline | FINDINGS §5 |
| Losses `E1_L*` | focal, GCE, group DRO (`QDRO`), quality-consistency loss (`QCONS`, λ 0.3 / 1 / 3) | 1–7 archs | qcons ≈ augmentation (+0.03 to +0.07), not above it; focal / DRO small; GCE breaks | FINDINGS §12 |

## 3. Study 1: more real data

| Group | What | Result | Where |
|---|---|---|---|
| `R25`, `R50` | 1,250 / 1,500 real images: matched controls for E3 / E4 | neutral to slightly positive (+0.00 to +0.03) | report §4.4 |
| `N250`, `N500`, `N2000`, `FULL`, `FULL_AUG` | scaling curve to all 4,093 training images | EfficientNet-B0: 0.74 (E1) → 0.79 (FULL, 4× the data); FULL_AUG 0.84. Variation, not size, limits Kermany | report §4.4 |
| `ERASE_REAL250 / 500` | extra real images on top of E2_ERASE | ≈ 0 | FINDINGS §7 |

## 4. Study 1: synthetic images from the Kermany generator (teammate's SD-1.5 LoRA, tag `sd_lora`)

Generator: SD-1.5 + LoRA r16 on Kermany, one fixed prompt per class, CFG 4.5, 30 steps; 4,000 normal + 1,000 pneumonia
images, 512 px (`docs/SYNTHETIC_DATA.md`). KID×1000 124, recall 0.14, teacher p(pneumonia) on synthetic pneumonia 0.37
(real 0.96). Linear probe real vs synthetic: AUROC 1.000.

| Group | What | Result | Where |
|---|---|---|---|
| `E3_sd_lora`, `E4_sd_lora` | E1 + 250 / 500 synthetic | **negative for all 7 archs** (14 / 14 comparisons; −0.02 to −0.15 balanced vs E1); worst on AP films | FINDINGS §1, §13 |
| `SYN1000`, `SYN2000`, `SYN_ONLY_sd_lora` | dose: +1,000 / +2,000; synthetic only (2,000) | penalty plateaus at −0.04 to −0.05; **synthetic-only 0.525 (chance)**, chance on PA films | FINDINGS §8 |
| `ERASE_SYN250 / 500` (+ `_HARM`, `_LQCONS`) | synthetic on top of the best recipe | −0.02 to −0.06 vs E2_ERASE for every arch | FINDINGS §7 |
| `E3/E4_sd_lora_HARM / HARMT` | synthetic with equalised preprocessing | penalty persists once re-tuned (−0.03 / −0.04); the earlier "harmonising removes it" claim was withdrawn | FINDINGS §5 |
| `E3/E4_sd_lora_QM / QC` | quality-matched / quality-reversed selection | halves the penalty pooled, about half remains view-balanced; never helps | FINDINGS §10 |
| `E3_sd_lora_L*` | robust losses with synthetic data | qcons shrinks the penalty (EffNet −0.025), still negative | FINDINGS §12 |

## 5. Study 1: why do synthetic images hurt? (mechanism arms, `docs/RESEARCH_PIVOT.md` §3–7)

Each arm adds 250 / 500 images to E1. EfficientNet-B0, ResNet-18, DenseNet-121 × 3 seeds. "Share" = how much of the
synthetic penalty the arm reproduces (0 = like real images, 1 = like synthetic).

| Group | What it copies with *real* images | Median share | Verdict |
|---|---|---|---|
| `E3/E4_MIMIC_Q` | the synthetic set's sharpness / contrast shortcut | 0.08 | not the cause |
| `E3/E4_MIMIC_N`, `_N50`, `_N70` | 30 / 50 / 70 % flipped pneumonia labels (weak class signal) | 0.10 → ≈0.25 | a minority (≤ 0.05 AUROC), except DenseNet-121, whose penalty is small anyway |
| `E3/E4_MIMIC_QN*` | both | 0.16 → ≈0.25 | not enough |
| `E3/E4_sd_lora_ANAT / ANATBAD` | synthetic, worst 20 % anatomy removed / worst first (PSPNet anatomy score) | 0.99 / 1.02 | gross anatomy is not the cause |
| `E3/E4_CF_S10 / S08` | **counterfactual pairs**: the same real image with the lungs inpainted once per prompt (strength 1.0 / 0.8) | — | hurt too; the near-identical pairs (0.8) hurt most (−0.10 to −0.16): the generator's own lung texture becomes the class cue |

## 6. Study 1: making synthetic images help (`docs/RESEARCH_PIVOT.md` §8)

| Group | What | Result (Δ balanced vs E1, ensemble) |
|---|---|---|
| `E3/E4_TEACH` | the same synthetic images, label = soft probability of a robust real-trained teacher (E2_ERASE ensemble) | **removes the penalty in 6 / 6 cells** (+0.08 to +0.13 over prompt labels) |
| `E3/E4_RTEACH` | teacher labels on extra *real* images (distillation control) | ≈ TEACH for EffNet / ResNet; DenseNet TEACH is +0.04 above it (edge of noise). The gain is mostly the teacher's |
| `E3/E4_CONF` | prompt labels, only images the teacher agrees with | still negative. Choosing images doesn't fix it, relabelling does |
| `E3/E4_CTX` | E1's real lungs, everything outside them redrawn by the generator | inconclusive (+0.01 / mixed / negative) |
| `E3/E4_SRC_S10 / S08` | CF pairs labelled with the source image's label | confounded (only the real context carries the label); mixed |

Second external test: Kermany-trained runs scored on CheXpert test replicate the pattern (prompt labels −0.01 to −0.04, teacher labels +0.04 / +0.05 for EffNet / DenseNet).

## 7. Study 2: CheXpert as the source (`docs/RESEARCH_PIVOT.md` §9)

| Group | What | Result |
|---|---|---|
| `CX_E1`, `CXVB_E1` | 1,000 CheXpert images (natural / view-balanced subsets) | **0.89–0.92 per-seed (ensemble 0.92–0.94): +0.12 to +0.16 over Kermany E1**. The source dataset was the bottleneck |
| `CX_E2_ERASE`, `CX_R25 / R50`, `CX_N250 / N500 / N2000`, `CX_FULL`, `CXVB_*` | augmentation, more real data, the full 9,968 | plateau: 250 → 2,000 real images 0.87 → 0.90; all 9,968 (= the generator's training set) **0.92–0.95**; augmentation within noise. `CX_E2_ERASE` ViT-S 0.934 is the best single classifier and a teacher member |

## 8. Study 2: our own generator (SD-1.5 LoRA retrained on CheXpert; `docs/RESEARCH_PIVOT.md` §9–13)

Main test (the restated goal, §10): **replace real data**, i.e. train on 2,000 synthetic images only and compare with
`CX_N2000` (2,000 real: 0.900). Generator metrics: `scripts/generator/evaluate.py` (KID / precision / recall on ImageNet
features, teacher signal, memorisation = nearest real training image, cosine) → `reports/results/generator/eval_*.csv`.

| Version (tag) | Recipe change | KID×1000 | recall | teacher p(pneu) on synthetic positives | Synthetic-only RSNA bal. (mean of 3 archs) | CheXpert test |
|---|---|---|---|---|---|---|
| v1 (`SD`) | the teammate's recipe as received: lr 2e-5, 2,500 steps, CFG 4.5 | 122 | 0.06 | 0.45 | 0.714 | 0.741 |
| v2 (`SD2`, `SD2V`) | lr 1e-4, 7,500 steps (3 epochs); V = synthetic validation | 213 | 0.35 | 0.13 | 0.806 / 0.833 | 0.799 / 0.776 |
| **v2 CFG 2.5 (`SD2C25V`) = final** | guidance sweep 1.5 / 2.5 / 3.5 / 4.5, 2.5 dominates | 103 | 0.62 | 0.37 | **0.833** | **0.807** |
| ↳ scaling (`SYN_ONLY5K / 10K`) | 5,000 / 10,000 synthetic images | 106 (10k) | 0.63 | — | 0.800 / 0.798 | 0.797 / 0.806 |
| ↳ mixed (`N250_SYN2000`, `SYN2000`) | 250 / 1,000 real + 2,000 synthetic, prompt labels | — | — | — | −0.016 / −0.018 vs real alone | −0.016 / −0.012 |
| v3 (`SD3V`) | rich report captions (view, sex, age, every finding) | 79 | — | 0.38 | 0.767 | 0.774 |
| rank 64 (`SDR64V`) | LoRA capacity ×4 | — | — | 0.42 | 0.764 | 0.794 |
| view captions (`SDVIEWV`) | ", AP / PA view" in the caption, 50/50 sampling | — | — | 0.33 | 0.814 | 0.809 |
| obvious cases (`SDOBVV`) | 29k images, positives = Consolidation / Pneumonia only, 15,000 steps | 64 | 0.63 | 0.31 | 0.744 | 0.761 |
| big (`SDBIGV`) | 29.5k images, v2 class definition, 15,000 steps | 82 | 0.57 | 0.26 | 0.756 | 0.769 |
| teacher labels (`TEACH`, `TEACHV2`) | v1 / final images, soft teacher labels, real val | — | — | — | 0.920 / 0.925 | 0.885 / 0.893 |

Constant across versions: **no memorisation** (no synthetic image is closer to a training image than two real patients
are, checked up to 10,000 samples). **Realism and usefulness move apart**: every change after v2 improved KID, and none
improved the classifier. Teacher p(pneumonia) on synthetic positives (0.26–0.42 vs 0.74 on real positives) is the metric
that tracks downstream usefulness. The generator under-draws disease. The VAE round trip keeps the signal (teacher AUROC
0.902 → 0.892), so the loss happens in the denoiser / conditioning.

Comparison with Ktena et al. 2024 (`scripts/generator/mixed_fairness.py`): adding synthetic images to 1,000 real
images gives 0.881–0.896 vs 0.898 for real alone. Teacher labels on real images (0.912 / 0.917) beat teacher labels on
synthetic (0.885 / 0.910). The sex AUROC gap is 0.003–0.011 in every arm, within noise, so there is no fairness gap to close.

## 9. Final controls (2026-11-12, `scripts/queue_final.sh`, `scripts/generator/teach_final.py`)

Question: is the 0.92 of synthetic-only + teacher labels (§8 last row) the images or the teacher? All arms are SYN_ONLY
with 2,000 images, real CheXpert validation and the same teacher. Result in `docs/RESEARCH_PIVOT.md` §13.y.

| Group | Training images | Labels | RSNA-test bal. (mean of 3 archs) | CheXpert test |
|---|---|---|---|---|
| `CX_N2000` (reference) | 2,000 real | hard | 0.900 | 0.884 |
| `CX_SYN_ONLY_RTEACH` | 2,000 real, never seen by the teacher (ranks 1,000–1,999) | teacher p | **0.940** | **0.908** |
| `CX_SYN_ONLY_TEACH` | 2,000 synthetic, v1 | teacher p | 0.920 | 0.885 |
| `CX_SYN_ONLY_TEACHV2` | 2,000 synthetic, final v2 CFG 2.5 | teacher p | 0.925 | 0.893 |
| `CX_SYN_ONLY_SD2C25` | 2,000 synthetic, final v2 CFG 2.5 | prompt | 0.839 | 0.815 |
| `CX_FULL` (reference) | all 9,968 real | hard | 0.930 | 0.909 |

Verdict: the gain belongs to the teacher. Real images with teacher labels beat synthetic images with teacher labels for every
architecture (−0.009 to −0.023, seed SD ≤ 0.004), and even beat all 9,968 real images with hard labels. Generator quality
barely matters under teacher labels (v1 → v2: +0.005), while under prompt labels it is worth +0.12. Synthetic validation
was not the problem either (SD2C25 0.839 with real validation vs SD2C25V 0.833).

## 10. Analyses without training (scripts in `scripts/synth_probe/`, `scripts/synth_cf/`, `scripts/generator/`)

| Script | Question |
|---|---|
| `probe.py`, `texture.py` | label signal of synthetic images by real-trained classifiers; sharpness / contrast cues (FINDINGS §2–3) |
| `degrade.py`, `natural_quality.py` | causal test: blurring healthy images raises the pneumonia score; the shortcut in real RSNA films (§4, §9) |
| `harm_check.py`, `view_confound.py` | effect of equalisation on the cues; the AP/PA confound (§5, §13) |
| `anatomy.py`, `anatomy_select.py` | PSPNet anatomy score of synthetic vs real (pivot arm C) |
| `cf_check.py` | do counterfactual edits look like pneumonia to a robust classifier? |
| `evaluate.py`, `vae_check.py`, `mixed_fairness.py` | generator metrics + memorisation; VAE bottleneck; Google-style mixing and fairness |
| `*_compare.py`, `syn_only_table.py`, `headline_tables.py` | paired seed-ensemble Δ tables in `reports/results/` |

## 11. Not run, abandoned, or superseded

| Item | Why |
|---|---|
| `E3/E4_PLACEHOLDER`, `E1_baseline` | pipeline tests before the synthetic images arrived; excluded |
| "harmonising removes the synthetic penalty" | came from an untuned baseline; withdrawn after HARMT |
| GCE / qcons re-tuning | GCE collapses with BCE-tuned parameters; not worth the GPU time |
| GAN generator | its motivation (anatomy) was ruled out by ANAT / ANATBAD |
| DiffChest / RoentGen / RadEdit as generators | the project's subject is *our own* generator; RadEdit was trained on ChestX-ray8, which contains RSNA |
| Real classifier on the 29.5k "big" set | fair ceiling for SDBIGV; unnecessary because SDBIGV is below its 9,968-image predecessor |
| Full UNet fine-tune, pixel-space DDPM from scratch (the Google setup) | 8 GB GPU; 201k labelled images would be needed |

## 12. All runs (generated)

<!-- generated by scripts/catalogue.py; do not edit by hand -->
1073 runs in 121 groups. Cell = view-balanced RSNA-test AUROC, mean ±SD over seeds (seed count in brackets when not 3).

### Study 1: Kermany source

| group | n train | EffNet-B0 | RN-18 | RN-50 | DN-121 | ConvNeXt-T | ViT-S | Swin-T |
|---|---|---|---|---|---|---|---|---|
| `E1` | 1000 | 0.736 ±0.006 | 0.771 ±0.045 | 0.747 ±0.024 | 0.787 ±0.019 | 0.713 ±0.037 | 0.692 ±0.019 | 0.698 ±0.016 |
| `E2` | 1000 | 0.833 ±0.024 | 0.818 ±0.019 | 0.825 ±0.036 | 0.836 ±0.025 | 0.808 ±0.015 | 0.775 ±0.086 | 0.745 ±0.060 |
| `N250` | 250 | 0.724 ±0.010 |  |  |  |  | 0.633 ±0.056 |  |
| `N500` | 500 | 0.731 ±0.017 |  |  |  |  | 0.675 ±0.085 |  |
| `N2000` | 2000 | 0.768 ±0.014 |  |  |  |  | 0.725 ±0.052 |  |
| `FULL` | 4093 | 0.785 ±0.005 |  |  |  |  | 0.782 ±0.020 |  |
| `FULL_AUG` | 4093 | 0.836 ±0.016 |  |  |  |  | 0.839 ±0.012 |  |
| `E3_PLACEHOLDER` | 1250 | 0.787 (1) |  |  |  |  |  |  |
| `E4_PLACEHOLDER` | 1500 | 0.707 (1) |  |  |  |  |  |  |
| `E1_HARM` | 1000 | 0.741 ±0.067 |  |  |  |  |  |  |
| `E3_sd_lora_HARM` | 1250 | 0.694 ±0.044 |  |  |  |  |  |  |
| `E4_sd_lora_HARM` | 1500 | 0.703 ±0.050 |  |  |  |  |  |  |
| `E2_HARM` | 1000 | 0.812 ±0.029 |  |  |  |  |  |  |
| `E2_ERASE_HARM` | 1000 | 0.820 ±0.057 |  |  |  |  |  |  |
| `ERASE_REAL500` | 1500 | 0.867 ±0.016 |  |  | 0.863 ±0.022 |  | 0.866 ±0.011 |  |
| `ERASE_SYN500_sd_lora` | 1500 | 0.704 ±0.051 |  |  | 0.750 ±0.019 |  | 0.729 ±0.025 |  |
| `E1_HARMT` | 1000 | 0.766 ±0.039 |  |  |  |  |  |  |
| `E2_ERASE_HARMT` | 1000 | 0.802 ±0.019 |  |  |  |  |  |  |
| `E3_sd_lora_HARMT` | 1250 | 0.712 ±0.031 |  |  |  |  |  |  |
| `E4_sd_lora_HARMT` | 1500 | 0.693 ±0.042 |  |  |  |  |  |  |
| `E2_ACQ` | 1000 | 0.826 ±0.018 | 0.822 ±0.026 | 0.849 ±0.007 | 0.788 ±0.065 | 0.844 ±0.004 | 0.789 ±0.016 | 0.828 ±0.009 |
| `E2_ERASE` | 1000 | 0.851 ±0.003 | 0.826 ±0.011 | 0.863 ±0.024 | 0.858 ±0.013 | 0.841 ±0.024 | 0.849 ±0.016 | 0.813 ±0.014 |
| `E2_ROBUST` | 1000 | 0.822 ±0.031 | 0.799 ±0.046 | 0.833 ±0.021 | 0.797 ±0.052 | 0.840 ±0.036 | 0.828 ±0.022 | 0.809 ±0.020 |
| `E2_MIX` | 1000 | 0.805 ±0.014 | 0.793 ±0.012 | 0.801 ±0.029 | 0.855 ±0.005 | 0.814 ±0.038 | 0.789 ±0.015 | 0.768 ±0.052 |
| `SYN1000_sd_lora` | 2000 | 0.690 ±0.014 |  |  |  |  |  |  |
| `SYN2000_sd_lora` | 3000 | 0.667 ±0.013 |  |  |  |  |  |  |
| `SYN_ONLY_sd_lora` | 2000 | 0.525 ±0.021 |  |  |  |  |  |  |
| `ERASE_SYN250_sd_lora_HARM` | 1250 | 0.704 ±0.057 |  |  |  |  |  |  |
| `ERASE_SYN500_sd_lora_HARM` | 1500 | 0.659 ±0.031 |  |  |  |  |  |  |
| `LP` | 1000 | 0.654 ±0.004 | 0.710 ±0.008 | 0.692 ±0.009 | 0.768 ±0.007 | 0.682 ±0.011 | 0.677 ±0.014 | 0.759 ±0.007 |
| `E3_sd_lora` | 1250 | 0.676 ±0.018 | 0.693 ±0.029 | 0.692 ±0.060 | 0.741 ±0.064 | 0.686 ±0.069 | 0.634 ±0.067 | 0.596 ±0.067 |
| `E4_sd_lora` | 1500 | 0.650 ±0.043 | 0.640 ±0.004 | 0.680 ±0.019 | 0.767 ±0.044 | 0.650 ±0.034 | 0.619 ±0.022 | 0.634 ±0.065 |
| `E3_sd_lora_QM` | 1250 | 0.645 ±0.031 |  |  |  |  |  |  |
| `E4_sd_lora_QM` | 1500 | 0.702 ±0.017 |  |  |  |  |  |  |
| `E3_sd_lora_QC` | 1250 | 0.713 ±0.031 |  |  |  |  |  |  |
| `E4_sd_lora_QC` | 1500 | 0.685 ±0.033 |  |  |  |  |  |  |
| `E1_LQDRO` | 1000 | 0.758 ±0.020 |  |  |  |  |  |  |
| `E3_sd_lora_LQDRO` | 1250 | 0.698 ±0.072 |  |  |  |  |  |  |
| `E1_LGCE` | 1000 | 0.696 ±0.026 |  |  |  |  |  |  |
| `E3_sd_lora_LGCE` | 1250 | 0.564 ±0.076 |  |  |  |  |  |  |
| `E1_LFOCAL` | 1000 | 0.760 ±0.021 |  |  |  |  |  |  |
| `E3_sd_lora_LFOCAL` | 1250 | 0.716 ±0.017 |  |  |  |  |  |  |
| `E2_ERASE_LQCONS` | 1000 | 0.863 ±0.008 |  |  |  |  |  |  |
| `ERASE_SYN250_sd_lora_LQCONS` | 1250 | 0.828 ±0.014 |  |  |  |  |  |  |
| `E1_LQCONS03` | 1000 | 0.809 ±0.003 |  |  |  |  |  |  |
| `E1_LQCONS3` | 1000 | 0.811 ±0.035 |  |  |  |  |  |  |
| `R25` | 1250 | 0.763 ±0.031 | 0.759 ±0.016 | 0.754 ±0.012 | 0.784 ±0.032 | 0.715 ±0.042 | 0.733 ±0.053 | 0.746 ±0.032 |
| `R50` | 1500 | 0.763 ±0.046 | 0.769 ±0.026 | 0.774 ±0.012 | 0.797 ±0.038 | 0.708 ±0.055 | 0.747 ±0.028 | 0.642 ±0.021 |
| `ERASE_REAL250` | 1250 | 0.834 ±0.019 | 0.836 ±0.002 | 0.841 ±0.019 | 0.854 ±0.042 | 0.862 ±0.034 | 0.850 ±0.010 | 0.808 ±0.015 |
| `ERASE_SYN250_sd_lora` | 1250 | 0.719 ±0.038 | 0.774 ±0.032 | 0.791 ±0.016 | 0.788 ±0.033 | 0.810 ±0.051 | 0.739 ±0.040 | 0.788 ±0.073 |
| `E1_LQCONS` | 1000 | 0.822 ±0.021 | 0.786 ±0.018 | 0.806 ±0.015 | 0.806 ±0.007 | 0.782 ±0.037 | 0.788 ±0.006 | 0.760 ±0.016 |
| `E3_sd_lora_LQCONS` | 1250 | 0.774 ±0.033 | 0.771 ±0.013 | 0.782 ±0.015 | 0.796 ±0.030 | 0.751 ±0.020 | 0.750 ±0.026 | 0.720 ±0.043 |
| `E3_MIMIC_Q` | 1250 | 0.772 ±0.023 | 0.763 ±0.020 |  | 0.767 ±0.076 |  |  |  |
| `E4_MIMIC_Q` | 1500 | 0.756 ±0.029 | 0.770 ±0.026 |  | 0.777 ±0.026 |  |  |  |
| `E3_MIMIC_N` | 1250 | 0.771 ±0.011 | 0.738 ±0.022 |  | 0.771 ±0.017 |  |  |  |
| `E4_MIMIC_N` | 1500 | 0.804 ±0.038 | 0.761 ±0.026 |  | 0.753 ±0.011 |  |  |  |
| `E3_MIMIC_QN` | 1250 | 0.786 ±0.007 | 0.744 ±0.044 |  | 0.799 ±0.045 |  |  |  |
| `E4_MIMIC_QN` | 1500 | 0.754 ±0.035 | 0.709 ±0.022 |  | 0.755 ±0.046 |  |  |  |
| `E3_sd_lora_ANAT` | 1250 | 0.703 ±0.024 | 0.665 ±0.027 |  | 0.737 ±0.043 |  |  |  |
| `E4_sd_lora_ANAT` | 1500 | 0.660 ±0.049 | 0.661 ±0.023 |  | 0.678 ±0.032 |  |  |  |
| `E3_sd_lora_ANATBAD` | 1250 | 0.701 ±0.072 | 0.687 ±0.022 |  | 0.764 ±0.028 |  |  |  |
| `E4_sd_lora_ANATBAD` | 1500 | 0.648 ±0.024 | 0.664 ±0.003 |  | 0.758 ±0.032 |  |  |  |
| `E3_MIMIC_N50` | 1250 | 0.733 ±0.021 | 0.729 ±0.039 |  | 0.743 ±0.070 |  |  |  |
| `E4_MIMIC_N50` | 1500 | 0.714 ±0.018 | 0.722 ±0.006 |  | 0.738 ±0.074 |  |  |  |
| `E3_MIMIC_N70` | 1250 | 0.740 ±0.047 | 0.735 ±0.058 |  | 0.780 ±0.012 |  |  |  |
| `E4_MIMIC_N70` | 1500 | 0.733 ±0.046 | 0.728 ±0.033 |  | 0.761 ±0.028 |  |  |  |
| `E3_MIMIC_QN50` | 1250 | 0.706 ±0.034 | 0.748 ±0.033 |  | 0.738 ±0.010 |  |  |  |
| `E4_MIMIC_QN50` | 1500 | 0.721 ±0.030 | 0.735 ±0.014 |  | 0.762 ±0.021 |  |  |  |
| `E3_MIMIC_QN70` | 1250 | 0.735 ±0.020 | 0.699 ±0.020 |  | 0.738 ±0.060 |  |  |  |
| `E4_MIMIC_QN70` | 1500 | 0.755 ±0.032 | 0.755 ±0.025 |  | 0.747 ±0.061 |  |  |  |
| `E3_CF_S10` | 1250 | 0.679 ±0.029 | 0.732 ±0.048 |  | 0.679 ±0.024 |  |  |  |
| `E4_CF_S10` | 1500 | 0.726 ±0.023 | 0.731 ±0.039 |  | 0.689 ±0.022 |  |  |  |
| `E3_CF_S08` | 1250 | 0.640 ±0.097 | 0.673 ±0.050 |  | 0.647 ±0.070 |  |  |  |
| `E4_CF_S08` | 1500 | 0.594 ±0.023 | 0.637 ±0.045 |  | 0.618 ±0.093 |  |  |  |
| `E3_SRC_S10` | 1250 | 0.700 ±0.006 | 0.715 ±0.017 |  | 0.768 ±0.030 |  |  |  |
| `E4_SRC_S10` | 1500 | 0.729 ±0.014 | 0.691 ±0.032 |  | 0.683 ±0.007 |  |  |  |
| `E3_TEACH` | 1250 | 0.788 ±0.036 | 0.747 ±0.013 |  | 0.848 ±0.021 |  |  |  |
| `E4_TEACH` | 1500 | 0.780 ±0.009 | 0.760 ±0.034 |  | 0.856 ±0.020 |  |  |  |
| `E3_CTX` | 1250 | 0.748 ±0.047 | 0.732 ±0.045 |  | 0.824 ±0.021 |  |  |  |
| `E4_CTX` | 1500 | 0.723 ±0.055 | 0.728 ±0.009 |  | 0.759 ±0.062 |  |  |  |
| `E3_RTEACH` | 1250 | 0.792 ±0.030 | 0.755 ±0.021 |  | 0.810 ±0.043 |  |  |  |
| `E4_RTEACH` | 1500 | 0.771 ±0.049 | 0.756 ±0.047 |  | 0.823 ±0.006 |  |  |  |
| `E3_CONF` | 1250 | 0.725 ±0.032 | 0.704 ±0.021 |  | 0.769 ±0.013 |  |  |  |
| `E4_CONF` | 1500 | 0.683 ±0.073 | 0.712 ±0.002 |  | 0.791 ±0.031 |  |  |  |
| `E3_SRC_S08` | 1250 | 0.752 ±0.016 | 0.708 ±0.028 |  | 0.756 ±0.031 |  |  |  |
| `E4_SRC_S08` | 1500 | 0.740 ±0.033 | 0.726 ±0.029 |  | 0.748 ±0.031 |  |  |  |

### Study 2: CheXpert source

| group | n train | EffNet-B0 | RN-18 | DN-121 | ViT-S |
|---|---|---|---|---|---|
| `CX_E1` | 1000 | 0.888 ±0.015 | 0.891 ±0.008 | 0.916 ±0.009 |  |
| `CXVB_E1` | 1000 | 0.897 ±0.015 | 0.901 ±0.012 | 0.893 ±0.030 |  |
| `CX_E2_ERASE` | 1000 | 0.905 ±0.028 | 0.908 ±0.010 | 0.919 ±0.006 | 0.934 ±0.004 |
| `CXVB_E2_ERASE` | 1000 | 0.919 ±0.009 | 0.903 ±0.007 | 0.924 ±0.009 |  |
| `CX_R50` | 1500 | 0.908 ±0.008 | 0.912 ±0.005 | 0.877 ±0.060 |  |
| `CXVB_R50` | 1500 | 0.891 ±0.019 | 0.902 ±0.022 | 0.910 ±0.003 |  |
| `CX_N2000` | 2000 | 0.901 ±0.002 | 0.897 ±0.005 | 0.902 ±0.010 |  |
| `CXVB_N2000` | 2000 | 0.900 ±0.013 | 0.902 ±0.006 | 0.902 ±0.018 |  |
| `CX_SYN_ONLY_SD` | 2000 | 0.687 ±0.084 | 0.700 ±0.026 | 0.756 ±0.063 |  |
| `CX_SYN_ONLY_TEACH` | 2000 | 0.927 ±0.004 | 0.911 ±0.004 | 0.924 ±0.006 |  |
| `CX_R25` | 1250 | 0.888 ±0.024 | 0.900 ±0.014 | 0.874 ±0.029 |  |
| `CX_E3_SD` | 1250 | 0.900 ±0.011 | 0.889 ±0.016 | 0.883 ±0.024 |  |
| `CX_E4_SD` | 1500 | 0.902 ±0.004 | 0.901 ±0.014 | 0.884 ±0.033 |  |
| `CX_E3_TEACH` | 1250 | 0.891 ±0.002 | 0.900 ±0.005 | 0.863 ±0.061 |  |
| `CX_E4_TEACH` | 1500 | 0.916 ±0.010 | 0.901 ±0.006 | 0.912 ±0.016 |  |
| `CX_E3_RTEACH` | 1250 | 0.913 ±0.004 | 0.914 ±0.005 | 0.908 ±0.007 |  |
| `CX_E4_RTEACH` | 1500 | 0.923 ±0.006 | 0.918 ±0.005 | 0.910 ±0.012 |  |
| `CX_SYN_ONLY_SD2` | 2000 | 0.736 ±0.048 | 0.847 ±0.027 | 0.836 ±0.061 |  |
| `CX_SYN_ONLY_SD2V` | 2000 | 0.814 ±0.043 | 0.851 ±0.007 | 0.835 ±0.021 |  |
| `CX_SYN_ONLY_SD2C25V` | 2000 | 0.811 ±0.008 | 0.836 ±0.009 | 0.852 ±0.033 |  |
| `CX_N250` | 250 | 0.873 ±0.012 | 0.865 ±0.012 | 0.882 ±0.029 |  |
| `CX_N500` | 500 | 0.899 ±0.013 | 0.881 ±0.022 | 0.896 ±0.018 |  |
| `CX_FULL` | 9968 | 0.947 ±0.006 | 0.923 ±0.024 | 0.922 ±0.010 |  |
| `CX_N250_SYN2000_SD2C25` | 2250 | 0.843 ±0.008 | 0.854 ±0.063 | 0.873 ±0.013 |  |
| `CX_SYN2000_SD2C25` | 3000 | 0.859 ±0.023 | 0.889 ±0.018 | 0.894 ±0.016 |  |
| `CX_SYN_ONLY5K_SD2C25V` | 5000 | 0.792 ±0.042 | 0.796 ±0.020 | 0.811 ±0.020 |  |
| `CX_SYN_ONLY10K_SD2C25V` | 10000 | 0.789 ±0.032 | 0.793 ±0.052 | 0.811 ±0.029 |  |
| `CX_SYN_ONLY_SD3V` | 2000 | 0.780 ±0.020 | 0.758 ±0.032 | 0.763 ±0.030 |  |
| `CX_SYN_ONLY_SDR64V` | 2000 | 0.710 ±0.037 | 0.730 ±0.019 | 0.852 ±0.050 |  |
| `CX_SYN_ONLY_SDVIEWV` | 2000 | 0.771 ±0.051 | 0.822 ±0.007 | 0.848 ±0.043 |  |
| `CX_SYN_ONLY_SDOBVV` | 2000 | 0.780 ±0.020 | 0.748 ±0.022 | 0.705 ±0.108 |  |
| `CX_SYN_ONLY_SDBIGV` | 2000 | 0.774 ±0.047 | 0.751 ±0.034 | 0.744 ±0.057 |  |
| `CX_SYN_ONLY_RTEACH` | 2000 | 0.943 ±0.003 | 0.936 ±0.001 | 0.939 ±0.003 |  |
| `CX_SYN_ONLY_TEACHV2` | 2000 | 0.931 ±0.002 | 0.913 ±0.001 | 0.930 ±0.002 |  |
| `CX_SYN_ONLY_SD2C25` | 2000 | 0.811 ±0.019 | 0.857 ±0.018 | 0.848 ±0.008 |  |
