# Experiment inventory

Everything trained or analysed so far, what is still planned, and what was dropped or superseded (state 2026-10-13).
**368 training runs** on the GPU box (`artifacts/runs/<experiment>/<arch>/seed<k>`), all 3 seeds unless noted.
Training set = 1,000 real Kermany images (500 per class) unless noted; external evaluation on RSNA (opacity vs normal).
Δ = RSNA-test seed-ensemble AUROC change, EfficientNet-B0, pooled (view-balanced in brackets where computed, FINDINGS §13).

Architectures: ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0, ConvNeXt-T, ViT-S/16, Swin-T ("7 archs").

## 1. Ran: training experiments

### Baseline and hyper-parameters

| Experiment | What | Archs | Result | Where |
|---|---|---|---|---|
| HPO | Optuna, 25 trials per arch, objective Kermany-val BCE | 7 | lr dominates; transformers want 10–30× lower lr | report §4.1, `configs/hpo/` |
| HPO `harmonize224` | re-tune for equalised preprocessing | EffNet-B0 | used by the HARMT runs | FINDINGS §5 |
| E1 | 1,000 real, no augmentation | 7 | RSNA-test 0.84–0.90 pooled; 0.72–0.81 balanced | report §4.2 |

### Conventional augmentation and anti-overfitting variants

| Experiment | What | Archs | Result | Where |
|---|---|---|---|---|
| E2 | conventional augmentation | 7 | +0.05 (EffNet) | report §4.3 |
| E2_ERASE | E2 + random erasing | 7 | best on RSNA-dev for 6/7; +0.024…+0.085 vs E1 (+0.049…+0.150 balanced) | report §4.7, FINDINGS §11 |
| E2_ACQ | acquisition jitter (blur, noise, gamma) | 7 | helps Swin-T / ConvNeXt, hurts DenseNet | FINDINGS §11 |
| E2_ROBUST | E2 + acquisition + erasing | 7 | no gain over E2_ERASE | notebook 06 |
| E2_MIX | E2 + mixup (α 0.4) | 7 | best calibration for all 7 | notebook 06 |
| LP | linear probe, frozen backbone | 7 | 0.07–0.15 worse, except Swin-T | notebook 06 |

### More real data (controls and scaling)

| Experiment | What | Archs | Result | Where |
|---|---|---|---|---|
| R25 / R50 | +250 / +500 real (matched controls for E3/E4) | EffNet, ViT-S | ≈0 pooled; **+0.027 / +0.017 balanced** | report §4.4, FINDINGS §13 |
| N250 / N500 / N2000 | 250 / 500 / 2,000 real | EffNet, ViT-S | ViT gains ~0.03 per doubling, EffNet ~0.009 | report §4.4 |
| FULL / FULL_AUG | all 4,093 Kermany train (imbalanced), ± augmentation | EffNet, ViT-S | augmentation on 1k ≈ 4× real data | report §4.4 |

### Synthetic images (teammate's SD 1.5 + LoRA, tag `sd_lora`)

| Experiment | What | Archs | Result | Where |
|---|---|---|---|---|
| E3 / E4 | +250 / +500 synthetic | 7 | 14/14 negative, −0.009…−0.067 (−0.019…−0.148 balanced) | FINDINGS §1 |
| SYN1000 / SYN2000 | +1,000 / +2,000 synthetic | EffNet | −0.042 / −0.053, no dose trend | FINDINGS §8 |
| SYN_ONLY | 2,000 synthetic, no real | EffNet | 0.66 pooled, chance within view | FINDINGS §8 |
| ERASE_SYN250 / 500 | E2_ERASE + synthetic | EffNet, ViT-S, DenseNet | −0.055 / −0.060 (−0.145 / −0.134 balanced) | FINDINGS §7 |
| ERASE_REAL250 / 500 | E2_ERASE + real (control) | EffNet, ViT-S, DenseNet | ≈0 / +0.010 | FINDINGS §7 |
| E3 / E4 `_QM`, `_QC` | quality-matched / quality-counter selection of the synthetic images | EffNet | best −0.012 / −0.014 pooled, about −0.04 balanced; never helps | FINDINGS §10 |

### Preprocessing (histogram equalisation, `harmonize224`)

| Experiment | What | Archs | Result | Where |
|---|---|---|---|---|
| `_HARMT`: E1, E2_ERASE, E3, E4 | equalised + re-tuned hyper-parameters | EffNet | E1 +0.012; synthetic penalty −0.030 / −0.038 persists | FINDINGS §5 |
| `_HARM`: E1, E2, E2_ERASE, E3, E4, ERASE_SYN250/500 | equalised, BCE hyper-parameters (first round) | EffNet | superseded by HARMT (see §4 below); ERASE_SYN −0.065 / −0.109 | FINDINGS §5 |

### Robust training losses (`--loss`)

| Experiment | What | Archs | Result | Where |
|---|---|---|---|---|
| E1 / E3 `_LQCONS` | quality-consistency loss (λ 1) | EffNet, Swin-T; E1 also DenseNet, ViT-S | **+0.031** (+0.066 balanced); ViT +0.040, Swin-T +0.006 (AP +0.079), DenseNet +0.005 | FINDINGS §12 |
| E1 `_LQCONS03` / `_LQCONS3` | λ 0.3 / 3 | EffNet | +0.023 / +0.025, λ not critical | FINDINGS §12 |
| E2_ERASE / ERASE_SYN250 `_LQCONS` | qcons on top of augmentation | EffNet | +0.000; synthetic still −0.016 | FINDINGS §12 |
| E1 / E3 `_LQDRO` | online group DRO, label × sharpness | EffNet | +0.009, seed-unstable | FINDINGS §12 |
| E1 / E3 `_LFOCAL` | focal loss γ 2 | EffNet | +0.007 | FINDINGS §12 |
| E1 / E3 `_LGCE` | generalised cross-entropy q 0.7 | EffNet | −0.019; collapses with synthetic data | FINDINGS §12 |

## 2. Ran: analyses (no training; `scripts/synth_probe/`)

| Analysis | Question | Result | Where |
|---|---|---|---|
| `probe.py` | do synthetic labels carry pneumonia? | partly; much of it is image quality | FINDINGS §2 |
| `texture.py` | which shortcut do they carry? | Kermany sharpness cue copied, contrast cue amplified | FINDINGS §3 |
| `degrade.py` | causal test: blur / flatten healthy images | score rises; rule learned from real Kermany | FINDINGS §4 |
| `harm_check.py` | what can equalisation remove? | contrast yes, sharpness partly content | FINDINGS §5 |
| nearest-neighbour | memorisation? generator training set? | no memorisation; a separate domain; 658 distinct neighbours | FINDINGS §6 |
| RSNA-dev / RSNA-test split | selection without touching the reporting set | dev and test agree within 0.01 | FINDINGS §7 |
| in-domain Kermany-test check | do synthetic images help in-domain? | uninformative (extra real images move it as much) | FINDINGS §8 |
| `natural_quality.py` | shortcut visible in unaltered RSNA films? | E4 weighs blur almost like an opacity; penalty concentrated on blurry AP films; age shortcut | FINDINGS §9 |
| `quality_select.py` | build quality-aware subsets | matched / counter manifests | FINDINGS §10 |
| `loss_compare.py` | loss comparisons + quality weights | see losses above | FINDINGS §12 |
| `view_confound.py` | AP/PA confounding of the external metric | view alone 0.796; no conclusion flips, effects larger | FINDINGS §13 |
| Grad-CAM border share, calibration, overfitting ladder | notebook 06 | erasing moves mass from borders (6/7) | notebook 06 |

Notebooks 01–07 executed; Streamlit demo `app/streamlit_app.py` with exported models E1, E2, E2_ERASE, E3 (EffNet-B0).

## 3. Planned / in progress

| Item | Status |
|---|---|
| **Architecture gaps** (`scripts/queue_gaps.sh`, 78 runs) | R25 / R50 on ResNet-18/50, DenseNet, ConvNeXt, Swin-T; E1 + qcons on ResNet-18/50, ConvNeXt; E3 + qcons on 5 archs; ERASE_SYN250 / ERASE_REAL250 on ResNet-18/50, ConvNeXt, Swin-T. Started 2026-10-13 19:40, stopped 19:48 after R25 ResNet-18 + 2/3 ResNet-50; resumable (HANDOFF session 7) |
| Report `docs/report.md` | §4.5 (synthetic) still a placeholder; §3 RSNA-dev exception; research-question wording; §5 objectives 3–5; §6 limitations; §7 scripts; §4.7 for 7 archs; losses; **switch headline to view-balanced AUROC** (FINDINGS §13) |
| HANDOFF.md | stale numbers (E4 −0.069, "harmonised removes the penalty") |
| Slides, roles / contribution table | not started |
| Teammate questions | strict vs transfer generator training set, LoRA steps, filtering: unanswered |
| Optional | qcons with re-tuned hyper-parameters; GCE re-tuned; qcons on the other 3 archs; view-balanced columns in notebooks 04 / 06 / 07 |

## 4. Abandoned or superseded

| Item | Why |
|---|---|
| E3 / E4 `_PLACEHOLDER` (1 seed) | pipeline test with blended real images before the synthetic images arrived; excluded from all results (`docs/SYNTHETIC_DATA.md`) |
| `_HARM` conclusion "equalisation removes the synthetic penalty" | came from an untuned, weak harmonised baseline; re-tuned HARMT runs show the penalty persists. Runs kept, claim withdrawn |
| E4 2-seed estimate (−0.069) | E4 was paused for priority, then completed; 3-seed value −0.065 replaces it |
| In-domain Kermany-test as evidence for synthetic data | extra real images move it as much (R50 −0.060); dropped as uninformative |
| GCE loss | under-fits / collapses with BCE-tuned hyper-parameters; not re-tuned, judged not worth the GPU time |
| Selecting on Kermany validation AUROC | ≈0.99 for every model (shortcut-contaminated); BCE used instead (report §2.1) |
| Selecting on RSNA | would make it non-external; only RSNA-dev (30%) used, for choosing the ERASE base |
| "Quality-aware selection removes most of the penalty" | true pooled, only about half under the view-balanced metric; softened (FINDINGS §13) |
| Demo with E2 export | E2_ERASE exported as well (best external result) |
| First notebook-06 pass (seeds 0–1) | CUDA OOM in the Grad-CAM cell; fixed and re-run, failed copy ignored |
| Queue cut at 18:35 on 2026-10-12 | workstation needed; unfinished runs resumed later (resumable queues) |
