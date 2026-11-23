# Research pivot: *why* does synthetic augmentation hurt?

Status: arms A, B, A+B and C **done** 2026-10-16 (results below; `scripts/synth_probe/pivot_compare.py` → `reports/results/synth_probe/pivot_compare.csv`). Deadline about 4 weeks away (around 2026-11-11).
Numbers below come from `docs/FINDINGS_SYNTHETIC.md` (section numbers in brackets). External metric: RSNA-test, view-balanced AUROC.

## 1. Why the current question is not enough

The original question was "does generative augmentation improve Kermany → RSNA?". The answer is no: the synthetic
penalty is negative in 14 of 14 comparisons across 7 architectures [§1, §13]. On its own, "the generator copies Kermany's
shortcut" is the obvious explanation. **It does not fit the data, though:**

| Added to E1 (1,000 real), EfficientNet-B0 | Δ balanced AUROC |
|---|---|
| +250 / +500 **real** Kermany images (R25 / R50) | **+0.027 / +0.017** |
| +250 / +500 **synthetic** images (E3 / E4) | **−0.075 / −0.103** |
| synthetic vs the same number of real (R25 → E3, R50 → E4) | **−0.102 / −0.120** |

Real Kermany images carry the same shortcut, but they help. So copying the shortcut can't be the whole story. The question
is **what the synthetic images have that real images don't**.

## 2. The new question

> **Why does generative augmentation hurt cross-dataset generalisation? Which property of the synthetic images causes the
> damage: (a) an amplified source shortcut, (b) a weak class signal, or (c) the synthetic domain itself, including
> anatomical implausibility?**

The existing work becomes the setup. The new contribution is to separate the causes.

### What we already measured on the synthetic set

| Property | Real Kermany | Synthetic | Source |
|---|---|---|---|
| (a) Contrast cue (AUROC of low contrast → pneumonia) | 0.749 | **0.889** | §3 |
| (a) Sharpness cue | 0.865 | 0.861 | §3 |
| (b) Pneumonia seen by a robust classifier (E2_ERASE, mean p) | 0.96–0.98 | **0.28–0.54** | §2 |
| (b) Label separability by real-trained classifiers | n/a | 0.75–0.91 | §2 |
| (c) Real vs synthetic linear probe | n/a | **AUROC 1.000** | §6 |
| (c) Diversity: distinct nearest real images among 5,000 synthetic | n/a | 658; one real image is the nearest neighbour of 404 | §6 |
| (c) Anatomy: missing hearts, extra ribs, missing bones (seen by eye) | n/a | **not quantified yet** | — |

Evidence already in hand:
- Removing the quality cue (quality-aware selection) roughly halves the penalty (E3 −0.075 → −0.044, E4 −0.103 → −0.042) [§10, §13].
- About −0.04 remains after that, which points to (b) and/or (c).

## 3. Experiments: copy one property at a time with *real* images

Each arm adds 250 images to E1. Run on EfficientNet-B0 first, about 2–4 h of GPU per arm, then on 1–2 more architectures.

| Arm | What it adds | Tests | Prediction if that factor is the cause |
|---|---|---|---|
| **A. Amplified shortcut** | 250 *real* images picked so the contrast/sharpness cue is as strong as in the synthetic set: softest, flattest pneumonia and sharpest normals | (a) | Hurts about as much as E3 |
| **B. Weak class signal** | 250 real images with a share of labels flipped (or near-boundary images), tuned to the synthetic label separability 0.75–0.91 | (b) | Hurts about as much as E3 |
| **C. Anatomy filter** | E3 using only the synthetic images that pass an anatomy check (heart, lungs, ribs, spine present and of plausible size) | (c) anatomy | Penalty shrinks compared with random E3 |
| **C′. Anatomy filter + quality-matched** | C combined with the quality-matched selection from §10 | (a) + (c) | Remaining −0.04 penalty shrinks further |
| Existing: QM / QC | quality-matched or reversed synthetic images [§10] | (a) | Already done: halves the penalty |

**Anatomy scoring (needed for C):**
- Option 1: an off-the-shelf chest X-ray anatomy segmenter, e.g. the torchxrayvision PSPNet (heart, lungs, clavicles, spine). Not installed yet, and licensing and contamination need checking.
- Option 2: hand-label 200 synthetic images as plausible or implausible. This also gives the *rate* of broken images, a responsible-AI talking point.

### Arm C: anatomy scoring (done 2026-10-14)

`scripts/synth_probe/anatomy.py` segments 14 structures (torchxrayvision ChestX-Det PSPNet) in 1,200 synthetic images (the
first 600 per class in E3/E4 order) and 600 real Kermany ones. `anatomy_select.py` then scores each image by how far the size
and position of the large structures are from real images of the same class (robust z). Absent structures score the maximum.
- **The score catches what the user saw.** The most atypical synthetic images in `reports/results/synth_probe/anatomy_grid.png`
  have no or fragmented hearts, wrong-looking rib cages and chaotic lung outlines. Random synthetic images mostly look plausible.
- **On average, though, synthetic anatomy is *less* variable than real.** Score median 1.01 vs 0.82 real, but the 99th
  percentile is 3.2 vs 6.5. Fine structures are missing much more often: hila absent in 73–80% vs 25–44%, aorta 75% vs 52%.
- **Real pneumonia films segment messily** because of the opacities, so a real-image threshold flags no synthetic pneumonia.
  Instead, the selection drops the most atypical 20% *per class* (`ANAT`); `ANATBAD` puts the worst first.
- **Neither selection changes the quality shortcut** (low-contrast AUROC 0.90 in both, as random), so the comparison
  isolates anatomy.

## Results (2026-10-16): neither the shortcut, the weak labels nor gross anatomy explains the penalty

Δ view-balanced AUROC vs E1 on RSNA-test, 3-seed ensemble. "Share" = (arm − real) / (synthetic − real): 0 means the arm
behaves like added real images, 1 means it reproduces the full synthetic penalty.

| Arch, size | + real | + synthetic | MIMIC_Q (shortcut) | MIMIC_N (weak signal) | MIMIC_QN (both) | ANAT (best 80%) | ANATBAD (worst first) |
|---|---|---|---|---|---|---|---|
| EfficientNet-B0, +250 | +0.027 | −0.075 | +0.029 | +0.032 | +0.046 | −0.039 | −0.063 |
| EfficientNet-B0, +500 | +0.017 | −0.103 | +0.003 | +0.067 | +0.009 | −0.094 | −0.135 |
| ResNet-18, +250 | −0.021 | −0.081 | −0.014 | −0.032 | −0.024 | −0.108 | −0.091 |
| ResNet-18, +500 | +0.003 | −0.148 | −0.004 | −0.002 | −0.064 | −0.115 | −0.119 |
| DenseNet-121, +250 | +0.030 | −0.038 | +0.014 | −0.009 | +0.013 | −0.041 | −0.027 |
| DenseNet-121, +500 | +0.028 | −0.019 | +0.007 | −0.022 | −0.007 | −0.100 | −0.028 |
| **median share** | 0 | 1 | **0.08** | **0.10** | **0.16** | **0.99** | **1.02** |

* **(a) and (b) are ruled out as the main cause.** Real images carrying the synthetic set's sharpness/contrast profile, 30%
  mislabelled pneumonia, or both, behave like ordinary added real images. Together they reproduce about a sixth of the penalty
  (median; up to 0.44–0.75 for ResNet-18 / DenseNet-121 at +500, so they contribute something).
* **(c) gross anatomy is ruled out too.** Dropping the most atypical 20% (ANAT) does not reduce the penalty (median share 0.99).
  Putting the worst anatomy first (ANATBAD) does not increase it consistently. ANAT is better than ANATBAD on EfficientNet-B0
  (+0.024 / +0.040) but worse on DenseNet-121 (−0.015 / −0.072).
* **Seed noise.** The per-seed SD of these differences is 0.02–0.10, and the patient bootstrap CIs do not include it. So
  differences below ~0.04 between arms are not reliable. The real-mimic vs synthetic gaps (0.05–0.15, in 6 of 6
  configurations) are.
* **What is left: the synthetic domain itself.** Synthetic images are perfectly separable from real ones (§6, AUROC 1.000).
  Hypothesis for the next step: the generator encodes the class through its own cues (prompt-conditioned texture or global
  appearance) that exist only in synthetic images. A classifier that learns them gains nothing on real films and dilutes what it
  learns from real images. Evidence already in hand: synthetic-only training is at chance on RSNA PA films (§8).
* **Consequence for the GAN.** As pre-registered in §4, the anatomy result removes its main justification. A GAN is now
  only interesting as a different *synthetic domain* (does any generator's class cue transfer?), not as a fix for anatomy.

## 4. Where a GAN fits (and why anatomy is the right motive)

The GAN idea makes sense under (c). Diffusion samples can be **anatomically wrong while looking sharp** (no heart, 20 pairs
of ribs). GANs tend to stay closer to the training distribution's global structure. If anatomy is what hurts, a GAN that
copies the same Kermany shortcut but keeps the anatomy correct should hurt **less**.

Caveats:
- A GAN trained on Kermany will still copy the shortcut, probably *more* strongly (GANs blur, and blur is the pneumonia cue here). Measure (a), (b) and (c) on the GAN set too, otherwise a difference can't be attributed.
- On about 5k images, use StyleGAN2-ADA at 256 px. It fits in 8 GB but takes days to train, and it has failure modes of its own (mode collapse, texture artefacts).
- **Run arm C first.** If anatomy-filtered diffusion samples hurt just as much, anatomy is not the cause, and the GAN loses its main justification.

Better framing: each generator, and each guidance setting, is **one point on a property → harm plot**. Measure shortcut
strength, class signal, anatomy rate and realism on each synthetic set, then see which property predicts the downstream penalty.

Cheaper points for that plot, with no new training:
- regenerate with the existing LoRA at **CFG 1.5 / 3 / 7.5** (the current set used 4.5). Guidance should raise class signal *and* shortcut strength, and possibly broken anatomy. Needs the teammate.

## 5. Four-week plan (draft)

| Week | Work |
|---|---|
| 1 (→ 10-21) | Finish the gap queue and its analyses; switch the headline to view-balanced; build arms A, B and C (anatomy scorer or hand labels); run them on EfficientNet-B0. Ask the teammate about the CFG sweep and the GAN. |
| 2 (→ 10-28) | Repeat the clear arms on 1–2 more architectures. CFG sweep, if the teammate can do it. Start GAN training only if arm C showed that anatomy matters. |
| 3 (→ 11-04) | Property → harm analysis across all synthetic sets; notebook; report rewrite around the new question. |
| 4 (→ 11-11) | Buffer, demo, slides, reproducibility, responsible-AI section. No new experiments. |

## 6. Open questions

- [ ] Can the teammate regenerate at other CFG values? Train a GAN? Which training set and how many LoRA steps were used?
- [ ] How common is broken anatomy? (count it in arm C)
- [ ] Is the pivot fine with the course, i.e. is a "why it fails" question acceptable instead of "does it work"?

## 7. Constructive direction: synthetic data that carries only the pathology (idea, 2026-10-16)

**Diagnosis so far.** A generator fine-tuned on one hospital and sampled with a "pneumonia" and a "normal" prompt
reproduces *everything* that separates the two classes *in that hospital*, not only the disease: acquisition quality, patient
age and size, positioning. Its synthetic-only classifier scores 0.85–0.91 on real Kermany test images but is at chance on RSNA
PA films. Filtering for quality or anatomy does not change this. Both classes used identical settings (one adapter, CFG 4.5,
30 steps, one prompt per class), so this is not a batch effect between generation runs.

**Idea: don't sample classes, edit images.** Produce **counterfactual pairs**: take a real image and change *only the lungs*
with the generator. Inside each pair, everything except the pathology is identical (patient, hospital, quality, view,
markers), so no shortcut can predict the label. The class difference then contains only what was edited: the pathology.
That is what "universal" synthetic data has to be: data whose class signal does not encode the source site.

| Variant | How | Cost |
|---|---|---|
| **CF-inpaint** (main) | real Kermany normal → inpaint opacity inside the lung mask (PSPNet masks from `anatomy.py`) with the "pneumonia" prompt; real pneumonia → inpaint inside the lungs with the "normal" prompt. Both directions, so the pairs are balanced. | SD-1.5 base (4 GB) + `diffusers` on the box; existing LoRA; about 1–2 h to generate 2×500 |
| CF-img2img (cheaper baseline) | the same real image through img2img with each prompt, strength 0.3–0.5, same seed | same setup, no masks |
| CF-nuisance (control) | the same pairs, then a random blur/contrast applied identically to both images of a pair | free |

**Tests** (same protocol as everything else: E1 + 250 / 500, 3 seeds, 3 architectures, RSNA-test view-balanced):
1. Within the CF set, sharpness/contrast must not predict the label (AUROC ≈ 0.5 by construction; check it, since inpainting can blur).
2. Label signal by a robust real-trained classifier: does the edit look like pneumonia?
3. The main result: E1 + CF vs E1 + R25 (real) vs E1 + E3 (sampled synthetic), with particular attention to **RSNA PA**, where sampled synthetic data is at chance.
4. Optional second "site" for universality: AP vs PA in RSNA already act as two domains. A second external dataset would make the claim stronger (time permitting).

**Outcomes.**
- **CF ≥ real:** a positive method. "Synthetic augmentation fails because sampled synthetic data copies the source site's class-conditional distribution. Counterfactual editing removes that and helps external performance."
- **CF ≈ 0:** the edits carry too little real pathology. The shortcut is gone, but that still says what synthetic data would need.
- **CF < 0:** editing introduces its own cue, e.g. inpainting texture. Test by inpainting with the *same* class prompt (a "no-op edit" control).

**Team split.** Generation belongs to the teammate (their adapter and prompts). The user runs the evaluation, as before.
Literature to cite: counterfactual data augmentation for spurious correlations, and diffusion-based counterfactual explanations in medical imaging. Check exact references before writing.

**Implementation and first check (2026-10-16).** `scripts/synth_cf/cf_generate.py` (separate `.venv-gen`, GPU capped at 35%):
every source image (Kermany pool, patients outside E1, half normal / half pneumonia) is edited twice, once per prompt, with the same
lung mask, noise seed and strength; pixels outside the dilated, feathered lung mask are pasted back from the source. Two strengths:
1.0 (`cf_lung_s10`) and 0.8 (`cf_lung_s08`), 250 pairs each, trained as E3/E4 with tags `CF_S10` / `CF_S08` (`scripts/queue_cf.sh`).
`scripts/synth_cf/cf_check.py` on the first 56 pairs at strength 1.0 (robust real-trained classifiers, E2_ERASE seed 0):

| | EfficientNet-B0 | DenseNet-121 |
|---|---|---|
| pneumonia-edit scores higher than the normal-edit of the same image | 82% of pairs | 89% |
| mean p, normal-edit / pneumonia-edit, **normal** source | 0.04 / 0.08 | 0.08 / 0.22 |
| mean p, normal-edit / pneumonia-edit, **pneumonia** source | 0.66 / 0.93 | 0.71 / 0.90 |
| AUROC of the score for the *edit* label / for the *source* label | 0.61 / **0.97** | 0.62 / **0.93** |

Quality cue inside the CF set: low sharpness 0.63, low contrast 0.66 (sampled synthetic: 0.88 / 0.90).
- The edits push in the right direction almost always, but weakly: the classifiers' score still follows the **source** image,
  not the prompt. A "normal" edit of a pneumonia film still looks like pneumonia (p ≈ 0.7), and a "pneumonia" edit of a normal film
  barely looks sick (p 0.08–0.22). So the CF set's labels are partly wrong from a real classifier's point of view.
- The quality shortcut is mostly gone (0.63–0.66 instead of 0.88–0.90). It is not 0.5, because the pneumonia prompt makes the
  lungs softer and flatter. That residue is the edit itself, which the CF < 0 outcome above anticipates.
- Expectation for training: much less shortcut, but also a weak class signal. "CF ≈ 0" is the most likely outcome; strength 1.0
  vs 0.8 shows whether a stronger edit buys signal.

**Stronger label-noise doses (2026-10-16 17:30, EfficientNet-B0; ResNet-18 / DenseNet-121 running).** Δ view-balanced vs E1,
ensemble [95% CI]; share of the synthetic penalty in brackets (E3: real +0.027, synthetic −0.075; E4: real +0.017, synthetic −0.103):

| | N (30%) | N50 | N70 | QN (30%) | QN50 | QN70 |
|---|---|---|---|---|---|---|
| +250 (E3) | +0.032 | −0.000 (0.26) | +0.013 (0.14) | +0.046 | **−0.036 (0.62)** | −0.008 (0.34) |
| +500 (E4) | +0.067 | −0.021 (0.32) | +0.001 (0.13) | +0.009 | −0.011 (0.23) | −0.002 (0.16; 2 seeds) |

Heavier noise moves the result part of the way towards the synthetic penalty (median share about 0.25), but not in proportion
to the dose: 70% is no worse than 50%. Per-seed SD is 0.015–0.05, so only "noise costs at most ~0.04" is robust. Label noise
explains a minority of the penalty at most.

**ResNet-18 doses (2026-10-19, `scripts/synth_probe/dose_compare.py`).** Δ view-balanced vs E1, ensemble; share of the synthetic
penalty in brackets (E3: real −0.021, synthetic −0.081; E4: real +0.003, synthetic −0.148):

| | N (30%) | N50 | N70 | QN (30%) | QN50 | QN70 |
|---|---|---|---|---|---|---|
| +250 (E3) | −0.032 | −0.040 (0.32) | −0.020 (0.0) | −0.024 | −0.018 (0.0) | −0.056 (0.58) |
| +500 (E4) | −0.002 | −0.047 (0.33) | −0.037 (0.26) | −0.064 | −0.029 (0.21) | −0.009 (0.08) |

Same picture as EfficientNet-B0: mislabelled real images cost up to about 0.05, synthetic images 0.08–0.15; median share about 0.25,
no dose trend. Per-seed SD is large here (0.02–0.10).

**Strength 0.8 CF pairs** (`cf_check.py cf_lung_s08`, 250 pairs): edits are much weaker than at 1.0. Pneumonia-edit higher in 72–75% of
pairs, median Δp 0.00, label AUROC 0.55 (1.0: 0.60), source-label AUROC 0.97–0.98; quality cue 0.60 / 0.57 (1.0: 0.64 / 0.67).
At 0.8 the two images of a pair are nearly identical, so CF_S08 behaves close to a "same image, both labels" control: no shortcut,
almost no class signal.

**DenseNet-121 doses (2026-10-19).** Δ view-balanced vs E1, ensemble; share in brackets (E3: real +0.030, synthetic −0.038;
E4: real +0.028, synthetic −0.019):

| | N (30%) | N50 | N70 | QN (30%) | QN50 | QN70 |
|---|---|---|---|---|---|---|
| +250 (E3) | −0.009 | −0.049 (1.16) | −0.002 (0.47) | +0.013 | −0.050 (1.18) | −0.030 (0.88) |
| +500 (E4) | −0.022 | −0.018 (0.98) | −0.012 (0.85) | −0.007 | −0.026 (1.15) | −0.021 (1.04) |

**Different from the other two architectures:** for DenseNet-121, whose synthetic penalty is the smallest (−0.02 to −0.04), 50–70%
mislabelled pneumonia slots reproduce the whole penalty (share 0.85–1.18, except N70 E3 0.47). So the synthetic harm has two parts:
a label-noise-sized part (≈ 0.02–0.05, which every architecture shows) and an extra part that only EfficientNet-B0 and ResNet-18
suffer (synthetic −0.08 to −0.15). Label noise explains the first, not the second. Per-seed SD 0.003–0.09.

## CF training results (2026-10-19, final: all 36 runs, 3 seeds each)

Δ vs E1 on RSNA-test, 3-seed ensemble, view-balanced [95% CI] (`scripts/synth_cf/cf_compare.py` → `cf_compare.csv`):

| Arch, size | + real | + sampled synthetic | + CF pairs, strength 1.0 | + CF pairs, strength 0.8 |
|---|---|---|---|---|
| EfficientNet-B0, +250 | +0.027 | −0.075 | −0.076 [−0.085, −0.067] | −0.102 |
| EfficientNet-B0, +500 | +0.017 | −0.103 | **−0.032** [−0.042, −0.022] | −0.162 |
| ResNet-18, +250 | −0.021 | −0.081 | **−0.028** [−0.035, −0.021] | −0.096 |
| ResNet-18, +500 | +0.003 | −0.148 | **−0.051** [−0.058, −0.044] | −0.139 |
| DenseNet-121, +250 | +0.030 | −0.038 | **−0.125** [−0.136, −0.115] | −0.117 |
| DenseNet-121, +500 | +0.028 | −0.019 | **−0.115** [−0.126, −0.105] | −0.150 |

**Outcome "CF < 0": counterfactual pairs hurt too.**
- Strength 1.0 hurts less than sampled synthetic data for ResNet-18 (both sizes) and EfficientNet-B0 +500, as much for EfficientNet-B0
  +250, and much more for DenseNet-121, which sampled synthetic data barely hurt.
- **Strength 0.8, where the two images of a pair are nearly identical, hurts most: −0.10 to −0.16 for every architecture.** That is far
  worse than real images with 35% wrong labels (≤ 0.05).
- Reading: mislabelled real images are *inconsistent* noise (no common feature separates them), so the model mostly ignores them.
  A CF pair offers a *consistent* but wrong class difference, the generator's own lung texture, and nearly nothing else. The model
  learns that texture as "pneumonia", which does not exist in RSNA. The weaker the real pathology in the edit (0.8 < 1.0), the more the
  model relies on that texture and the worse it transfers.
- Together with the mimic arms: the harm is not the hospital shortcut, label noise or anatomy as such. It arises **whenever the only
  consistent difference between the synthetic classes is a generator-specific feature rather than real disease**. Removing the
  hospital shortcut (CF) does not help, because the generator's texture takes its place.
- Implication: synthetic data needs a class difference a robust real-trained classifier recognises as disease (here mean p 0.08–0.22
  for the "pneumonia" edit of a normal film). A check of that kind (label signal by a robust real classifier) predicts harm before training.

## 8. Can the synthetic images be made to help? (started 2026-10-19 ~16:00)

The CF result says the harm comes from a *consistent class difference that belongs to the generator* (its prompt-driven
texture), not from the hospital look or from wrong labels. If that is right, three cheap changes to the *labels* or the
*selection* of images we already have should remove the harm. Queue `scripts/queue_fix.sh` (tmux `fix`, log `logs_fix.log`),
manifests from `scripts/synth_fix/make_manifests.py`, results via `scripts/synth_fix/fix_compare.py`. E3 / E4 sizes,
EfficientNet-B0, ResNet-18, DenseNet-121, 3 seeds; about 90 runs of 1–3 min each.

Teacher = mean of the robust E2_ERASE models (seed 0; EfficientNet-B0, DenseNet-121, ViT-S). It gives the sampled synthetic
pneumonia images mean p 0.37 (real pneumonia 0.96), the synthetic normals 0.06 (real 0.04).

| Arm | What changes | Prediction if "generator-specific class difference" is the cause |
|---|---|---|
| **SRC_S10 / SRC_S08** (no-op-edit control) | the CF pairs, but both edits labelled with the **source image's** real label: generated lung texture is now independent of the label | harm largely gone (close to + real) |
| **TEACH** | sampled synthetic images, label = **teacher probability** instead of the prompt | harm largely gone; positive if the extra images transfer the teacher's robustness |
| **CONF** | sampled synthetic images the teacher agrees with (pneumonia p ≥ 0.60 first; normals p < 0.5), prompt labels | penalty shrinks |
| **RTEACH** (control for TEACH) | the extra **real** images of R25 / R50 with teacher labels | if TEACH ≈ RTEACH, the gain is distillation, not the synthetic images |

| **CTX** (added 16:00) | E1's **own** images with the real lungs kept and everything else (body outline, neck, abdomen, borders, markers) redrawn by our generator with one class-neutral prompt, strength 1.0 (`scripts/synth_fix/ctx_generate.py`, prototype `reports/results/synth_fix/ctx_prototype.png`) | a generative version of E2_ERASE's border masking (+0.10): no new real information, real pathology, class-independent generated texture; **the arm most likely to be positive** |

If SRC still hurts, the generated texture harms *as such* (a domain effect), whatever the labels.

**First result (EfficientNet-B0, Δ balanced vs E1):** SRC_S10 +250 −0.051 [−0.058, −0.043] (CF_S10 −0.076), +500 −0.023
[−0.031, −0.016] (CF_S10 −0.032). The source label removes a third or less of the CF harm. **ResNet-18 +250: SRC_S10 −0.065
[−0.071, −0.060], worse than CF_S10 (−0.028).**
**Caveat (design flaw found 16:05): SRC is not a clean "texture only" control.** At strength 1.0 the lungs are fully
regenerated from the prompt, so they no longer show the source's pathology; with the source label, the only consistent
evidence for the label is the *untouched context outside the lungs*, i.e. exactly the Kermany context shortcut. SRC therefore
tests "label explained by context only" and is expected to hurt. Together, the three edit arms decompose the image:
CF = generated lungs carry the label; SRC = real context carries the label; **CTX = real lungs carry the label, context is
generated and class-independent**. Only CTX keeps the label on the real pathology.

**Decision 2026-10-19: not pursued as a replacement.** The project's aim is to build and evaluate *our own* generator, so a pretrained one is at most an external reference point. The paragraph below is kept as context for the discussion section.

**Bigger option, not started: a generator trained on adult multi-site data.** Every generator here was fine-tuned on
Kermany (paediatric, one hospital), so it cannot add what RSNA needs: adult films, AP/portable films, devices. The literature
gains under distribution shift (Ktena et al. 2024) come from generators trained on broader data. Publicly downloadable
candidate: **DiffChest** (github.com/peterhan91/diffchest; diffusion autoencoder trained on MIMIC-CXR + CheXpert + PadChest,
no NIH, so no overlap with RSNA; 256 px; checkpoints on Google Drive; MIT code; has a classifier-guided latent manipulation,
i.e. it can make pneumonia / normal versions of an adult film). RoentGen and ChexGen need credentialed (PhysioNet) access;
RadEdit (Microsoft) was trained on ChestX-ray8, which contains the RSNA images, so it would leak the test set.

### Why the Google paper found gains and we do not (Ktena et al., DeepMind / Google, arXiv 2304.09218, Nature Medicine 2024)
Read from the paper's text (§3.1–3.2.2), 2026-10-19:
- **Chest X-ray setup:** generator *and* classifier trained on **201,055 CheXpert images** (adult, Stanford), five conditions
  (atelectasis, consolidation, cardiomegaly, effusion, edema), tested on CheXpert (in-distribution) and **ChestX-ray14**
  (OOD: adult → adult, demographic and acquisition shift). Diffusion model conditioned on the diagnostic label (optionally
  also on an attribute such as sex or hospital); synthetic data sampled "fairly" over the attribute. Gain: **+5.2% average
  AUC OOD**, fairness gap −44.6%. In histopathology the generator also saw *unlabelled images of the other hospitals*.
- **Differences to our setting:** (1) 200× more real data behind the generator (ours: a LoRA on Kermany's ~5k training
  images); (2) source and target are both adult, while ours is paediatric single-site → adult multi-site, so our generator
  cannot produce what RSNA needs; (3) their generator is trained from scratch on the medical data with label
  conditioning, ours is a LoRA on SD-1.5 with one text prompt per class; (4) where it helped most, the generator had seen
  unlabelled target-like data. Our results are therefore consistent with theirs: synthetic data helps when the generator
  has seen the variation the target needs, and adds a generator-specific class cue when it has not.

## 9. Second source dataset (proposed 2026-10-19, user: "fits 3 weeks, GPU 24/7"; not started)
Kermany looks like the bottleneck: all ~2,300 real Kermany images give only +0.05 on RSNA-test (EfficientNet-B0 0.736 →
0.785), while erasing augmentation on 1,000 gives +0.12; image width alone predicts the Kermany label (AUROC 0.95).
A generator trained on Kermany cannot exceed that. Plan: **keep Kermany as study 1 and repeat the core design with an adult
source as study 2**, same target (RSNA-dev / RSNA-test), same generator recipe (SD-1.5 LoRA, retrained by us).
- **Source: CheXpert** (Stanford, adult, ~224k images; small version ≈ 11 GB). It does not overlap NIH/RSNA, has AP/PA
  view labels, and is the source the Google paper used (Ktena et al.: CheXpert → ChestX-ray14), so study 2 is directly
  comparable to their positive result. Access: Stanford AIMI research-use agreement, which the user signs (not Claude).
  NIH ChestX-ray14 is excluded (RSNA comes from it); PadChest (registration) is the fallback.
- **Task mapping:** positive = Lung Opacity, Pneumonia or Consolidation = 1; negative = No Finding = 1; uncertain (−1)
  excluded; frontal only; patient-level splits.
- **Runs:** E1, E2, E2_ERASE, R25, R50, E3, E4 (+ CTX if it works in study 1) × 3 architectures × 3 seeds ≈ 70 runs plus
  the generator (LoRA training ≈ hours on 8 GB, then ~5k samples). Study 1 needed ~660 runs in two weeks, so this fits.
- **Needs from the user:** CheXpert download link / files on the box; the teammate's LoRA training script (not in the repo).

### §8 results (2026-10-19 18:50; CONF and SRC_S08 not run yet, queue paused)
Δ view-balanced vs E1 on RSNA-test, 3-seed ensemble (CIs ±0.01; `reports/results/synth_fix/fix_compare.csv`):

| Arch, +250 / +500 | + real | + real, teacher labels (RTEACH) | + synthetic, prompt labels | **+ synthetic, teacher labels (TEACH)** | CTX (real lungs, redrawn context) |
|---|---|---|---|---|---|
| EfficientNet-B0 | +0.027 / +0.017 | +0.040 / +0.037 | −0.075 / −0.103 | **+0.038 / +0.027** | +0.011 / +0.011 |
| ResNet-18 | −0.021 / +0.003 | −0.023 / −0.016 | −0.081 / −0.148 | **−0.032 / −0.019** | −0.032 / −0.050 |
| DenseNet-121 | +0.030 / +0.028 | +0.014 / +0.026 | −0.038 / −0.019 | **+0.053 / +0.062** | +0.036 / +0.003 |

- **Relabelling the same synthetic images with the robust real-trained teacher (E2_ERASE ensemble) removes the whole
  penalty in 6/6 cells** (gain over prompt labels +0.08 to +0.13). Synthetic images with teacher labels are now as useful as
  real images (EfficientNet-B0, ResNet-18) or more useful (DenseNet-121: +0.053 / +0.062 vs real +0.030 / +0.028).
- **Is it just distillation?** For EfficientNet-B0 and ResNet-18 TEACH ≈ RTEACH (teacher labels on real images), so the
  gain is the teacher's. For DenseNet-121 TEACH beats RTEACH by +0.039 / +0.036, at the edge of the ±0.03–0.04 seed-noise
  rule: weak evidence that synthetic images carry something beyond the teacher.
- **Mechanism confirmed:** the images are not the problem, the *prompt labels* are. The teacher says the average synthetic
  "pneumonia" image is p = 0.37 pneumonia; training on hard prompt labels forces the classifier to find whatever separates
  the prompts (generator texture). Soft teacher labels remove that incentive.
- **CTX is not a clear win:** small positive for EfficientNet-B0 (+0.011), mixed for DenseNet-121, negative for ResNet-18
  (driven by AP films, −0.065 / −0.099). Inconclusive under the seed-noise rule.
- Report framing: "generative augmentation fails with prompt labels and works with teacher labels" is the constructive result.

### §9 status (2026-10-20 11:05)
- Manifests (`scripts/chexpert/make_manifest.py`): one frontal adult image per patient, 4,983 / 4,985 train, 250 / 250 val,
  750 / 750 internal test (`C_test`). CheXpert has its own view shortcut (pneumonia-type 86% AP vs normals 65% AP), so two
  sources: `chexpert` (natural mix, runs `CX_*`) and `chexpert_vb` (training subsets alternate AP/PA, exactly 50% AP per class
  for every subset up to 1,412 pneumonia-type images; runs `CXVB_*`). Val/test identical in both.
- `run.py --source` swaps the real train + val manifest; HPO parameters are the Kermany ones (not re-tuned; say so in the report).
  `C_test` was added to the eval sets: Kermany runs get a second external score (`--eval-only` at the end of the queue).
- Queue `scripts/queue_cx.sh` (tmux `cx`, starts when `fix` ends): E1, E2_ERASE, R50, N2000 × both sources × 3 archs × 3 seeds,
  + E2_ERASE ViT-S (CheXpert teacher). Next: CX_TEACH once a CheXpert-trained generator exists (friend's LoRA script).

### §8 last arms (2026-10-20 12:20, fix queue complete)
Δ view-balanced vs E1, seed ensemble, +250 / +500: **CONF** (prompt labels, teacher-convincing images first) EfficientNet-B0
−0.018 / −0.083, ResNet-18 −0.070 / −0.063, DenseNet-121 −0.027 / +0.007: choosing images does not fix it, relabelling does.
**SRC_S08** (strength-0.8 pairs, source label) EfficientNet-B0 +0.023 / +0.010, ResNet-18 −0.065 / −0.057, DenseNet-121
−0.040 / −0.048: mixed, confounded like SRC_S10.

### §9 first result (2026-10-20 13:00): the source dataset was the main problem
E1 with 1,000 CheXpert images instead of 1,000 Kermany images, view-balanced RSNA-test AUROC (seed ensemble):

| Arch | Kermany E1 | CX_E1 (natural views) | CXVB_E1 (view-balanced) | CX_E1 on CheXpert test |
|---|---|---|---|---|
| EfficientNet-B0 | 0.770 | **0.922** (+0.153) | 0.926 (+0.157) | 0.894 |
| ResNet-18 | 0.797 | **0.917** (+0.120) | 0.922 (+0.125) | 0.890 |
| DenseNet-121 | 0.810 | **0.939** (+0.129) | 0.929 (+0.118) | 0.902 |

- +0.12 to +0.16: far larger than anything found on Kermany (best interventions +0.03 to +0.06). Adult, multi-view source data
  transfers to RSNA; Kermany's paediatric single-hospital shortcut was the bottleneck.
- View-balancing the training set changes nothing measurable (|Δ| ≤ 0.01) under the view-balanced metric.
- RSNA (external) scores above CheXpert's own test set: CheXpert test labels come from report text (noisier positives),
  RSNA's are radiologist boxes vs clean normals.
- First CX_E2_ERASE seed: +0.004 vs CX_E1, i.e. the strong augmentation that helped on Kermany may not be needed here.
- Consequence for the synthetic question: the baseline is now 0.92–0.94, so less room to gain; the generator must be
  retrained on CheXpert for a fair study-2 synthetic arm.

### §9 generator on CheXpert (2026-10-20 13:35, queued)
Teammate's `train_lora.py` received and vendored as `scripts/generator/train_lora.py` (it is the recipe of the Kermany
adapter: same r / alpha / target modules). Changes: `--variant fp16` (load the cached fp16 base weights, upcast to fp32;
no 4 GB download), `--seed`, and the model id (runwayml/... was removed from the Hub). Training data
(`scripts/generator/prepare_chexpert.py`): all 9,968 CheXpert train images, 512 px squash, captions = the two study-1 class
prompts; val/test never reach the generator. Defaults kept: 2,500 steps × batch 4 (~1 epoch), lr 2e-5.
Sampling (`scripts/generator/sample.py`): text-to-image with the study-1 prompts, negative prompt, CFG 4.5, 30 steps,
1,000 per class, fixed seeds. Teacher (`scripts/generator/cx_manifests.py`): CX_E2_ERASE seed 0 EfficientNet-B0 +
DenseNet-121 + ViT-S. Arms (`scripts/queue_gen_cx.sh`, tmux `gen`, after `cx`): CX_R25, CX_E3/E4_SD (prompt labels),
CX_E3/E4_TEACH, CX_E3/E4_RTEACH, all vs CX_E1. Both scripts smoke-tested (2 CPU training steps; 2 GPU samples).
Note: the generator sees ~10k real images while CX_E1 sees 1k; the fair references are CX_R25/R50/N2000 (more real data).

### §9 baselines (2026-10-20 14:45, 69/75 runs)
On CheXpert nothing moves the RSNA score: Δ vs CX_E1 (seed ensemble, view-balanced) E2_ERASE −0.006..+0.013,
R50 −0.031..+0.014, N2000 (2× the real data) −0.016..+0.005; view-balanced (CXVB) the same. All within seed noise.
At 1,000 adult images the classifier is already on a plateau (0.92–0.94) where even doubling the *real* data does not help,
so the study-2 synthetic arms can at best be neutral; what they can still show is whether prompt-labelled synthetic
images *hurt* there too, and whether teacher labels remove that harm.

### Second external test for study 1 (2026-10-20 15:45): Kermany runs scored on CheXpert test (`C_test`)
Mean per-seed AUROC (EfficientNet-B0 / ResNet-18 / DenseNet-121): E1 0.734 / 0.792 / 0.791; E2_ERASE 0.797 / 0.790 / 0.816;
E4_TEACH 0.784 / 0.773 / 0.827; E4_sd_lora (prompt labels) 0.725 / 0.755 / 0.776; for scale CX_E1 (in-domain) 0.871 /
0.873 / 0.880. The study-1 pattern replicates on an independent adult hospital: prompt-labelled synthetic images hurt
(−0.01 to −0.04), teacher-labelled ones help (EfficientNet-B0 +0.05, DenseNet-121 +0.04; ResNet-18 −0.02).
Also: CX_E2_ERASE ViT-S (teacher member) 0.941 view-balanced RSNA-test, the best single configuration so far.

## 10. Project goal, restated (2026-10-20)
The original aim: **our own generator should produce data that can replace real chest X-ray datasets** (no patient
data; real datasets are scattered and often not open, e.g. CheXpert needs approved access). So the main test is
**synthetic-only training vs the same number of real images**, scored on internal and external tests:
- study 1 (Kermany generator): SYN_ONLY_sd_lora (2,000 synthetic) vs N2000 — already run; at chance on RSNA PA films.
- study 2 (CheXpert generator): CX_SYN_ONLY_SD and CX_SYN_ONLY_TEACH vs CX_N2000 (queued, `queue_gen_cx.sh`).
  TEACH variant = share a generator + a real-trained labelling model instead of patient images.
- "no PII" needs evidence: memorization check (nearest real training image of each synthetic image, compared with
  the real-holdout-to-train distance). To do once the CheXpert samples exist.
Sections §1–9 become the analysis of *why* synthetic data falls short and what fixes it (labels, source dataset).
After this batch: freeze experiments, write the report.

## 11. Generator round 2 (2026-10-21): evaluate and improve our generator
Review finding: the generator had never been evaluated or tuned (recipe as received: lr 2e-5, 2,500 steps ≈ 1 epoch,
CFG 4.5, one prompt per class), while the classifier side got ~900 runs.

**Main test of the restated goal, study 2 (2,000 images, view-balanced RSNA-test, mean of 3 seeds, EfficientNet-B0 /
ResNet-18 / DenseNet-121):** real CX_N2000 0.901 / 0.897 / 0.902; synthetic, prompt labels CX_SYN_ONLY_SD 0.687 / 0.700 /
0.756; synthetic, teacher labels CX_SYN_ONLY_TEACH 0.927 / 0.911 / 0.924. Study 1 (Kermany generator) SYN_ONLY 0.525.
Teacher labels are distillation from a real-trained model, so only the prompt-label row tests replacement.

**Generator metrics** (`scripts/generator/evaluate.py` → `reports/results/generator/eval_<source>.csv`; ImageNet
EfficientNet-B0 features; KID / precision / recall on 100 per class vs real train, 10 draws; teacher = E2_ERASE seed 0
ensemble of the source; memorisation = nearest real train image, cosine):

| Set | KID×1000 | precision | recall | teacher p normal / pneumonia | teacher AUROC | NN sim median / max | closer than any real | distinct NN |
|---|---|---|---|---|---|---|---|---|
| real CheXpert test (floor) | ≈0 | 0.85 | 0.83 | 0.26 / 0.74 | 0.90 | 0.853 / 0.926 | 0 | 85% |
| CheXpert generator v1 (`cx_sd`) | 122 | 0.29 | **0.06** | 0.25 / **0.45** | 0.74 | 0.811 / 0.875 | **0** | 28% |
| real Kermany test (floor) | 9 | 0.90 | 0.77 | 0.32 / 0.99 | 0.97 | 0.903 / 0.954 | 0 | 78% |
| Kermany generator (`sd_lora`) | 124 | 0.14 | 0.14 | 0.06 / 0.37 | 0.88 | 0.837 / 0.904 | 0 | 13% |

- No memorisation in either generator: no synthetic image is as close to a training image as held-out real images are.
- Both generators are far from real (KID ≈ 120) and cover little of the real distribution (recall 0.06–0.14; few distinct
  nearest neighbours). Pneumonia is under-drawn (teacher p 0.37–0.45 vs 0.74–0.99 real). Low diversity + weak class signal
  is the generator-side explanation of the synthetic-only gap.
- Round 2 (`scripts/queue_gen_v2.sh`, tmux `gen2`): same recipe with lr 1e-4 and 7,500 steps (~3 epochs), adapters at
  2,500 / 5,000 / 7,500; 2,000 samples (`cx_sd_v2`); SYN_ONLY tag SD2. `scripts/queue_gen_v2_eval.sh` (tmux `gen2eval`)
  samples 100 per class from the intermediate adapters and re-runs the evaluation on all CheXpert sets.

**Round 2 result (2026-10-22): synthetic-only, prompt labels, 2,000 images** (view-balanced RSNA-test / CheXpert test,
mean ± SD of 3 seeds; EfficientNet-B0, ResNet-18, DenseNet-121):

| Training set | RSNA-test | CheXpert test |
|---|---|---|
| 2,000 real (CX_N2000) | 0.901 / 0.897 / 0.902 | 0.891 / 0.879 / 0.883 |
| generator v1 (CX_SYN_ONLY_SD) | 0.687 / 0.700 / 0.756 | 0.744 / 0.724 / 0.754 |
| **generator v2 (CX_SYN_ONLY_SD2)** | **0.736 / 0.847 / 0.836** (SD 0.03–0.06) | **0.778 / 0.824 / 0.795** |

| **v2, synthetic validation too (CX_SYN_ONLY_SD2V)** | **0.814 / 0.851 / 0.835** (SD 0.01–0.04) | 0.765 / 0.772 / 0.792 |

SD2V uses no real image anywhere in the classifier pipeline: early stopping / checkpoint / threshold on 250 + 250 held-out
synthetic images (sampler seed 1) instead of the 500 real CheXpert val images (`run.py --val`). Removing the real
validation costs nothing on RSNA (mean 0.833 vs 0.806 with real val; EfficientNet-B0 within its seed noise) and up to
−0.05 on CheXpert test (ResNet-18). Gap to 2,000 real images on RSNA: −0.067 (mean of 3 archs).

Training the same LoRA longer at a normal LoRA learning rate halves the gap to real data (mean over the 3 archs, RSNA
0.900 real: v1 −0.186, v2 −0.094). Generator metrics (same protocol as above; intermediate adapters with 100 samples per class, same seeds):

| Set | KID×1000 | precision | recall | teacher p normal / pneumonia | teacher AUROC | NN sim median / max | closer than any real |
|---|---|---|---|---|---|---|---|
| real CheXpert test | ≈0 | 0.85 | 0.83 | 0.26 / 0.74 | 0.90 | 0.853 / 0.926 | 0 |
| v1 (lr 2e-5, 2,500 steps) | 122 | 0.29 | 0.06 | 0.25 / 0.45 | 0.74 | 0.811 / 0.875 | 0 |
| v2 step 2,500 | 94 | 0.35 | 0.26 | 0.10 / 0.24 | 0.81 | 0.826 / 0.885 | 0 |
| v2 step 5,000 | 179 | 0.21 | 0.18 | 0.05 / 0.14 | 0.86 | 0.827 / 0.886 | 0 |
| v2 step 7,500 (final) | 213 | 0.34 | 0.35 | 0.04 / 0.13 | 0.86 | 0.837 / 0.910 | 0 |

- More training buys class signal (teacher AUROC 0.74 → 0.86, near real 0.90) and coverage (recall 0.06 → 0.35); this
  matches the synthetic-only gain. KID gets *worse* with training (94 → 213) and the teacher's p drops for both classes:
  the samples drift in some global appearance away from real films while their class difference sharpens. KID (generic
  ImageNet features) does not predict downstream usefulness here; class signal and recall do.
- Still no memorisation (max similarity 0.910 < 0.926 between different real patients), but samples move closer to the
  training images with training (median 0.811 → 0.837): the check must be repeated for any longer-trained generator.

**Guidance sweep on the v2 adapter (2026-10-22; 100 per class, same seeds; `eval_chexpert_cfg.csv`):**

| CFG | KID×1000 | precision | recall | teacher p normal / pneumonia | teacher AUROC | NN sim max |
|---|---|---|---|---|---|---|
| real CheXpert test | ≈0 | 0.85 | 0.83 | 0.26 / 0.74 | 0.90 | 0.926 |
| 1.5 | 79 | 0.42 | 0.57 | 0.33 / 0.62 | 0.78 | 0.891 |
| **2.5** | 118 | **0.49** | 0.50 | 0.11 / 0.35 | **0.87** | 0.894 |
| 3.5 | 173 | 0.47 | 0.38 | 0.05 / 0.19 | 0.88 | 0.889 |
| 4.5 (used so far) | 213 | 0.34 | 0.35 | 0.04 / 0.13 | 0.86 | 0.910 (n = 2,000) |

Guidance trades realism and coverage against class separation: lower CFG is closer to real (KID, recall, teacher
probabilities), higher CFG separates the classes more, and drives the dark, high-contrast drift seen at 4.5. CFG 2.5 keeps
the class signal of 4.5 (0.87 vs 0.86) with half the KID gap and +0.15 precision / recall, so it dominates the old
setting. Queued (`scripts/queue_gen_v2_cfg25.sh`, tmux `c25`): 2,000 + 500 samples at CFG 2.5 and the fully synthetic
pipeline (tag SD2C25V) vs SD2V.

**CFG 2.5 full set (2026-10-22), fully synthetic pipeline (tag SD2C25V; `scripts/generator/syn_only_table.py` →
`reports/results/generator/syn_only.csv`), mean ± SD of 3 seeds, EfficientNet-B0 / ResNet-18 / DenseNet-121:**

| 2,000 training images, validation | RSNA-test (balanced) | CheXpert test |
|---|---|---|
| real, real val (CX_N2000) | 0.901 / 0.897 / 0.902 | 0.891 / 0.879 / 0.883 |
| v2 CFG 4.5, synthetic val (SD2V) | 0.814 / 0.851 / 0.835 | 0.765 / 0.772 / 0.792 |
| **v2 CFG 2.5, synthetic val (SD2C25V)** | **0.811 / 0.836 / 0.852** (SD ≤ 0.03) | **0.811 / 0.792 / 0.818** |

Generator metrics of the 2,000-image CFG 2.5 set (`eval_chexpert_c25.csv`): KID 103, precision 0.43, recall 0.62, teacher
p 0.13 / 0.37, teacher AUROC 0.82, NN similarity max 0.913 (< 0.926 real-to-real, 0 images above), distinct nearest
neighbours 45% (CFG 4.5: 27%).
- RSNA is unchanged (mean 0.833 both), CheXpert test +0.03 (0.807 vs 0.776) and seeds vary less: CFG 2.5 is the final
  generator setting. Remaining gap of the fully synthetic pipeline to 2,000 real images: RSNA −0.067, CheXpert test −0.077
  (v1 with real validation: −0.186 / −0.133).
- Teacher AUROC on 2,000 CFG 2.5 images is 0.82, on its first 200 0.87: the sweep's class-signal differences between
  CFG 2.5 / 3.5 / 4.5 (0.86–0.88) are within that sampling noise; its realism / coverage differences are not.
- **Final generator: SD-1.5 + LoRA r16 on 9,968 CheXpert train images, lr 1e-4, 7,500 steps, CFG 2.5, 30 steps.**

## 12. Scaling round (2026-10-27, last experiment batch; queue `scripts/queue_gen_scale.sh`, tmux `scale`, `logs_scale.log`)
The generator is frozen (§11). Open question for the replacement claim: is the −0.07 gap a matter of quantity, i.e. does
more synthetic data close it, and how many real images is the generator's output worth? Plus the practical use case.
- **Real reference curve on CheXpert:** CX_N250, CX_N500 (with CX_E1 = 1,000, CX_N2000) and CX_FULL = all 9,968 train
  images, i.e. exactly what the generator was trained on (the honest ceiling: replacing the data the generator consumed).
- **Synthetic scaling:** SYN_ONLY5K / SYN_ONLY10K (tag SD2C25V, synthetic validation) from `cx_sd_v2c25_10k` = 5,000 per
  class at CFG 2.5, seed 0 (indices 1–1,000 are the `cx_sd_v2c25` images, same seeds, hard-linked). 2,000 = SD2C25V.
- **Scarce real data + synthetic (prompt labels, real validation):** CX_N250_SYN2000_SD2C25 (250 real + 2,000 synthetic,
  vs CX_N250) and CX_SYN2000_SD2C25 (1,000 real + 2,000 synthetic, vs CX_E1).
- Every run keeps the shared 1,500-step budget: more images = more variety at equal compute, not more training.
- Generator evaluation of the 10k set (`eval_chexpert_10k.csv`): memorisation check over all 10,000 images.
- Table: `scripts/generator/syn_only_table.py` (EXPS extended). Estimated ~9 h (8,000 new samples at 2.4 s + 63 runs).

### §12 results (2026-10-27, queue complete; `reports/results/generator/syn_only.csv`, mean of 3 archs × 3 seeds)

| Training data | RSNA-test (balanced) | CheXpert test |
|---|---|---|
| real 250 / 500 / 1,000 / 2,000 | 0.873 / 0.892 / 0.898 / 0.900 | 0.852 / 0.859 / 0.875 / 0.884 |
| real 9,968 (= the generator's training data) | **0.931** | **0.909** |
| synthetic 2,000 / 5,000 / 10,000 (synthetic val) | 0.833 / 0.800 / 0.798 | 0.807 / 0.797 / 0.806 |
| 250 real + 2,000 synthetic (vs 250 real) | 0.857 (−0.016) | 0.836 (−0.016) |
| 1,000 real + 2,000 synthetic (vs 1,000 real) | 0.881 (−0.018) | 0.863 (−0.012) |

- **Quantity does not close the gap.** Synthetic-only plateaus at 0.80–0.83 RSNA (2k → 5k/10k −0.03, at the seed-noise edge);
  CheXpert test is flat. 2,000 synthetic images are worth fewer than 250 real ones. Gap to the generator's own training
  data (9,968 real): RSNA −0.10 to −0.13, CheXpert test −0.10.
- **Why:** synthetic validation AUROC rises with more synthetic data (best val 0.91–0.93 at 2k, 0.92–0.96 at 10k) while RSNA
  does not: the classifier separates the generator's two prompts almost perfectly, and the extra separation is
  generator-specific. Early stopping is not the limit (best step 600–1,350 of 1,500). Same mechanism as study 1, smaller.
- **Mixing** real + synthetic with prompt labels costs ≈ −0.02 (edge of noise): no benefit even with scarce real data.
- **10k generator eval** (`eval_chexpert_10k.csv`): KID 106, precision 0.44, recall 0.63, teacher AUROC 0.82 (same as the 2k
  set); NN similarity max 0.917 < 0.926 real-to-real, **0 of 10,000 above**: no memorisation at 10k either. Distinct
  nearest neighbours 21.5% of 10,000 = ~2,150 different training images.
- Report conclusion: the generator produces realistic-ish, non-memorised, class-informative images (0.80–0.83 vs chance 0.5,
  study-1 Kermany generator 0.53), but it cannot replace real data at this scale; its class signal is weaker than and partly
  different from real pathology. Experiments frozen here.

## 13. Why Google's generator worked and ours does not yet (2026-10-29)
Ktena et al. 2024 (Nature Medicine; CheXpert → ChestX-ray14), chest X-ray arm: cascaded pixel-space DDPMs (64 px → 224 px)
trained from scratch on **201k labelled CheXpert images**, conditioned on the **diagnostic label vector**, and sampled with
the **labels of the real data** (so synthetic data has the real label / co-occurrence mix); synthetic-only training gave
the best accuracy-fairness trade-off, +5.2% relative OOD AUC. Differences to ours: 20× data, full model trained on
X-rays (ours: SD-1.5 LoRA on a natural-image latent space), and rich conditioning. Our two fixed prompts merge a very mixed
"pneumonia-type" class (lung opacity 91%, effusion 41%, support devices 47%, edema 23%, cardiomegaly 12%; normals: support
devices 38%) into one text, and the classifier learns whatever separates the two prompts (§12: synthetic val 0.92–0.96,
RSNA 0.80).

Plan, one lever at a time, each judged by the fully synthetic pipeline (SYN_ONLY, synthetic val, 2,000 images):
1. **v3 = rich captions** (`queue_gen_v3.sh`, tmux `gen3`): v2 recipe, captions = view, sex, age decade and every CheXbert
   finding of the report (`prepare_chexpert.py rich`, 1,517 distinct captions over 9,968 images); sampling uses each
   class's real training captions (`sample.py --captions`). Tag SD3V vs SD2C25V.
2. **More data** (if 1 helps): CheXpert Plus has 223k images; `fetch.py` downloads ~8.6k / h (12k in 1.4 h). With rich
   captions every frontal image is usable, not only the opacity-vs-normal subset.
3. **More capacity**: LoRA rank 64–128 or full UNet fine-tune (8 GB: 8-bit Adam + gradient checkpointing).

### §13 v3 result (2026-10-29 06:44): rich captions are worse
| 2,000 synthetic, synthetic val | RSNA-test (balanced) | CheXpert test | best synthetic val | KID | teacher AUROC | teacher p normal / pneumonia |
|---|---|---|---|---|---|---|
| v2, CFG 2.5 (SD2C25V) | **0.833** | **0.807** | 0.914 | 103 | 0.82 | 0.13 / 0.37 |
| v3 rich captions, CFG 2.5 (SD3V) | 0.767 | 0.774 | 0.843 | **79** | 0.76 | 0.18 / 0.38 |

Per arch (EfficientNet-B0 / ResNet-18 / DenseNet-121) RSNA 0.780 / 0.758 / 0.763: worse for every architecture. No memorisation.
- More realistic (KID 79 vs 103) and the synthetic classes are less trivially separable (val 0.84 vs 0.91), as intended,
  but the class signal is weaker (teacher AUROC 0.76), and that is what transfers. Rich captions spread the class over many
  tokens and attributes; with a frozen CLIP text encoder and CFG 2.5 the opacity tokens may be under-weighted.
- Cheap check queued (`queue_gen_v3_cfg.sh`, tmux `v3cfg`): v3 at CFG 4.5 / 7, 100 per class, teacher signal only.

**v3 guidance check (100 per class, `eval_chexpert_v3cfg.csv`):** CFG 4.5: KID 160, P / R 0.32 / 0.22, teacher p 0.06 / 0.18,
AUROC 0.84; CFG 7: KID 277, R 0.05, p 0.05 / 0.13, AUROC 0.86. Higher guidance separates the classes more but pushes both
towards the dark, "nothing there" drift (pneumonia p falls) and collapses coverage, as for v2. **Rich captions dropped.**
Next: **capacity** (`queue_gen_r64.sh`, tmux `r64`): v2 recipe with LoRA rank 64 (alpha 64), tag SDR64V. More data needs
the user's go for the CheXpert Plus download.

**More data, prepared (2026-10-29):** `fetch.py select-gen` → 14,769 No Finding + 14,769 opacity-type frontal adult images
of CheXpert train patients outside our val / test (21,378 patients; 5,761 already on disk, 23,777 to fetch ≈ 3 h on the
box's network). Redivis OAuth expired on the box; the user signs in at the box. Then `queue_gen_big.sh` (download →
`prepare_chexpert.py big` → v2 recipe, 15,000 steps, rank from the r64 result → SYN_ONLY tag SDBIGV). Fair real reference
for that generator = a real classifier on the same 29.5k images (to add). evaluate.py's memorisation reference stays the
9,968-image train split (a lower bound for the big set).

### 13.x Rank 64 and the label/view diagnosis (10-29)
- **LoRA rank 64** (v2 recipe otherwise): SYN_ONLY RSNA view-balanced ~0.76 (EffNet 0.710, ResNet 0.730, DenseNet 0.834; seed
  spread up to ±0.055) vs 0.833 at rank 16; CheXpert test ~0.80 (same). Capacity is not the bottleneck; all-module LoRA dropped.
- **Diagnosis (comparison corrected 11-04).** CX_SYN_ONLY_TEACH (RSNA 0.921) uses the first-generation cx_sd images and real validation;
  its prompt-label pair is CX_SYN_ONLY_SD (0.714), not v2 (0.833, other images, synthetic validation). The labels matter a lot, but
  the gain may be mostly the teacher's (distillation): the Kermany RTEACH control found that for 2 of 3 archs, and no such control
  exists for this synthetic-only arm, so 0.92 does not show that the images alone are good enough. The generator's training set is
  confounded: AP view 65% of No Finding vs 86% of opacity-type images, mean age 51 vs 63, support devices 38% vs 47%. With a
  1-bit prompt the generator learns "sick" = AP/old/devices, the classifier learns that shortcut (synthetic val 0.91-0.96, worse with
  more synthetic data), and the view-balanced RSNA metric removes exactly that shortcut.
- **Test (queue_gen_view.sh, tag SDVIEWV):** v2 recipe with ", AP view" / ", PA view" appended to the caption
  (prepare_chexpert.py view), sampled 50/50 AP/PA in both classes (sample.py --balance-view). queue_gen_big.sh now also waits for it.
- **Result SDVIEWV:** RSNA view-balanced 0.814 (EffNet 0.771, ResNet 0.822, DenseNet 0.848) vs 0.833, CheXpert test 0.809 vs 0.807:
  no effect, the view shortcut is not the main failure. Teacher p(pneumonia) on synthetic "pneumonia" images: 0.37 (v2), 0.33 (view),
  0.42 (r64) vs 0.74 on real positives, so most synthetic positives show little disease: the generator under-draws the finding.
  Likely cause: the positive class mixes subtle findings (lung opacity, atelectasis, ...), so the prompt maps to a mild average.
- **Next (obvious cases + more data):** `fetch.py select-gen obvious` -> selection_obv.csv: 14,524 No Finding + 14,524 images with
  Consolidation or Pneumonia = 1 (19,993 patients, val/test patients excluded); download via scripts/fetch_gen.sh (tmux fetch).
- **Result SDOBVV (obvious cases, 29k images, 15,000 steps):** RSNA view-balanced 0.744 (EffNet 0.780, ResNet 0.748, DenseNet 0.705 ±0.108)
  vs 0.833, CheXpert test 0.761 vs 0.807: worse. KID improves (64 vs 103) but teacher p(pneumonia) on synthetic positives does not
  rise (0.31 vs 0.37; AUROC 0.82 both). So clearer positives + 3x data do not make the generator draw more disease; the "mixed
  class" explanation is not supported. Better realism did not give better classifiers. Data and clarity changed together here.
- **VAE check (scripts/generator/vae_check.py, 300 + 300 real test images, teacher scores):** original vs SD-1.5 VAE encode + decode:
  AUROC 0.902 vs 0.892, p(pneumonia) 0.734 vs 0.686, p(normal) 0.259 vs 0.227. The latent space keeps nearly all of the signal the
  teacher uses, so the loss (synthetic p 0.37) happens in the denoiser / conditioning, not the VAE. Only tests the VAE, not the SD prior.
- **Last chance (queue_gen_big.sh, tag SDBIGV, started 11-04):** v2 class definition, 29,538 images, 15,000 steps, rank 16. Pass mark:
  clearly above 0.833 on RSNA view-balanced; otherwise the generator is frozen and the report is written from these results.
- **Result SDBIGV (2026-11-04 21:36, finalised 11-12; `syn_only.csv`, `eval_chexpert_big.csv`): fails the pass mark.** RSNA
  view-balanced 0.756 (EffNet 0.774 ±0.047, ResNet 0.751 ±0.034, DenseNet 0.744 ±0.057) vs 0.833 for v2 (SD2C25V); CheXpert
  test 0.769 (0.758 / 0.760 / 0.790) vs 0.807. Worse for every architecture, by more than the seed noise. Generator metrics
  (2,000 images, CFG 2.5) vs v2: KID 82 vs 103, precision 0.54 vs 0.43, recall 0.57 vs 0.62, teacher p normal / pneumonia
  0.08 / **0.26** vs 0.13 / 0.37, teacher AUROC 0.83 vs 0.82, NN similarity max 0.912 (0 above the 0.926 real-to-real max),
  distinct nearest neighbours 46% vs 45%. Same pattern as SDOBVV (also 29k images, 15,000 steps: 0.744): 3× the data and 2×
  the steps make the samples more realistic but draw *less* disease, and the classifier gets worse. Realism (KID) and
  downstream usefulness move in opposite directions in every round since v2; teacher p(pneumonia) on synthetic positives is
  the metric that tracks the classifier. **Generator frozen at v2 (SD2C25V, RSNA 0.833).** Not run: a real-image classifier
  on the same 29.5k images (the fair ceiling for this generator); not needed for the conclusion, since SDBIGV is below its
  9,968-image predecessor.

### 13.y Final controls (2026-11-12): is synthetic-only + teacher labels the images or the teacher?
`scripts/queue_final.sh` (27 runs, `logs_final.log`), manifests from `scripts/generator/teach_final.py`. All arms: SYN_ONLY,
2,000 training images, real CheXpert validation (as CX_SYN_ONLY_TEACH), same teacher (CX_E2_ERASE seed 0: EffNet-B0 +
DenseNet-121 + ViT-S). RTEACH uses CheXpert train subset_rank 1,000–1,999 (never seen by the teacher, which trained on
ranks < 500). View-balanced RSNA-test / CheXpert test, mean of 3 archs × 3 seeds (per arch EffNet / ResNet / DenseNet):

| 2,000 training images | labels | RSNA-test (balanced) | per arch | CheXpert test |
|---|---|---|---|---|
| real (CX_N2000) | hard | 0.900 | 0.901 / 0.897 / 0.902 | 0.884 |
| real, unseen by the teacher (**CX_SYN_ONLY_RTEACH**) | teacher p | **0.940** | 0.943 / 0.936 / 0.939 | **0.908** |
| synthetic v1 (CX_SYN_ONLY_TEACH) | teacher p | 0.920 | 0.927 / 0.911 / 0.924 | 0.885 |
| synthetic final v2 CFG 2.5 (**CX_SYN_ONLY_TEACHV2**) | teacher p | 0.925 | 0.931 / 0.913 / 0.930 | 0.893 |
| synthetic final v2 CFG 2.5 (**CX_SYN_ONLY_SD2C25**) | prompt | 0.839 | 0.811 / 0.857 / 0.848 | 0.815 |
| reference: all 9,968 real (CX_FULL) | hard | 0.930 | 0.947 / 0.923 / 0.922 | 0.909 |

Seed SD 0.001–0.004 for the teacher-label arms, 0.008–0.019 for SD2C25.
- **The 0.92 is the teacher's.** Teacher labels on real images (0.940) beat teacher labels on synthetic images (0.925) for
  every architecture (−0.012 / −0.023 / −0.009, well outside the seed SD here), and even beat all 9,968 real images with hard
  labels (0.930). Distilling a 3-model teacher is the strongest single intervention of the project.
- **Generator quality barely matters once the teacher labels:** v1 → v2 adds +0.005 with teacher labels, but +0.12 with
  prompt labels (0.714 → 0.839). The teacher sees mean p 0.13 / 0.37 (normal / "pneumonia") on the v2 images vs 0.26 / 0.73 on
  real ones, so the synthetic set covers less of the decision boundary, which is the residual −0.015.
- **Synthetic validation was not the problem:** prompt labels give 0.839 with real validation vs 0.833 with synthetic validation.
- Final reading for the report: with prompt labels, our generator cannot replace real data (0.839 vs 0.900). With teacher
  labels, synthetic images carry a real-trained model's knowledge nearly as well as real images (−0.015), but that route needs
  a model trained on real patients, so it is distillation through synthetic images, not replacement of the data.
- **Google's claim on our runs (scripts/generator/mixed_fairness.py, mean of 9 runs, 1,000 real base CX_E1 RSNA 0.898, sex gap 0.011):**
  + synthetic, prompt labels 0.891 / 0.896 (E3 / E4); + 2,000 v2 synthetic 0.881; 250 real + 2,000 synthetic 0.856 vs 0.874 real alone.
  + synthetic, teacher labels 0.885 / 0.910; + real, teacher labels 0.912 / 0.917, so the best gain is the teacher's, not the images'.
  Sex gap (|AUROC M - F|) is 0.003-0.011 on RSNA in every arm, within noise: there is no gap for synthetic data to close, and only
  sex was measured. Closest positive: study 1, Kermany -> CheXpert test, teacher labels +0.05 / -0.02 / +0.04, with the §8 RTEACH caveat.
