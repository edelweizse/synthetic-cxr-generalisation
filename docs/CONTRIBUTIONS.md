# Contributions, borrowed components and AI tools

Team AAI-2605M: **Trubarov Nikita** (prediction component) and **Yerlan Kubesov** (generative component).

## 1. Individual contributions

| Work package | Member | Evidence (files, experiments, report sections) |
|---|---|---|
| Problem framing, research questions, success criteria | Trubarov Nikita | report §1; slides 01–02 |
| Data pipeline: manifests, patient-level splits, near-duplicate grouping, Kermany train/test leak removal, RSNA dev/test split, CheXpert selection and download | Trubarov Nikita | `cxr/data.py`, `cxr/imaging.py`, `scripts/chexpert/`, notebook 01; report §3; slide 03 |
| Shortcut analyses (Kermany acquisition shortcut, RSNA view confound) | Trubarov Nikita | notebook 01, `scripts/synth_probe/view_confound.py`; report §3; slide A1 |
| Classifier engine, hyper-parameter search, 7 architectures, augmentation variants | Trubarov Nikita | `cxr/engine.py`, `cxr/models.py`, `cxr/augment.py`, `scripts/hpo.py`, `scripts/run.py`, `configs/hpo/`; notebooks 02, 03, 06; report §4.1, §6.1 |
| Study 1 generator: Stable Diffusion 1.5 LoRA training script and recipe, the Kermany adapter, 5,000 synthetic images | Yerlan Kubesov | `scripts/generator/train_lora.py` (original script), `data/synthetic/sd_lora/`, `data/synthetic/sd_lora_adapter/`; report §4.2; slide 04 |
| Study 1 synthetic experiments and mechanism arms (mimic arms, anatomy filter, counterfactual pairs, teacher labels) | Trubarov Nikita | `scripts/synth_probe/`, `scripts/synth_cf/`, `scripts/synth_fix/`, `docs/FINDINGS_SYNTHETIC.md`, `docs/RESEARCH_PIVOT.md` §1–8; report §6.2–6.3; slides 05, A2 |
| Study 2 generator: retraining on CheXpert with the LoRA script, v2 recipe, guidance sweep, evaluation (KID, precision/recall, class signal, memorisation), later variants | Trubarov Nikita | `scripts/generator/{prepare_chexpert,sample,evaluate,vae_check}.py`, `scripts/queue_gen_*.sh`, `reports/results/generator/`; report §4.2, §6.5; slides 04, A4, A5 |
| Study 2 classifier experiments: baselines, scaling, final controls, comparison with Ktena et al. | Trubarov Nikita | `scripts/queue_cx.sh`, `scripts/queue_gen_scale.sh`, `scripts/queue_final.sh`, `scripts/generator/{mixed_fairness,teach_final,syn_only_table}.py`; report §6.4–6.7; slides 06–07, A3 |
| Optimisation and resource measurements | Trubarov Nikita | `scripts/optimize_bench.py`, `reports/results/optimize_bench.csv`; report §7 |
| Demo | Trubarov Nikita | `app/streamlit_app.py`, `scripts/export_model.py`, `models/` |
| Report, slides, experiment catalogue | Trubarov Nikita | `docs/report/`, presentation, `docs/CATALOGUE.md` |

## 2. Borrowed components

| Component | Role in the project | What we changed / how we checked it |
|---|---|---|
| Stable Diffusion 1.5 (base weights, via `diffusers`) | frozen base of both generators | LoRA adapters trained on our data; outputs checked by KID, precision/recall, teacher signal and memorisation (`scripts/generator/evaluate.py`) |
| LoRA training script (`scripts/generator/train_lora.py`) | generator training | written for the generative component (Yerlan Kubesov); vendored with three changes (fp16 base weights, `--seed`, model id); recipe changes documented in report §4.2 |
| timm ImageNet backbones | classifier initialisation | one-logit head; all training ours; ImageNet-only weights to avoid NIH/RSNA contamination |
| torchxrayvision PSPNet segmenter | anatomy scoring of synthetic images (one analysis) | used as a measurement tool; scores sanity-checked against real images |
| PyTorch, diffusers, PEFT, transformers, Optuna, MLflow, scikit-learn, pandas, Streamlit | libraries | versions pinned in `requirements.txt` / `requirements-gen.txt` |
| Datasets: Kermany, RSNA, CheXpert Plus | data | used under their licences / agreements; no images redistributed |

## 3. AI tools

| Tool | Used by / for | How results were checked |
|---|---|---|
| Claude (Anthropic), via Claude Code | prediction component: writing and refactoring scripts; orchestrating experiment queues on the GPU workstation; computing result tables; drafting documentation, the report, the slides and the teammate update | every reported number is recomputed by scripts from saved per-run predictions (`scripts/catalogue.py`, `*_compare.py`) and is traceable to MLflow runs and `artifacts/runs/`; controls were added when a result looked too good (e.g. real images with teacher labels as the control for synthetic images with teacher labels); claims were withdrawn when controls contradicted them (e.g. "harmonisation removes the penalty"); code and text were reviewed by the team before submission |

The Kermany generator (study 1) was produced with the original LoRA training script; no AI-assistant use was reported for it.
