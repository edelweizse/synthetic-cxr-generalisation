# Can a generator replace real chest X-rays?

Synthetic training data for cross-hospital pneumonia classification. A pneumonia classifier is trained on one hospital's chest X-rays and tested on another hospital it never sees. We measure whether images from a Stable Diffusion 1.5 + LoRA generator help (as augmentation) or can replace real images, and why Google's positive result (Ktena et al., *Nature Medicine* 2024) does not carry over at our scale.

> Research prototype. The models are not medical devices and must not be used for diagnosis.

Team AAI-2605M: Trubarov Nikita (prediction component), Yerlan Kubesov (generative component). Defence version: git tag `defense-v1`.

## Submission contents

| Required item | Where |
|---|---|
| Report (LaTeX PDF + sources): task and success criteria, related work, data, baseline and method, experiment protocol, results and errors, optimisation, risks, conclusions, contributions | `docs/report/report.pdf`, `docs/report/report.tex`, `docs/report/figures/` |
| Source code, README, dependencies and versions | this repository; `requirements.txt` (classifiers, demo), `requirements-gen.txt` (generator) |
| Configurations, prompts and generation parameters | `configs/hpo/`, `cxr/experiments.py`; generator recipe and prompts below and in `scripts/generator/sample.py`, `scripts/queue_gen_*.sh` |
| How to repeat the main experiment | [Reproducing the experiments](#reproducing-the-experiments) |
| Data links and how to obtain them; restricted data and anonymised example | [Data](#data), `docs/DATA.md`, `docs/data_examples/` |
| Working prototype | `app/streamlit_app.py`, `models/`; [Demo](#demo) |
| Personal contribution table with links to files, experiments and report sections | `docs/CONTRIBUTIONS.md` §1; report §11 |
| Borrowed components and AI tools, their role and how results were checked | `docs/CONTRIBUTIONS.md` §2–3; report Appendix A |
| Presentation | submitted separately (PDF) |
| Every experiment in one place | `docs/CATALOGUE.md` |

## Results in one table

View-balanced AUROC on RSNA-test (mean of AUROC within AP and within PA films), mean of 3 seeds × 3 architectures unless noted.

| Setting | RSNA-test |
|---|---|
| Kermany (children, 1 hospital), 1,000 real images, EfficientNet-B0 | 0.736 |
| … + 250 synthetic images (Kermany generator) | 0.676 (worse in 14 of 14 comparisons) |
| CheXpert (adults), 250 / 2,000 / 9,968 real images | 0.873 / 0.900 / 0.930 |
| CheXpert, 2,000 synthetic images only, final generator | 0.833 |
| … same synthetic images, labels from a real-trained teacher | 0.925 |
| CheXpert, 2,000 real images, teacher labels (distillation control) | 0.940 |

## Data

| | Dataset | Role | How to get it |
|---|---|---|---|
| Kermany | [Guangzhou paediatric CXR](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), 5,856 images, CC BY 4.0 | study-1 source | Kaggle download → `data/kaggle/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/` |
| RSNA | [RSNA Pneumonia Detection Challenge](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018) (NIH ChestX-ray14 subset), adjudicated MD.ai labels | **external test only** (30 % dev / 70 % test by patient hash) | `data/pneumonia-challenge-annotations-adjudicated-kaggle_2018.json` + DICOMs in `data/rsna/` |
| CheXpert | CheXpert Plus, Stanford AIMI | study-2 source and generator training set | requires accepting the Stanford AIMI research-use agreement; then `python scripts/chexpert/fetch.py meta`, `select`, `images` (Redivis client, your own sign-in). Images may not be redistributed. |

Access restrictions, what is and isn't shared, and an anonymised example: **`docs/DATA.md`**.

Key data decisions: patient-level splits, removal of 170 patients shared between the official Kermany train and test folders, near-duplicate grouping, full-field-of-view resizing (Kermany's file size alone predicts the label: AUROC 0.998), and the view-balanced external metric (the view alone scores 0.80 on RSNA). Details: report §3, notebook 01.

## Environments

```bash
# classifiers, analysis, demo (Python 3.12, CUDA 12.8)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple

# generator (Stable Diffusion 1.5 + LoRA), separate env
uv venv --python 3.12 .venv-gen
uv pip install --python .venv-gen -r requirements-gen.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
```

Hardware used for every result: one RTX 3060 Ti (8 GB). CPU is enough for the demo.

## Reproducing the experiments

All queues are resumable and idempotent: finished runs are skipped, interrupted runs resume from `last.pt`.

```bash
# Study 1: Kermany source
jupyter nbconvert --execute --inplace notebooks/01_data.ipynb   # manifests, splits, caches
scripts/pipeline.sh                                             # Optuna HPO per architecture -> E1/E2 x 3 seeds
scripts/pipeline_stage2.sh                                      # real-data controls and scaling
scripts/pipeline_stage3.sh                                      # anti-overfitting variants (E2_ERASE etc.)
python scripts/prepare_synthetic.py data/synthetic/sd_lora --name sd_lora
scripts/pipeline_synth.sh                                       # E3/E4 with synthetic images

# Study 2: CheXpert source and our generator
python scripts/chexpert/make_manifest.py
scripts/queue_cx.sh                    # CheXpert baselines
scripts/queue_gen_cx.sh                # generator v1: LoRA training, sampling, teacher labels, classifier arms
scripts/queue_gen_v2.sh && scripts/queue_gen_v2_cfg25.sh   # generator v2 and the guidance setting
scripts/queue_gen_scale.sh             # scaling: real curve, 5k/10k synthetic, mixed
scripts/queue_final.sh                 # final controls (teacher labels on real vs synthetic)

# Tables, figures, resources
python scripts/catalogue.py                     # all runs (joins mlflow.db) -> reports/results/catalogue_*
python scripts/generator/syn_only_table.py      # synthetic-only table
python scripts/generator/evaluate.py chexpert cx_sd_v2c25   # KID, precision/recall, class signal, memorisation
python scripts/optimize_bench.py                # latency, memory, bf16 and int8 trade-offs
python scripts/report_figures.py                # figures for docs/report/
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The generator recipe (final): SD 1.5 + LoRA r16 / alpha 16 on the U-Net attention projections, 9,968 CheXpert training images at 512 px, one caption per class, lr 1e-4, 7,500 steps × batch 4, sampling with 30 steps, CFG 2.5, fixed seeds. Prompts: `"a frontal cxr_scan showing pneumonia, lung consolidation, radiograph"` and `"a frontal cxr_scan of normal lungs, clear lung fields, radiograph"`; negative prompt in `scripts/generator/sample.py`.

## Demo

```bash
python scripts/export_model.py --exp CX_E2_ERASE --arch efficientnet_b0   # seed chosen by source-validation loss
streamlit run app/streamlit_app.py
```

Upload a chest X-ray (PNG, JPEG or DICOM) or pick an example. The default model is trained on CheXpert; Kermany models are selectable in the sidebar, which shows each model's training data and test metrics. Main scenario: `B_PA_opacity_F52.png` (RSNA, adult PA opacity: score 0.995, above the 0.566 threshold). Hard case: `B_AP_normal_F35.png`, a normal portable AP film: the CheXpert model scores 0.002, while the Kermany baseline `E1_efficientnet_b0` flags it (0.98) because it learned acquisition cues. The stress-test panel shows how the score reacts to blur and contrast loss.

## Layout

```
cxr/            library: imaging, data (manifests, caches), models, augment, engine, metrics, analysis, explain, tracking
scripts/        run.py, hpo.py, catalogue.py, optimize_bench.py, export_model.py, queues (queue_*.sh, pipeline*.sh)
scripts/generator/   train_lora.py (generator training), sample.py, evaluate.py, teacher-label manifests, tables
scripts/synth_probe/, synth_cf/, synth_fix/   mechanism experiments (study 1)
notebooks/      01 data · 02 HPO · 03 architectures · 04 data experiments · 05 errors · 06 overfitting · 07 synthetic
docs/           report/ (LaTeX), CATALOGUE.md, RESEARCH_PIVOT.md (lab notebook), FINDINGS_SYNTHETIC.md, CONTRIBUTIONS.md
reports/        results/ (CSV tables), figures/
app/            Streamlit demo
```
