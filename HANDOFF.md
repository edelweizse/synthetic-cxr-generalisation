# Handoff: prediction component (state as of 2026-10-10 ~12:15)

## Where things are
| What | Where |
|---|---|
| Code | `cxr/` (library), `scripts/` (CLI), committed locally in git (no remote) |
| Notebooks (executed, with explanations + insights) | `notebooks/01`–`05` (done), `06_overfitting` (see "Not done") |
| Notebook sources | `notebooks/_build/build_*.py` → regenerate; `merge_outputs.py` copies outputs into regenerated notebooks without re-running |
| Results tables / figures | `reports/results/*.csv`, `reports/figures/*.png` |
| Experiment tracking | MLflow: `mlflow.db` + `mlruns/` (laptop copy synced; live copy on GPU box). `mlflow ui --backend-store-uri sqlite:///mlflow.db` |
| Checkpoints (`best.pt`) | **only on the GPU box** (`ssh desktop`, same path `/home/edelweizse/Projects/uni/AIE/final`), not synced to the laptop |
| Optuna studies | `artifacts/optuna/optuna.db` (GPU box + laptop), best params in `configs/hpo/*.json` |
| Report draft | `docs/report.md`: methods, data, shortcut §2.1 and responsible AI are written; **results sections are empty** |
| Synthetic-data contract for teammate | `docs/SYNTHETIC_DATA.md` |
| GPU box | `ssh desktop` as `edelweizse` (key auth). `sshadmin` is **not** key-enabled (run `ssh-copy-id -i ~/.ssh/id_ed25519.pub sshadmin@desktop` yourself if needed) |

## Done
* Data: patient-level splits, near-duplicate grouping, Kermany train/test patient leak removed, B label harmonisation, EDA, shortcut analysis (metadata-only AUROC 0.998 on A-val)
* Optuna HPO for 7 architectures (25 trials each); E1/E2 × 7 archs × 3 seeds; real-data controls R25/R50, scaling N250–FULL(_AUG) for EfficientNet-B0 + ViT-S × 3 seeds
* Anti-overfitting variants (E2_ACQ, E2_ERASE, E2_ROBUST, E2_MIX, LP) for EfficientNet-B0 + ViT-S, **seeds 0–1 only**
* Notebooks 01–05 executed, with explanations and verified, results-based insights
* Placeholder E3/E4 plumbing test (tag `PLACEHOLDER`, excluded from analysis)

## Not done / open
1. **Notebook 06 (overfitting) and the remaining stage-3 runs are running detached on the GPU box** (`scripts/finish_stage3.sh`, started 12:11, log `artifacts/logs/finish_stage3.log`). It does three things in order:
   1. runs notebook 06 on seeds 0–1 (copy saved as `notebooks/06_overfitting_seeds01.ipynb`), about 10 min;
   2. trains seed 2 for EfficientNet-B0/ViT-S and all 15 DenseNet-121 variant runs (`scripts/pipeline_stage3.sh`), about 1–1.5 h;
   3. runs notebook 06 again with all seeds.

   The log ends with `finish_stage3 done` when it is finished. Then pull the results:
   ```bash
   rsync -a desktop:/home/edelweizse/Projects/uni/AIE/final/notebooks/06_overfitting.ipynb notebooks/
   scripts/sync_from_gpu.sh
   ```
   **Notebook 06 has explanations but no results-based "Insights" section yet.** Preliminary B AUROC (opacity vs normal, mean of seeds 0–1), vs E2 = EfficientNet 0.915 / ViT 0.867:
   * E2_ERASE: Eff 0.920, ViT **0.922**: best variant, especially for ViT (which attended to borders most)
   * E2_ROBUST: Eff 0.901, ViT 0.915
   * E2_ACQ: Eff 0.912, ViT 0.885
   * E2_MIX: Eff 0.890, ViT 0.890
   * LP (frozen backbone): Eff 0.780, ViT 0.795: **much worse**. Generic ImageNet features do not transfer; the network must learn CXR features

   Overfitting table already measured (E1 vs E2, selected checkpoints): train BCE ≈0.01 (train AUROC 1.000), A-val ≈0.04–0.09, Test A 0.43–0.70, B 0.51–1.36. Memorisation is contained by early stopping, but loss grows 10–17× out of distribution; E2 cuts the Test A/B loss by 25–50%.
2. (covered by the finisher above: seed 2 + DenseNet-121 for the stage-3 variants)
3. **E3/E4 with real synthetic images**: waiting for the teammate. One command after `scripts/prepare_synthetic.py` (see `docs/SYNTHETIC_DATA.md`). Then re-run notebook 04 (it picks up E3/E4 automatically) and compare against **E2 and E2_ERASE**, not just E1.
4. **Demo**: `app/streamlit_app.py` is written but never run end to end. Needs a model export: on the GPU box run `python scripts/export_model.py --exp E2 --arch efficientnet_b0`, copy `models/` to the laptop, then `.venv/bin/streamlit run app/streamlit_app.py`. Consider exporting E2_ERASE instead (best external result), selecting by A-val to stay consistent.
5. **Report results sections** (`docs/report.md` §4–5) are empty. All numbers are in notebooks 03–06 and `reports/results/`.
6. **Presentation slides** and the **roles/contribution table** required by the grading criteria are not started.
7. Final sync: run `scripts/sync_from_gpu.sh` (pulls MLflow, results and reports, not notebooks or checkpoints) and commit.

## Known caveats (keep in the report)
* A-validation is shortcut-contaminated (AUROC ≈0.99 for everything); selection uses val BCE. In-domain validation rejected E2 even though E2 is better externally.
* Pooled B AUROC is inflated by AP/PA confounding; report within-view AUROC too.
* B bootstrap CIs are tight (n = 17k); seed variance (SD 0.01–0.03) is the real uncertainty. Treat |Δ| < 0.02 as inconclusive.
* Test A behaves erratically with more training data (often decreases); treat it with care.
* The scikit-learn version affects the patient-grouped split (desktop 1.9.1 is canonical; laptop base conda differs). Use the pinned `requirements.txt`.
* The `--n-boot 200` used for the stage-3 per-run CIs is lighter than the 1000 used elsewhere (comparisons in notebooks use their own 2000-resample bootstrap).

## Update 12:18
* The finisher's **first** notebook-06 pass failed with CUDA OOM in the Grad-CAM border-share cell (batch 50), so `notebooks/06_overfitting_seeds01.ipynb` on the GPU box is a failed run. Ignore it.
* Fixed (batch 10 + cache clearing) and the fixed notebook was pushed to the GPU box *before* the finisher's final pass, which runs after the remaining training (~1–1.5 h). If that pass still fails, run `scripts/run_notebook.sh 06_overfitting` on the GPU box once the GPU is idle.

## Update 16:35 (session 3)
* The finisher **completed all training** at 13:15 (seed 2 for EfficientNet-B0/ViT-S + all DenseNet-121 variant runs). Only the final notebook-06 pass failed: CUDA OOM again, in cell 4b `border_share`, inside `gradcam` on a **DenseNet-121** forward pass (the process held 7.5 GB). Cells 1–4a ran (`reports/results/overfitting_{ladder,variants_auroc,variants_vs_e2,check}.csv` exist on the GPU box).
* **Root cause found and fixed** (commit "Grad-CAM: use autograd.grad…"): `cxr/explain.py:gradcam` leaked ~8 MB of GPU memory per call (tensor hook → closure → dict → tensor reference cycle), and ~840 calls in cell 4b reached 7 GB. It now uses `torch.autograd.grad`. Verified flat memory on DenseNet/EfficientNet/ViT. The fix is also copied to the GPU box.
* **Notebook 06 is running in tmux on the GPU box** since 16:29: `ssh desktop -t tmux attach -t nb06`. Output goes to `artifacts/logs/nb06_tmux.log`, ending with `OK <secs>` or `FAIL`, then `=== nb06 finished`.
* Next once it's OK: `rsync -a desktop:/home/edelweizse/Projects/uni/AIE/final/notebooks/06_overfitting.ipynb notebooks/` + `scripts/sync_from_gpu.sh`, write the notebook-06 Insights section (in `notebooks/_build/build_06_overfitting.py`, then `merge_outputs.py`), and commit. Then demo export (E2_ERASE), report §4–5, slides, roles table.
* Report §4.1–4.4 and §4.6 are written (from the CSVs and the notebook 03–05 insights). §4.5 (E3/E4) is a placeholder. §4.7 (anti-overfitting variants) and §5 (discussion) wait for notebook 06.

## Update 17:25 (session 3)
* **Notebook 06 done** (all seeds, OK in 465 s), pulled, results synced, Insights section written (builder + `merge_outputs.py`), committed. Headline: **E2_ERASE is best for all 3 archs** (B ≈0.92; ViT +0.035 vs E2, DenseNet +0.011, EfficientNet ≈0). Erasing does *not* consistently move Grad-CAM off the border (EfficientNet border share goes up). ACQ doesn't work; MIX best calibration; LP worst.
* **Report §4.1–4.4, 4.6, 4.7 written**; §4.5 placeholder (E3/E4); §5 Discussion still empty.
* Next: export E2_ERASE model for the demo → test Streamlit → §5 Discussion → slides + roles table.
* **Demo tested end to end (17:35).** Exported `models/{E1,E2,E2_ERASE}_efficientnet_b0` (model.pt is gitignored; it's on both machines). E2_ERASE is the default model. Added 3 RSNA examples (`app/examples/B_*.png`): a PA opacity, a PA normal, and an AP normal that E1 flags (0.98) but E2_ERASE does not (0.81 < 0.93). That makes a good live-demo story. Run: `.venv/bin/streamlit run app/streamlit_app.py` (or the `streamlit-demo` entry in `.claude/launch.json`).
* Remaining: report §5 Discussion, slides, roles/contribution table, E3/E4 (teammate).
* **Report §5 Discussion written (17:45).** The report is now complete except §4.5 (E3/E4 placeholder) and the E3/E4 parts of §5 (marked "pending").
* Remaining: slides, roles/contribution table, E3/E4 (teammate).

## Update 2026-10-12 (session 4): real synthetic data arrived
* Teammate's data downloaded on the GPU box straight from Drive with rclone (`~/.local/bin/rclone`, remote `gdrive`, read-only). Zip kept at `~/synthetic_sd/Generated_lora.zip`.
* Extracted to `data/synthetic/sd_lora/` (4,000 NORMAL + 1,000 PNEUMONIA, 512×512 RGB, `manifest.csv` with prompt/seed per image) and `data/synthetic/sd_lora_adapter/` (SD-1.5 LoRA, r=16, alpha=16, attention projections). One fixed prompt per class, 30 steps, CFG 4.5.
* **Still missing from teammate:** strict vs transfer training set, LoRA steps, any filtering (see `docs/SYNTHETIC_DATA.md` §2). Visually the pneumonia samples look much like the normal ones.
* Manifest: `artifacts/manifests/synthetic_sd_lora.csv`. **E3/E4 × 7 archs × 3 seeds running** in tmux `synth` on the GPU box (`scripts/pipeline_synth.sh`, log `artifacts/logs/pipeline_synth.log`, ends with `pipeline_synth done`). Tag `sd_lora` → runs in `artifacts/runs/E3_sd_lora`, `E4_sd_lora`.
* Next: sync results, re-run notebook 04, fill report §4.5 and the E3/E4 parts of §5; compare against E2 and E2_ERASE and against R25/R50.
* **Re-prioritised (same day):** E3/E4 paused after 4 runs (EfficientNet E3 seeds 0–2 B ≈0.81–0.83; E4 seed0 0.715, both below E1 0.859). tmux `stage3b` now runs the anti-overfitting variants for resnet18, resnet50, swin_tiny, convnext_tiny × 3 seeds (log `artifacts/logs/pipeline_stage3b.log`, ends `pipeline stage 3 done`), then automatically resumes `scripts/pipeline_synth.sh` (finished runs are skipped).
* Notebook 06 builder now covers all 7 archs (ladder plot in a 2×4 grid). After stage3b: regenerate + run notebook 06 on the GPU box, rewrite its Insights and report §4.7 for 7 archs.
* **Findings log:** `docs/FINDINGS_SYNTHETIC.md` (synthetic data + image-quality shortcut, causal degradation test, harmonised preprocessing). Probe scripts in `scripts/synth_probe/`.
* **Notebook 07 (`notebooks/07_synthetic.ipynb`)**: visual walkthrough of all synthetic-data findings (builder `notebooks/_build/build_07_synthetic.py`, runs on CPU: `CUDA_VISIBLE_DEVICES="" scripts/run_notebook.sh 07_synthetic`). Re-run it when the HARM and remaining E3/E4 runs finish; its harmonised-training cell fills in automatically.
* Harmonised runs (`harm`) and stage3b share the GPU; 7 HARM runs hit CUDA OOM. tmux `harm2` re-runs `pipeline_harm.sh` via `scripts/retry_until_done.sh` after `harm` exits. `pipeline_synth.sh` now retries too. **Check `pipeline_stage3b.log` for `FAILED` at the end** (it is not wrapped; re-run `ARCHS="..." scripts/pipeline_stage3.sh` if so).

## Update 2026-10-12 ~14:00 (session 5) — read this first
**GPU box state (all long jobs queued, nothing to babysit):**
* tmux `queue` runs `scripts/queue_remaining.sh` serially (one GPU job at a time, no OOM): harmonised runs (`pipeline_harm.sh`) → ERASE series (`pipeline_erase.sh`) → HPO re-tune for `harmonize224` + HARMT runs (`pipeline_harmt.sh`) → remaining anti-overfitting variants for resnet18/resnet50/swin/convnext (`pipeline_stage3.sh`, log `pipeline_stage3b.log`) → E3/E4 on the other archs (`pipeline_synth.sh`). Ends with `queue done` in `artifacts/logs/queue.log`. Estimated 10+ h in total.
* tmux `after` (`scripts/queue_after.sh`) waits for `queue done`, then runs aggregate + notebooks 06 and 07 (CPU). Ends with `queue_after done` in `artifacts/logs/queue_after.log`.
* Old sessions `stage3b`, `harm2`, `erase` were removed: `harm2` deadlocked because `tmux has -t harm` prefix-matches `harm2`. Use `tmux has -t =name` for exact matches. Also: never `pkill -f <pattern>` over ssh with a pattern that appears in the ssh command itself (it kills its own shell); use `pkill -f "[s]cripts/run.py"`.
* **Don't rsync `artifacts/runs/` to the laptop over mobile data** (hundreds of MB of predictions). Run analyses on the box and pull small CSVs (`reports/results/...`).

**New this session:**
* RSNA split into **RSNA-dev (30%, selection only) / RSNA-test (70%, reporting)** by patient-ID hash (`cxr/analysis.py: b_role`, column `b_role` = dev/test). Dev and test agree within 0.01 everywhere. E2_ERASE is best on RSNA-dev for all 5 archs with complete variants → base of the ERASE series.
* ERASE series (`ERASE_SYN250/500`, `ERASE_REAL250/500` in `cxr/experiments.py`; EffNet-B0, ViT-S, DenseNet-121): does synthetic data add anything on top of E2_ERASE?
* HPO re-tune under harmonised preprocessing: `scripts/hpo.py --resize-mode harmonize224` → `configs/hpo/efficientnet_b0_harmonize224.json`; `run.py --hpo-variant harmonize224`, tag `HARMT`.
* `cxr/analysis.compare` takes several filters: `"B_external:label@b_role=test&view=AP"`. RSNA-test comparisons → `reports/results/synth_probe/compare_btest.csv` (computed on the box; pull it).
* Demo: image-quality stress test panel (blur/contrast sliders, score-vs-blur curve per exported model; `cxr/quality.py`). Exported `models/E3_sd_lora_efficientnet_b0`.
* Findings doc and notebook 07 rechecked; corrected claim: augmentation only protects against **mild** blur (σ=1); at σ=2 every model goes to ≈0.9.

**Key numbers (EfficientNet-B0, RSNA-test, ensemble Δ [95% CI]):** E1→E3 −0.044 [−0.049, −0.040]; E1→E4 −0.069 (2 seeds); R25→E3 on AP films −0.156; E1→E2_ERASE +0.050. With harmonised preprocessing the synthetic penalty disappears (E1_HARM 0.834, E3_HARM 0.833, E4_HARM 0.838) but it never helps, and AP films stay worse.

**User requests still open:**
* **Use dataset names, not A/B/C**, in everything people read. Done in `docs/report.md` §1–§7 (A→Kermany, B→RSNA, Test A→Kermany test, A-validation→Kermany validation). **Still to do:** `docs/FINDINGS_SYNTHETIC.md` (B-test→RSNA-test, B-dev→RSNA-dev, "B AP" → "RSNA AP"), notebook 07 builder labels (`"B-test AUROC"` etc., only labels/markdown, not code identifiers like `B_external`), other notebook builders' plot labels.
* Report: write **§4.5 "Synthetic training data (E3/E4)"** from `docs/FINDINGS_SYNTHETIC.md` §1–6 + `compare_btest.csv`; fix §3 "RSNA is never used for any choice" (exception: RSNA-dev picked the ERASE-series base); research question wording ("generative augmentation" → "adding synthetic images"); §5 "Implications for synthetic augmentation" (prediction confirmed: the generator reproduces Kermany's quality shortcut) and objectives 3–5; §6 limitations (generator training set unknown; HARM reused squash HPO until HARMT); §7 new scripts.
* After the queue: fill ERASE / HARMT results into findings + notebook 07; rewrite notebook 06 insights and report §4.7 for 7 archs.
* Ask teammate: strict vs transfer generator training set, LoRA steps, filtering.
* **Harmonised runs complete (3 seeds each), RSNA-test seed-ensemble Δ:** equalisation does not help any baseline (E1 −0.014, E2 −0.014, E2_ERASE −0.001), but removes the synthetic penalty (E1_HARM→E3_HARM −0.010, →E4_HARM +0.002; vs −0.044 / −0.069 unharmonised). Full table on the box: `reports/results/synth_probe/compare_btest.csv`.

## Update 2026-10-12 ~17:30 (session 5, later) — read this first
**GPU box:** tmux `queue` runs `scripts/queue_tonight.sh` (log `artifacts/logs/queue_tonight.log`): dose-response
SYN1000 / SYN2000 / SYN_ONLY (EffNet-B0 × 3, tag `sd_lora`) → ERASE_SYN250/500 with `harmonize224` (tag `sd_lora_HARM`) →
ConvNeXt-T remaining variants (E2_MIX, LP) → aggregate → notebooks 07 and 06 (CPU). tmux `deadline` **kills everything at 18:35**
(user needs the workstation); the log then ends with `stopped at ...`. Old `queue` / `after` sessions are gone.

**Still to run later (all resumable, finished runs are skipped):**
* whatever `queue_tonight.sh` did not finish: just re-run it (`tmux new -d -s queue "scripts/queue_tonight.sh 2>&1 | tee -a artifacts/logs/queue_tonight.log"`).
* E3/E4 on the other 6 archs + EfficientNet E4 seed 2: `scripts/pipeline_synth.sh` (~2 h). Lower priority: the ERASE series already covers 3 archs.
* Notebook 06 insights + report §4.7 for 7 archs once ConvNeXt is complete.

**Results since the last update (all in `docs/FINDINGS_SYNTHETIC.md`, RSNA-test, ensemble Δ):**
* ERASE series (§7): synthetic images hurt on top of E2_ERASE for all 3 archs (EffNet −0.055/−0.060, ViT −0.056/−0.057,
  DenseNet −0.017/−0.047); AP films down to 0.632 for EffNet (below unaugmented E1). Extra real images ≈0. `erase_compare.csv`.
* Re-tuned harmonised preprocessing (§5, `compare_harmt.csv`, `configs/hpo/efficientnet_b0_harmonize224.json`): E1 +0.012,
  synthetic penalty returns (−0.030 / −0.038). **Corrects** the earlier "harmonising removes the penalty" claim (weak untuned baseline).
* Dose-response first run: SYN1000 seed 0 RSNA 0.832 (E1 0.860, E3 0.820, E4 0.762).

**Answer to the research question so far:** synthetic images from a generator fine-tuned on Kermany never help external
performance; they hurt in every setting tested (no augmentation, strong augmentation, harmonised, re-tuned), more images hurt
more, worst on AP films. Mechanism: the generator copies and amplifies Kermany's image-quality shortcut (§3–4).

**Open for the next session:**
* Write the dose-response result into findings (+ a plot in notebook 07: RSNA-test AUROC vs number of synthetic images, 0/250/500/1000/2000, plus SYN_ONLY).
* Report: §4.5 synthetic data section; §3 exception (RSNA-dev picked the ERASE base); research-question wording; §5 implications + objectives 3–5; §6 limitations; §7 scripts.
* Notebook 07 builder: dataset names in labels (B-test → RSNA-test etc.); add ERASE series / HARMT / dose-response figures.
* Ask teammate: strict vs transfer generator training set, LoRA steps, filtering.
* **Dose-response done (findings §8, `dose_response.csv`):** penalty plateaus at −0.04…−0.05 from 250 to 2,000 synthetic images; SYN_ONLY 0.659 on RSNA-test, chance (0.50) on PA films. Also corrected "more synthetic hurts more" (only E4, 2 noisy seeds). Still running until 18:35: ERASE_SYN with harmonize224, ConvNeXt variants, notebooks.

## Update 2026-10-13 ~19:15 (session 6) — read this first; supersedes the numbers above
**GPU box: idle, nothing queued.** All runs finished (368 runs + Swin-T qcons). Inventory of everything ran / planned / dropped:
`docs/EXPERIMENTS.md`. Findings: `docs/FINDINGS_SYNTHETIC.md` §1–§13. Notebook 07 has §9–§11 (natural quality, selection, losses).

**Corrected key numbers** (EfficientNet-B0, RSNA-test, ensemble Δ; view-balanced = mean of AP and PA AUROC, FINDINGS §13):
* E1 → E3 −0.044 (balanced −0.075); E1 → E4 **−0.065** (3 seeds; the −0.069 above was 2 seeds), balanced −0.103.
* Harmonised preprocessing does **not** remove the penalty (the HARM claim above came from an untuned baseline; HARMT −0.030 / −0.038).
* 7 architectures: 14/14 synthetic comparisons negative (−0.009…−0.067; balanced −0.019…−0.148).
* Quality-aware selection: at best ≈ E1 pooled, about half the penalty left view-balanced. Never helps.
* Losses: quality-consistency loss (qcons) E1 +0.031 (balanced +0.066), works on 4 archs, ≈ augmentation but not above it; with
  synthetic data the penalty shrinks (EffNet −0.025, Swin-T −0.009) but stays negative. Focal / group DRO small, GCE breaks.
* **View confound** (AP 76% opacity vs PA 17%): view alone scores 0.796 pooled; no conclusion flips under the view-balanced metric,
  effects get larger; +500 real images help (+0.017 balanced). Recommendation: view-balanced AUROC as the headline external metric.

**Open:** report (`docs/report.md` §4.5 placeholder, §3 RSNA-dev exception, research-question wording, §5–§7, §4.7 for 7 archs,
losses, view-balanced headline); slides; roles table; teammate questions (generator training set, LoRA steps, filtering).
Optional: re-tune hyper-parameters for qcons / GCE; view-balanced columns in notebooks 04 / 06.

## Update 2026-10-13 ~19:55 (session 7) — architecture gaps
All 371 run directories on the box are complete (every seed has `preds_B_external.csv`). The gaps are in the **design matrix**:
R25 / R50 (the matched real controls for E3 / E4) existed only for EffNet-B0 and ViT-S, qcons only for 4 / 2 archs, ERASE_SYN250 /
ERASE_REAL250 only for 3 archs. `scripts/queue_gaps.sh` (78 runs, ≈7 h, most important first) fills them; it was started at 19:40
and **stopped at 19:48 for the box shutdown**: done R25 ResNet-18 (3 seeds) and ResNet-50 (seeds 0–1); ResNet-50 seed 2 partial (resumes). Resume on the box (finished runs are skipped):

    tmux new -d -s gaps "scripts/retry_until_done.sh scripts/queue_gaps.sh logs_gaps.log 5"

Afterwards (CPU, `CUDA_VISIBLE_DEVICES=`): `scripts/synth_probe/{synth_archs,erase_compare,loss_compare,view_confound}.py`
(already extended to all archs), then rebuild + run notebook 07 (new §2 figure `synth_vs_real_archs.png`; replace the
`SYNTH_ARCHS_NOTE` placeholder with the reading) and update FINDINGS §1 / §7 / §12 / §13 and `docs/EXPERIMENTS.md`.
Notebook 04 was stale (run 10-10, before the synthetic images): its builder now uses E3_sd_lora / E4_sd_lora; re-run this session.

## 2026-10-14 — research pivot idea (not started)
Write-up: `docs/RESEARCH_PIVOT.md` ("why does synthetic augmentation hurt?": amplified shortcut vs weak class signal vs
synthetic domain / anatomy; real-image mimic arms, anatomy filter, optional CFG sweep / GAN). Deadline ~2026-11-11. Undecided.

## 2026-10-16 ~17:45 — queues PAUSED (GPU box needed by the user); how to resume
Stopped on purpose, nothing failed. All three jobs are resumable (finished runs / pairs are skipped, an interrupted run resumes from `last.pt`).

| Job | State at pause | Resume (on the box, `cd ~/Projects/uni/AIE/final`) |
|---|---|---|
| label-noise doses `queue_mimic_dose.sh` (72 runs) | EfficientNet-B0 complete (24 runs); ResNet-18 N50 E3+E4 done, N70 E3 1/3 seeds | `tmux new -d -s dose "scripts/retry_until_done.sh scripts/queue_mimic_dose.sh logs_dose.log 5"` |
| counterfactual generation `cf_generate.py` | strength 1.0 complete (250 pairs, `artifacts/manifests/synthetic_cf_lung_s10.csv`); strength 0.8 at 8/250 | `tmux new -d -s cfgen 'THREADS=4 .venv-gen/bin/python scripts/synth_cf/cf_generate.py --pairs 250 --strength 0.8 --name cf_lung_s08 >> logs_cfgen.log 2>&1'` |
| CF training `queue_cf.sh` (36 runs, CF_S10 / CF_S08) | not started (it waits for the generator and the dose queue) | start after the two above: `tmux new -d -s cfq "scripts/retry_until_done.sh scripts/queue_cf.sh logs_cfq.log 5"` |

Then: `scripts/synth_probe/pivot_compare.py` (dose arms included) and CF results vs R25/R50/sd_lora (balanced, PA especially).
Remaining GPU time at pause: about 2 h dose + 2 h generation (runs alongside) + 5 h CF training.
Done this session: report headline switched to view-balanced RSNA-test AUROC (`scripts/headline_tables.py` → `reports/results/headline_*.csv`;
report §1, §3, §4.2–4.5, §4.7, §5, §6 rewritten); EfficientNet-B0 dose results and the first CF-pair check are in `docs/RESEARCH_PIVOT.md`.
Open idea: a multi-site generator (RoentGen, trained on MIMIC-CXR) as a contrast to the Kermany-only LoRA; check weight access first.

## 2026-10-19 ~16:15 — "make synthetic data help" arms RUNNING; DiffChest downloaded (not set up)
Earlier queues (doses, CF generation, CF training) are all complete; results in report §4.5 and `docs/RESEARCH_PIVOT.md`.
Running on the box (left running when the session stopped; all resumable, finished runs are skipped):

| tmux | Job | Notes |
|---|---|---|
| `fix` | `scripts/retry_until_done.sh scripts/queue_fix.sh logs_fix.log 5`: E3/E4 × EfficientNet-B0 / ResNet-18 / DenseNet-121, arms SRC_S10 → TEACH → CTX → RTEACH → CONF → SRC_S08 (108 runs, ~5 min each) | CTX waits for `ctxgen` |
| `ctxgen` | `scripts/synth_fix/ctx_generate.py --strength 1.0 --name ctx_e1` → `logs_ctxgen.log`, 500 images (E1's own images, real lungs, redrawn context) | GPU memory capped at 0.35 of the card |
| `dcdl` | DiffChest checkpoints → `external/diffchest/checkpoints/` (2.5 GB + 133 MB + 1.6 MB; repo cloned there) | download only, nothing loaded |

Pause everything: `tmux kill-session -t fix; tmux kill-session -t ctxgen` (resume with the same commands as in the table, ctx_generate resumes too).
Results: `CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/synth_fix/fix_compare.py` → `reports/results/synth_fix/fix_compare.csv`.
First result (EfficientNet-B0, SRC_S10, Δ balanced vs E1): +250 −0.051 (CF_S10 −0.076), +500 −0.023 (CF −0.032): giving both
edits of a pair the source label reduces but does not remove the harm, so the generated lung texture hurts somewhat by itself. BUT ResNet-18 +250 SRC_S10 −0.065 (worse than CF −0.028), and SRC
is confounded: with fully regenerated lungs only the real context backs the source label (= the context shortcut); pivot §8 caveat.
DiffChest (download complete, `dl_done.txt`): the two `last.ckpt` files hold only tensors plus a pytorch_lightning
`ModelCheckpoint` callback state (checked with pickletools, nothing executed), so load them with `weights_only=True` after
registering a stub class of that name (`torch.serialization.add_safe_globals`). `latent.pkl` (weights_only OK) holds
`conds` (68,226 × 512): the semantic codes of the training images, so new adult images CAN be made without the unreleased
latent sampler (decode a stored code with fresh noise x_T, optionally pushed along the classifier's pneumonia / normal /
consolidation direction; 193 PadChest labels in `d.json`). Code needs pytorch-lightning 1.4-era APIs: write a minimal
loader that builds the model from `templates.padchest256_autoenc()` and loads `ema_model.*` weights, 256 px.
User constraint: our own generator is the subject; DiffChest is a reference point only (user agreed to check it).

## 2026-10-19 ~16:50 — CheXpert Plus slice downloading (study 2, pivot §9)
Redivis dataset `aimi.chexpert_plus:5yyj:v1_0`; the user signed in on the box (redivis OAuth, cached by the client; never read it).
`scripts/chexpert/fetch.py` done: `meta` (223,462 rows, 8 columns, no report text) + CheXbert label files, `select` →
`data/chexpert/selection.csv` (frontal, age ≥ 18, earliest study per patient; positive = Lung Opacity / Pneumonia /
Consolidation = 1 on the impression labels, negative = No Finding = 1 with no positive or uncertain opacity label;
6,000 per class; positives 86% AP vs normals 65% AP, median age 65 vs 51). `images` running in tmux `cxdl`
(log `logs_chexpert.log`, ~160 images/min, ~75 min, ~130 KB per 512-px PNG; resumable, rerun the same command).
31 selected images are in the `valid` split (PNG_valid), not downloaded: drop them.
Next: a Kermany-style manifest (`artifacts/manifests/chexpert.csv`: patient-level train/val/test, `subset_rank`), make
`cxr/experiments.py` take the source manifest as a parameter (study 2 tags), HPO or reuse, then E1/E2/E2_ERASE/R25/R50;
generator: needs the teammate's SD-1.5 LoRA training script (not in the repo).

## 2026-10-19 18:50 — GPU box STOPPED by the user
`fix` queue killed after SRC_S10, TEACH, CTX, RTEACH (72 runs done); CONF and SRC_S08 (36 runs) left. Resume:
`tmux new -d -s fix "scripts/retry_until_done.sh scripts/queue_fix.sh logs_fix.log 5"` (finished runs are skipped).
CTX generation done (500 images). CheXpert download done: 11,969 images in `data/chexpert/images` (31 valid-split ones skipped).
Results of the finished arms: pivot §8 results (TEACH removes the penalty in 6/6 cells).

## 2026-10-20 11:05 — box running again
tmux `fix` resumed (CONF, SRC_S08); tmux `cx` waits for it, then runs study 2 baselines (`scripts/queue_cx.sh`, log
`logs_cx.log`). CheXpert manifests: `artifacts/manifests/chexpert{,_vb}.csv` (pivot §9 status). Sync note: the `desktop`
ssh alias needs `-o HostName=<tailnet-ip>`; for `scripts/sync_to_gpu.sh` export `RSYNC_RSH="ssh -o HostName=<tailnet-ip>"`.

## 2026-10-20 11:30 — queue dashboard
`scripts/dashboard/server.py` (stdlib, read-only) runs in tmux `dash` on the box: http://<tailnet-ip>:8765 from any tailnet
device. Shows the current run + live step, each queue's arms with why / progress dots / Δ vs reference (seed ensemble, same
point estimates as the report), ETA, and what comes after. Its PLAN mirrors queue_fix.sh / queue_cx.sh: update both together.
Restart: `tmux new -d -s dash ".venv/bin/python scripts/dashboard/server.py 2>&1 | tee logs_dash.log"`.
`run.py` is now line-buffered so queue logs (and the live step) are current.

## 2026-10-20 13:35 — generator queue
Friend's LoRA script -> `scripts/generator/train_lora.py`; training set prepared (`data/generator/chexpert`, 9,968 images).
tmux `gen` waits for `cx`, then `scripts/retry_until_done.sh scripts/queue_gen_cx.sh logs_gen.log 4`: LoRA training
(~1-1.5 h) -> 2,000 samples (~2 h) -> teacher labels -> 63 classifier runs. Every stage skips when its output exists.
The dashboard shows the generator stages (queue "gen"). Details: pivot §9 "generator on CheXpert".

## 2026-10-20 16:40 — LoRA OOM fixed; goal restated
queue_gen_cx failed 4× with CUDA OOM at step 2 (7.3 GB). Fix: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (no
recipe change); now 1.1 s/step, ~45 min. Added CX_SYN_ONLY_SD / CX_SYN_ONLY_TEACH (synthetic only vs CX_N2000) as the main
test, per the restated goal (pivot §10). tmux `gen` restarted.

## 2026-11-12 — experiments finalised; catalogue; presentation (read this first)
Everything after 10-20 lives in `docs/RESEARCH_PIVOT.md` §9–13 (study 2, generator rounds). New one-stop index of every
experiment: **`docs/CATALOGUE.md`** (descriptions + generated run tables; regenerate with `scripts/catalogue.py`, which joins
run dirs with `mlflow.db` and recomputes view-balanced RSNA-test → `reports/results/catalogue_{runs,summary}.csv`).
* SDBIGV (29.5k images, 15k steps) finalised: RSNA 0.756 vs 0.833 (v2) → **generator frozen at v2 CFG 2.5**. `syn_only.csv` regenerated.
* Final controls (`scripts/queue_final.sh`, tmux `final`, `logs_final.log`; manifests `scripts/generator/teach_final.py`):
  RTEACH (2,000 unseen real CheXpert images, teacher labels), TEACHV2 (final generator images, teacher labels), SD2C25
  (final images, prompt labels, real val). Results: RESEARCH_PIVOT §13.x and CATALOGUE §9.
* Defence deck: claude.ai artifact (Slides), 22 slides, thesis structure, speaker notes carry the script.
* Still open: report.md has no study 2 yet; roles / contribution table; teammate questions.

## 2026-11-13 — submission materials (grading criteria check)
* Report: `docs/report/report.tex` (LaTeX, both studies, related work, optimisation, risks, contributions, AI tools; figures in
  `docs/report/figures/`, made by `scripts/report_figures.py`). Built on the laptop with tectonic 0.17 (`tectonic -X compile report.tex`); PDF committed.
  Red `[...]` markers = items the team must confirm (contribution mapping, AI-tool wording).
* Optimisation: `scripts/optimize_bench.py` -> `reports/results/optimize_bench.csv` (bf16 1.9-2.5x throughput, AUROC within 0.0006;
  ViT-S int8 -73% size, -31% CPU latency, -0.004 AUROC; 3-seed ensemble +0.007..+0.029 at 3x cost).
* Demo: CheXpert model exported (`models/CX_E2_ERASE_efficientnet_b0`, seed 2 by source val loss) and is the default; the app shows
  the training data of each model; tested headless (PA opacity 0.995, AP normal 0.002).
* README rewritten for both studies; `requirements-gen.txt`; `docs/CONTRIBUTIONS.md` (contribution table, borrowed components, AI tools).
* Deck: 10 main slides + 12 appendix slides (claude.ai artifact).
