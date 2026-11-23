"""Optuna hyper-parameter search for one architecture on the E1 setting.

Objective: best validation BCE on A-validation (never B). TPE sampler + median pruning on the
validation-loss curve. The study lives in artifacts/optuna/optuna.db, so an interrupted search
resumes where it stopped; the best parameters are written to configs/hpo/<arch>.json.

    python scripts/hpo.py --arch convnext_tiny --trials 30
    python scripts/hpo.py --arch efficientnet_b0 --resize-mode harmonize224   # -> configs/hpo/efficientnet_b0_harmonize224.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optuna
import torch
from optuna.trial import TrialState

from cxr import PROJECT_ROOT
from cxr.engine import train_run
from cxr.experiments import HPO_DIR, make_cfg
from cxr.models import DROP_PATH

STORAGE = f"sqlite:///{PROJECT_ROOT / 'artifacts/optuna/optuna.db'}"


def search_space(trial: optuna.Trial, arch: str) -> dict:
    p = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 0.2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "schedule": trial.suggest_categorical("schedule", ["cosine", "plateau"]),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 200, step=50),
        "head_lr_mult": trial.suggest_categorical("head_lr_mult", [1.0, 10.0]),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.1),
        "drop_rate": trial.suggest_float("drop_rate", 0.0, 0.5),
    }
    if arch in DROP_PATH:
        p["drop_path_rate"] = trial.suggest_float("drop_path_rate", 0.0, 0.3)
    return p


def objective(trial: optuna.Trial, arch: str, resize_mode: str | None = None) -> float:
    params = search_space(trial, arch)
    key = f"{arch}_{resize_mode}" if resize_mode else arch
    cfg = make_cfg("E1", arch, overrides={**params, "name": f"hpo/{key}", **({"resize_mode": resize_mode} if resize_mode else {})})
    rd = PROJECT_ROOT / "artifacts/optuna/trials" / key / f"trial{trial.number:03d}"
    try:
        res = train_run(cfg, seed=0, run_dir=rd, trial=trial, save_model=False, mlflow_experiment="cxr-hpo",
                        tags={"arch": arch, "trial": str(trial.number), "stage": "hpo"}, verbose=False)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise optuna.TrialPruned("CUDA OOM")
    trial.set_user_attr("best_val_auroc", res["best_val_auroc"])
    trial.set_user_attr("best_step", res["best_step"])
    trial.set_user_attr("train_time_s", res["train_time_s"])
    return res["best_val_loss"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--trials", type=int, default=30, help="target number of finished (complete+pruned) trials")
    ap.add_argument("--resize-mode", default=None, help="search under other preprocessing, e.g. harmonize224")
    args = ap.parse_args()
    key = f"{args.arch}_{args.resize_mode}" if args.resize_mode else args.arch

    (PROJECT_ROOT / "artifacts/optuna").mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"hpo_{key}", storage=STORAGE, load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0, multivariate=True, n_startup_trials=8),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=300))
    # trials left RUNNING by a crash/interruption are marked failed, then the search continues
    for t in study.get_trials(states=(TrialState.RUNNING,)):
        study._storage.set_trial_state_values(t._trial_id, TrialState.FAIL)
    finished = len(study.get_trials(states=(TrialState.COMPLETE, TrialState.PRUNED)))
    remaining = max(0, args.trials - finished)
    print(f"[hpo {key}] {finished} finished trials, running {remaining} more", flush=True)
    t0 = time.time()
    study.optimize(lambda t: objective(t, args.arch, args.resize_mode), n_trials=remaining, gc_after_trial=True,
                   callbacks=[lambda s, t: print(f"  trial {t.number:3d} {t.state.name:8s} "
                                                  f"val_loss={t.value if t.value is not None else float('nan'):.4f} "
                                                  f"best={s.best_value:.4f} ({time.time() - t0:.0f}s)", flush=True)])
    best = study.best_trial
    HPO_DIR.mkdir(parents=True, exist_ok=True)
    out = {"arch": args.arch, "resize_mode": args.resize_mode or "squash", "params": best.params, "best_val_loss": best.value,
           "best_val_auroc": best.user_attrs.get("best_val_auroc"), "trial": best.number,
           "n_complete": len(study.get_trials(states=(TrialState.COMPLETE,))),
           "n_pruned": len(study.get_trials(states=(TrialState.PRUNED,)))}
    (HPO_DIR / f"{key}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
