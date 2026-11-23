"""Training, prediction and evaluation.

* Budget is expressed in optimizer steps (`max_steps`), so runs whose training sets differ in size
  (real-only vs real+synthetic) get the same maximum optimisation budget.
* Validation loss (BCE, a strictly proper scoring rule) drives checkpoint selection, early stopping,
  the plateau LR schedule and Optuna. Validation AUROC on dataset A saturates near 0.99 and is
  therefore too coarse to rank checkpoints or hyper-parameters; it is still logged.
* Runs are resumable: `last.pt` holds model/optimizer/scheduler/RNG/early-stopping state and is
  written at every evaluation; re-invoking `train_run` with the same run directory continues it.
* Reproducibility: seeded CPU/CUDA generators for sampling and augmentation, deterministic cuDNN,
  and the software/hardware environment recorded with every run.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; must precede CUDA init

import json
import math
import platform
import random
import socket
import subprocess
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.entities import Metric
from sklearn.metrics import roc_auc_score

from . import PROJECT_ROOT, tracking
from .augment import PRESETS, augment_batch
from .data import load_set, resolve
from .metrics import all_metrics, bootstrap_ci, choose_threshold
from .models import Classifier

RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"

DEFAULTS = dict(
    arch="convnext_tiny", pretrained=True, image_size=224, resize_mode="squash",
    augment="none", aug_params=None,
    batch_size=32, max_steps=2000, eval_every=50,
    lr=1e-4, weight_decay=0.05, head_lr_mult=1.0, warmup_steps=100,
    schedule="cosine", min_lr_ratio=0.01, plateau_factor=0.3, plateau_patience=3,
    early_stopping_patience=8, min_delta=1e-4,
    label_smoothing=0.0, drop_rate=0.0, drop_path_rate=0.0, grad_clip=1.0,
    mixup_alpha=0.0, freeze_backbone=False,
    loss="bce", loss_params=None,  # see LOSSES
    amp=True, deterministic=True, threshold="youden", eval_batch_size=128,
)


# Training losses. All reduce to BCE on the clean label at the evaluation side: validation, checkpointing and
# early stopping always use plain BCE, so only the training signal differs.
#   bce    binary cross-entropy (default)
#   focal  focal loss (Lin et al. 2017): down-weights examples the model already gets right
#   gce    generalised cross-entropy (Zhang & Sabuncu 2018): bounded, robust to noisy labels (synthetic labels are noisy)
#   qdro   online group DRO (Sagawa et al. 2020) over label x image-sharpness groups (sharper / softer half of the
#          training set): up-weights the groups where "soft = pneumonia" fails (sharp pneumonia, soft normals)
#   qcons  BCE + consistency: penalise the squared logit change under a random blur + contrast reduction, i.e.
#          explicitly ask the score not to depend on image quality (cf. the causal test, FINDINGS_SYNTHETIC §4)
LOSSES = {"bce": {}, "focal": {"gamma": 2.0}, "gce": {"q": 0.7}, "qdro": {"eta": 0.01},
          "qcons": {"lam": 1.0, "sigma": (0.5, 2.0), "contrast": (0.6, 1.0)}}


def sharpness(x: torch.Tensor) -> torch.Tensor:
    """Mean absolute high-frequency residual per image (x: N,1,H,W in [0, 1]); the statistic of FINDINGS_SYNTHETIC §3."""
    return (x - torch.nn.functional.avg_pool2d(x, 5, 1, 2)).abs().mean((1, 2, 3))


def degrade_batch(x: torch.Tensor, sigma: tuple, contrast: tuple, g: torch.Generator) -> torch.Tensor:
    """One Gaussian blur sigma per batch (separable conv) and a per-image contrast factor."""
    s = float(sigma[0] + (sigma[1] - sigma[0]) * torch.rand(1, generator=g, device=g.device))
    k = int(2 * round(3 * s) + 1)
    t = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    w = torch.exp(-t ** 2 / (2 * s ** 2)); w = w / w.sum()
    F = torch.nn.functional
    y = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), w.view(1, 1, 1, k))
    y = F.conv2d(F.pad(y, (0, 0, k // 2, k // 2), mode="reflect"), w.view(1, 1, k, 1))
    f = contrast[0] + (contrast[1] - contrast[0]) * torch.rand(len(x), 1, 1, 1, generator=g, device=g.device)
    m = y.mean((2, 3), keepdim=True)
    return (m + f * (y - m)).clamp(0, 1)


# ----------------------------------------------------------------------------- utilities

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def environment() -> dict:
    import timm

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                                text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""
    env = {"host": socket.gethostname(), "python": platform.python_version(), "torch": torch.__version__,
           "timm": timm.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
           "git_commit": commit}
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    return env


def _autocast(device, enabled=True):
    if device.type == "cuda" and enabled:
        return torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    return torch.autocast(device.type, enabled=False)


@torch.no_grad()
def predict_logits(model, tset, device, batch_size=128, amp=True) -> np.ndarray:
    model.eval()
    out = []
    for x in tset.batches(batch_size, device):
        with _autocast(device, amp):
            out.append(model(x.contiguous(memory_format=torch.channels_last)).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, dtype=np.float32)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _bce(logits, y):
    z = torch.from_numpy(logits)
    return float(torch.nn.functional.binary_cross_entropy_with_logits(z, torch.from_numpy(y.astype(np.float32))))


def _lr_factor(cfg, step, plateau_scale):
    warm = cfg["warmup_steps"]
    if step < warm:
        return (step + 1) / warm
    if cfg["schedule"] == "cosine":
        t = (step - warm) / max(1, cfg["max_steps"] - warm)
        return cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    if cfg["schedule"] == "plateau":
        return plateau_scale
    if cfg["schedule"] == "constant":
        return 1.0
    raise ValueError(cfg["schedule"])


def build_train_tensors(cfg, device):
    """Concatenate all training sources (e.g. real A subset + synthetic) into on-device tensors."""
    xs, ys, dfs = [], [], []
    for src in cfg["train"]:
        ts = load_set(src["manifest"], src.get("where"), cfg["image_size"], cfg["resize_mode"], device=device)
        xs.append(ts.images)
        ys.append(ts.labels)
        dfs.append(ts.df.assign(train_source=src.get("tag", Path(src["manifest"]).stem)))
    return torch.cat(xs), torch.cat(ys), pd.concat(dfs, ignore_index=True)


# ----------------------------------------------------------------------------- training

def train_run(cfg: dict, seed: int, run_dir: Path | None = None, *, trial=None, save_model: bool = True,
              track: bool = True, mlflow_experiment: str | None = None, tags: dict | None = None,
              verbose: bool = True) -> dict:
    """Train one model from the pretrained initialisation (resuming if `last.pt` exists).

    trial      optional optuna.Trial: val loss is reported at each evaluation and pruning honoured.
    save_model write best.pt / last.pt (disable for HPO trials).
    Returns a summary dict (also stored as run.json in run_dir).
    """
    cfg = {**DEFAULTS, **cfg}
    run_dir = Path(run_dir or RUNS_DIR / cfg["name"] / f"seed{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    done = run_dir / "run.json"
    if done.exists() and json.loads(done.read_text()).get("status") == "complete":
        if verbose:
            print(f"[skip] {run_dir.relative_to(PROJECT_ROOT)} already complete")
        return json.loads(done.read_text())

    device = get_device()
    seed_everything(seed, cfg["deterministic"])
    ckpt_path = run_dir / "last.pt"
    resume = torch.load(ckpt_path, map_location="cpu", weights_only=False) if save_model and ckpt_path.exists() else None

    run_id = resume.get("mlflow_run_id") if resume else None
    if track:
        tracking.setup(mlflow_experiment or cfg.get("mlflow_experiment", tracking.EXPERIMENT))
        mlflow.start_run(run_id=run_id, run_name=None if run_id else f"{cfg['name']}-seed{seed}",
                         tags=None if run_id else {**(tags or {}), "seed": str(seed), "arch": cfg["arch"]})
    try:
        return _train(cfg, seed, run_dir, device, resume, trial, save_model, track, verbose)
    finally:
        if track:
            mlflow.end_run(status="FINISHED" if done.exists() else "KILLED")


def _train(cfg, seed, run_dir, device, resume, trial, save_model, track, verbose):
    t_start = time.time()
    x_tr, y_tr, train_df = build_train_tensors(cfg, device)
    val = load_set(cfg["val"]["manifest"], cfg["val"].get("where"), cfg["image_size"], cfg["resize_mode"],
                   device=device)
    y_val = val.labels.cpu().numpy()
    n = len(y_tr)

    model = Classifier(cfg["arch"], cfg["pretrained"], cfg["drop_rate"], cfg["drop_path_rate"])
    model = model.to(device).to(memory_format=torch.channels_last)
    if cfg.get("grad_checkpointing"):  # memory only (activations recomputed, RNG replayed): same updates
        model.net.set_grad_checkpointing(True)
    if cfg["freeze_backbone"]:  # linear probe: only the new classification head is trained
        clf = model.net.get_classifier()
        torch.nn.init.zeros_(clf.weight)  # timm's head init can be large (EfficientNet: fan-out=1)
        torch.nn.init.zeros_(clf.bias)
        head = {id(p) for p in clf.parameters()}
        for p in model.net.parameters():
            p.requires_grad_(id(p) in head)
    groups = model.param_groups(cfg["lr"], cfg["weight_decay"], cfg["head_lr_mult"])
    base_lrs = [g["lr"] for g in groups]
    opt = torch.optim.AdamW(groups, fused=device.type == "cuda")
    aug_params = {**PRESETS[cfg["augment"]], **(cfg["aug_params"] or {})} if cfg["augment"] != "none" else None

    g_sample = torch.Generator().manual_seed(seed)                    # batch order
    g_aug = torch.Generator(device=device).manual_seed(seed + 10_000)  # augmentation parameters
    st = {"step": 0, "perm": torch.randperm(n, generator=g_sample), "pos": 0, "history": [],
          "best_loss": math.inf, "best_auroc": float("nan"), "best_step": 0, "bad_evals": 0,
          "plateau_scale": 1.0, "plateau_bad": 0, "plateau_best": math.inf, "stopped_early": False,
          "running": []}
    best_state = None
    if resume:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["opt"])
        st.update(resume["state"])
        g_sample.set_state(resume["g_sample"])
        g_aug.set_state(resume["g_aug"])
        torch.set_rng_state(resume["torch_rng"])
        if torch.cuda.is_available() and resume.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(resume["cuda_rng"])
        if (run_dir / "best.pt").exists():
            best_state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)["state_dict"]
        if verbose:
            print(f"[resume] {run_dir} from step {st['step']}")

    if track and not resume:
        n_params = sum(p.numel() for p in model.parameters())
        mlflow.log_params({**tracking.flatten(cfg), "seed": seed, "n_train": n, "n_train_pos": int(y_tr.sum()),
                           "n_val": len(val), "n_params": n_params})
        mlflow.set_tags({f"env.{k}": v for k, v in environment().items()})
    train_df.to_csv(run_dir / "train_images.csv", index=False)
    if verbose and not resume:
        print(f"[{cfg['name']} seed{seed}] {cfg['arch']} on {device}: train={n} (pos {int(y_tr.sum())}) "
              f"val={len(val)} max_steps={cfg['max_steps']} bs={cfg['batch_size']} aug={cfg['augment']}")

    ls = cfg["label_smoothing"]
    lp = {**LOSSES[cfg["loss"]], **(cfg["loss_params"] or {})}
    if cfg["loss"] == "qdro":  # 4 groups: label x (softer / sharper half of the training images)
        with torch.no_grad():
            hf = torch.cat([sharpness(x_tr[k:k + 256].float().div(255)) for k in range(0, n, 256)])
        group = y_tr.long() * 2 + (hf < hf.median()).long()  # 0 normal-sharp, 1 normal-soft, 2 pneu-sharp, 3 pneu-soft
        st.setdefault("q", [0.25] * 4)

    def set_train():
        model.train()
        if cfg["freeze_backbone"]:
            model.net.eval()
            model.net.get_classifier().train()

    set_train()
    while st["step"] < cfg["max_steps"]:
        # ---- one optimisation step
        if st["pos"] + cfg["batch_size"] > n:
            st["perm"], st["pos"] = torch.randperm(n, generator=g_sample), 0
        idx = st["perm"][st["pos"]:st["pos"] + cfg["batch_size"]].to(device)
        st["pos"] += cfg["batch_size"]
        x = x_tr[idx].float().div_(255)
        if aug_params:
            x = augment_batch(x, aug_params, g_aug)
        y = y_tr[idx]
        y_s = y * (1 - ls) + 0.5 * ls if ls else y
        if cfg["mixup_alpha"] > 0:  # mixup within the batch; lambda drawn from a step-seeded RNG (resumable)
            lam = float(np.random.default_rng(seed * 1_000_003 + st["step"]).beta(cfg["mixup_alpha"], cfg["mixup_alpha"]))
            perm = torch.randperm(len(idx), generator=g_aug, device=device)
            x = lam * x + (1 - lam) * x[perm]
            y_s = lam * y_s + (1 - lam) * y_s[perm]

        factor = _lr_factor(cfg, st["step"], st["plateau_scale"])
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * factor
        with _autocast(device, cfg["amp"]):
            logits = model(x.contiguous(memory_format=torch.channels_last))
        z = logits.float()
        bce = torch.nn.functional.binary_cross_entropy_with_logits(z, y_s, reduction="none")
        if cfg["loss"] == "bce":
            loss = bce.mean()
        elif cfg["loss"] == "focal":
            p_t = torch.where(y > 0.5, torch.sigmoid(z), torch.sigmoid(-z))
            loss = ((1 - p_t) ** lp["gamma"] * bce).mean()
        elif cfg["loss"] == "gce":
            p_t = torch.where(y > 0.5, torch.sigmoid(z), torch.sigmoid(-z)).clamp_min(1e-6)
            loss = ((1 - p_t ** lp["q"]) / lp["q"]).mean()
        elif cfg["loss"] == "qdro":
            gb = group[idx]
            q = torch.tensor(st["q"], device=device)
            lg = torch.stack([bce[gb == k].mean() if (gb == k).any() else bce.new_zeros(()) for k in range(4)])
            present = torch.stack([(gb == k).any() for k in range(4)]).float()
            q = q * torch.exp(lp["eta"] * lg.detach()) * present + q * (1 - present)
            q = q / q.sum()
            st["q"] = q.tolist()
            loss = (q * lg * present).sum() / (q * present).sum()
        elif cfg["loss"] == "qcons":
            with _autocast(device, cfg["amp"]):
                z2 = model(degrade_batch(x, lp["sigma"], lp["contrast"], g_aug).contiguous(memory_format=torch.channels_last)).float()
            loss = bce.mean() + lp["lam"] * ((z - z2) ** 2).mean()
        else:
            raise ValueError(cfg["loss"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        st["step"] += 1
        st["running"].append(loss.item())

        # ---- evaluation / bookkeeping
        if st["step"] % cfg["eval_every"] and st["step"] != cfg["max_steps"]:
            continue
        z_val = predict_logits(model, val, device, cfg["eval_batch_size"], cfg["amp"])
        set_train()
        val_loss, val_auc = _bce(z_val, y_val), float(roc_auc_score(y_val, z_val))
        rec = {"step": st["step"], "epoch": st["step"] * cfg["batch_size"] / n,
               "train_loss": float(np.mean(st["running"])), "val_loss": val_loss, "val_auroc": val_auc,
               "lr": opt.param_groups[0]["lr"], "elapsed_s": time.time() - t_start}
        st["running"] = []
        improved = val_loss < st["best_loss"] - cfg["min_delta"]
        if improved:
            st.update(best_loss=val_loss, best_auroc=val_auc, best_step=st["step"], bad_evals=0)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if save_model:
                torch.save({"state_dict": best_state, "cfg": cfg, "seed": seed}, run_dir / "best.pt")
        else:
            st["bad_evals"] += 1
        if cfg["schedule"] == "plateau" and st["step"] >= cfg["warmup_steps"]:
            if val_loss < st["plateau_best"] - cfg["min_delta"]:
                st["plateau_best"], st["plateau_bad"] = val_loss, 0
            else:
                st["plateau_bad"] += 1
                if st["plateau_bad"] > cfg["plateau_patience"]:
                    st["plateau_scale"] *= cfg["plateau_factor"]
                    st["plateau_bad"] = 0
        rec["best"] = improved
        st["history"].append(rec)
        if track:
            mlflow.log_metrics({k: rec[k] for k in ("train_loss", "val_loss", "val_auroc", "lr", "epoch")},
                               step=st["step"])
        if verbose:
            print(f"  step {st['step']:5d} | train {rec['train_loss']:.4f} | val loss {val_loss:.4f} "
                  f"auroc {val_auc:.4f}{' *' if improved else ''} | lr {rec['lr']:.2e} | {rec['elapsed_s']:.0f}s")
        if trial is not None:
            import optuna

            trial.report(val_loss, st["step"])
            if trial.should_prune():
                raise optuna.TrialPruned()
        if st["bad_evals"] >= cfg["early_stopping_patience"]:
            st["stopped_early"] = True
        if save_model:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": st,
                        "g_sample": g_sample.get_state(), "g_aug": g_aug.get_state(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                        "mlflow_run_id": mlflow.active_run().info.run_id if track else None},
                       run_dir / "last.pt.tmp")
            os.replace(run_dir / "last.pt.tmp", run_dir / "last.pt")
        if st["stopped_early"]:
            if verbose:
                print(f"  early stop at step {st['step']} (best step {st['best_step']})")
            break

    # ---- finalise: best checkpoint, validation-chosen threshold
    model.load_state_dict(best_state)
    z_val = predict_logits(model, val, device, cfg["eval_batch_size"], cfg["amp"])
    p_val = sigmoid(z_val)
    thr = choose_threshold(y_val, p_val, cfg["threshold"])
    summary = {"status": "complete", "name": cfg["name"], "arch": cfg["arch"], "seed": seed, "threshold": thr,
               "best_val_loss": st["best_loss"], "best_val_auroc": st["best_auroc"], "best_step": st["best_step"],
               "steps_done": st["step"], "stopped_early": st["stopped_early"],
               "train_time_s": time.time() - t_start, "n_train": n, "cfg": cfg, "env": environment(),
               "mlflow_run_id": mlflow.active_run().info.run_id if track else None, "run_dir": str(run_dir)}
    pd.DataFrame(st["history"]).to_csv(run_dir / "history.csv", index=False)
    val.df.assign(logit=z_val, prob=p_val).to_csv(run_dir / "preds_val.csv", index=False)
    if save_model:
        torch.save({"state_dict": best_state, "cfg": cfg, "seed": seed, "threshold": thr}, run_dir / "best.pt")
        (run_dir / "last.pt").unlink(missing_ok=True)
    if track:
        mlflow.log_metrics({"best_val_loss": st["best_loss"], "best_val_auroc": st["best_auroc"],
                            "best_step": st["best_step"], "threshold": thr, "train_time_s": summary["train_time_s"],
                            "steps_done": st["step"]})
        for f in ("history.csv", "preds_val.csv", "train_images.csv"):
            mlflow.log_artifact(str(run_dir / f))
    (run_dir / "run.json").write_text(json.dumps(summary, indent=2, default=str))
    if track:
        mlflow.log_artifact(str(run_dir / "run.json"))
    return summary


# ----------------------------------------------------------------------------- evaluation

def load_model(run_dir: Path, device=None):
    device = device or get_device()
    ckpt = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = Classifier(cfg["arch"], pretrained=False, drop_rate=cfg.get("drop_rate", 0.0),
                       drop_path_rate=cfg.get("drop_path_rate", 0.0))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).to(memory_format=torch.channels_last).eval(), ckpt


def evaluate_run(run_dir: Path, eval_sets: dict, n_boot: int = 1000, force: bool = False, log: bool = True) -> dict:
    """Score each eval set with the run's frozen, validation-chosen threshold.

    eval_sets: {name: {manifest, where, label_cols}}. Missing manifests are skipped. Predictions are
    cached as preds_<name>.csv; metrics (with patient-clustered bootstrap CIs) go to metrics.json and MLflow.
    """
    run_dir = Path(run_dir)
    info = json.loads((run_dir / "run.json").read_text())
    thr, cfg = info["threshold"], info["cfg"]
    metrics_path = run_dir / "metrics.json"
    results = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    device, model, new = get_device(), None, {}
    for name, es in eval_sets.items():
        if not resolve(es["manifest"]).exists():
            continue
        pred_path = run_dir / f"preds_{name}.csv"
        keys = [f"{name}:{c}" for c in es.get("label_cols", ["label"])]
        if not force and pred_path.exists() and all(k in results for k in keys):
            continue
        if force or not pred_path.exists():
            if model is None:
                model, _ = load_model(run_dir, device)
            ts = load_set(es["manifest"], es.get("where"), cfg["image_size"], cfg["resize_mode"], device=None,
                          require_label=False)
            z = predict_logits(model, ts, device, 256, cfg.get("amp", True))
            ts.df.assign(logit=z, prob=sigmoid(z)).to_csv(pred_path, index=False)
        df = pd.read_csv(pred_path, dtype={"patient_id": str})
        for col, key in zip(es.get("label_cols", ["label"]), keys):
            sub = df[df[col].notna()]
            y, p = sub[col].to_numpy().astype(int), sub["prob"].to_numpy()
            m = all_metrics(y, p, thr)
            m["ci"] = bootstrap_ci(y, p, thr, groups=sub["patient_id"].to_numpy(), n_boot=n_boot)
            results[key] = new[key] = m
    metrics_path.write_text(json.dumps(results, indent=2))
    if log and new and info.get("mlflow_run_id"):
        _log_eval(info["mlflow_run_id"], new, run_dir)
    return results


def _log_eval(run_id: str, results: dict, run_dir: Path):
    flat = {}
    for key, m in results.items():
        prefix = key.replace(":", "/")
        for k in ("auroc", "auprc", "sensitivity", "specificity", "precision", "f1", "balanced_accuracy"):
            flat[f"{prefix}/{k}"] = m[k]
        for k, (lo, hi) in m["ci"].items():
            flat[f"{prefix}/{k}_ci_lo"], flat[f"{prefix}/{k}_ci_hi"] = lo, hi
    client = mlflow.MlflowClient(tracking.TRACKING_URI)
    ts = int(time.time() * 1000)
    client.log_batch(run_id, metrics=[Metric(k, float(v), ts, 0) for k, v in flat.items() if np.isfinite(v)])
    client.log_artifact(run_id, str(run_dir / "metrics.json"))
    for name in {k.split(":")[0] for k in results}:
        client.log_artifact(run_id, str(run_dir / f"preds_{name}.csv"))
