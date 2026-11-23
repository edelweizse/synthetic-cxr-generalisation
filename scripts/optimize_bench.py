"""Resource measurements and optimisation trade-offs for the deployed classifier (report: Optimisation).

Models: CX_E2_ERASE seed 0 (the CheXpert teacher members) for EfficientNet-B0, ResNet-18, DenseNet-121, ViT-S.
Quality: view-balanced RSNA-test AUROC (same metric as everywhere else), recomputed for every variant.

  1. size: parameters, checkpoint MB
  2. CPU latency, batch 1, 224 px, 4 threads (a laptop-class demo), median of 50 after 10 warm-up
  3. GPU throughput (images/s, batch 128) and peak memory, fp32 vs bf16 autocast, with the AUROC of each
  4. CPU int8 dynamic quantisation of the Linear layers (matters for ViT-S, whose compute is mostly Linear):
     latency, size and AUROC vs fp32
  5. 3-seed ensemble: AUROC gain vs 3x inference cost (from the saved per-seed predictions)

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/optimize_bench.py  -> reports/results/optimize_bench.csv
"""
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from cxr.analysis import b_role, balanced_auroc
from cxr.data import load_set
from cxr.engine import RUNS_DIR, load_model, predict_logits, sigmoid
from cxr.experiments import EVAL_SETS

ARCHS = ["efficientnet_b0", "resnet18", "densenet121", "vit_small"]
EXP = "CX_E2_ERASE"
OUT = ROOT / "reports/results/optimize_bench.csv"

spec = EVAL_SETS["B_external"]
full = load_set(spec["manifest"], spec["where"], 224, "squash", device=None)
mask = (full.df.patient_id.astype(str).map(b_role) == "test").to_numpy()
y, view = full.labels.numpy(), full.df.view.to_numpy()


def rsna_auc(logits):
    return balanced_auroc(y[mask], sigmoid(logits)[mask], view[mask])


def size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 2**20


def cpu_latency_ms(model, reps=50, warm=10):
    torch.set_num_threads(4)
    x = torch.rand(1, 1, 224, 224).contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        for _ in range(warm):
            model(x)
        t = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model(x)
            t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t))


rows = []
gpu = torch.device("cuda")
for arch in ARCHS:
    rd = RUNS_DIR / EXP / arch / "seed0"
    model, _ = load_model(rd, gpu)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    for amp in (False, True):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        logits = predict_logits(model, full, gpu, batch_size=128, amp=amp)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        rows.append(dict(arch=arch, variant="gpu bf16" if amp else "gpu fp32", params_m=n_params,
                         size_mb=size_mb(model), imgs_per_s=len(full) / dt,
                         peak_mem_mb=torch.cuda.max_memory_allocated() / 2**20, rsna_balanced=rsna_auc(logits)))
        print(rows[-1], flush=True)
    cpu_model = model.float().cpu().eval()
    rows.append(dict(arch=arch, variant="cpu fp32", params_m=n_params, size_mb=size_mb(cpu_model),
                     latency_ms=cpu_latency_ms(cpu_model), rsna_balanced=rows[-2]["rsna_balanced"]))
    print(rows[-1], flush=True)
    if arch == "vit_small":
        q = torch.ao.quantization.quantize_dynamic(cpu_model, {torch.nn.Linear}, dtype=torch.qint8)
        torch.set_num_threads(12)
        ql = predict_logits(q, full, torch.device("cpu"), batch_size=64, amp=False)
        rows.append(dict(arch=arch, variant="cpu int8 (dynamic, Linear)", params_m=n_params, size_mb=size_mb(q),
                         latency_ms=cpu_latency_ms(q), rsna_balanced=rsna_auc(ql)))
        print(rows[-1], flush=True)
    del model, cpu_model
    torch.cuda.empty_cache()

# 3-seed ensemble from the saved predictions (made with the training-time bf16 autocast)
for arch in ARCHS:
    ps = []
    for s in (0, 1, 2):
        p = pd.read_csv(RUNS_DIR / EXP / arch / f"seed{s}" / "preds_B_external.csv",
                        usecols=["image_id", "label", "view", "prob", "patient_id"], dtype={"patient_id": str})
        ps.append(p[p.label.notna() & (p.patient_id.map(b_role) == "test")].set_index("image_id").sort_index())
    single = [balanced_auroc(p.label.values, p.prob.values, p.view.values) for p in ps]
    ens = pd.concat([p.prob for p in ps], axis=1).mean(1)
    rows.append(dict(arch=arch, variant="3-seed ensemble (vs mean single)", rsna_balanced=balanced_auroc(
        ps[0].label.values, ens.values, ps[0].view.values), rsna_single_mean=float(np.mean(single)), cost_x=3))
    print(rows[-1], flush=True)

df = pd.DataFrame(rows).round(4)
df.to_csv(OUT, index=False)
train = []
for rj in RUNS_DIR.glob("CX_*/*/seed*/run.json"):
    r = json.loads(rj.read_text())
    train.append(dict(group=rj.parts[-4], arch=r["arch"], train_time_s=r.get("train_time_s"), steps=r.get("steps_done")))
t = pd.DataFrame(train).dropna()
print(df.to_string())
print("CheXpert-study training time per run (s): median", round(t.train_time_s.median(), 1),
      "IQR", t.train_time_s.quantile([.25, .75]).round(1).tolist(), "n", len(t))
