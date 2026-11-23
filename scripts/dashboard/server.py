"""Live, read-only queue dashboard for the GPU box: what is running, why, how far, what comes next, results so far.

    .venv/bin/python scripts/dashboard/server.py              # http://<tailscale ip>:8765 (binds the tailnet address only)
    .venv/bin/python scripts/dashboard/server.py --host 127.0.0.1

Reads only: artifacts/runs/*/run.json + preds_B_external.csv + metrics.json, the queue logs, `tmux ls`, nvidia-smi.
It never starts, stops or changes anything. The plan below mirrors scripts/queue_fix.sh and scripts/queue_cx.sh; when a
queue script changes, update PLAN too. Numbers are the view-balanced RSNA-test AUROC of the
mean prediction over the finished seeds, as in the report (point estimates; the report adds bootstrap CIs).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "artifacts/runs"
HERE = Path(__file__).resolve().parent
ARCHS = ["efficientnet_b0", "resnet18", "densenet121"]
SEEDS = [0, 1, 2]
MAX_STEPS = 1500


def arm(key, title, why, exps, ref, archs=ARCHS):
    return dict(key=key, title=title, why=why, exps=exps, ref=ref, archs=archs)


PLAN = [
    dict(id="fix", tmux="fix", log="logs_fix.log", title="Can the synthetic images be made to help?",
         why="Synthetic images from our generator lowered RSNA AUROC in every earlier setting. Each arm changes one thing "
             "about how they are labelled or chosen, to find out what causes the harm and whether it can be removed.",
         arms=[
             arm("SRC_S10", "Counterfactual pairs, source label", "Both edits of a real image get the real image's own label, "
                 "so the generated lung texture says nothing about the class (control: is the editing itself harmful?).",
                 ["E3_SRC_S10", "E4_SRC_S10"], "E1"),
             arm("TEACH", "Synthetic images, teacher labels", "Same synthetic images as before, but each one is labelled with the "
                 "probability a strong real-trained model gives it, instead of the prompt's 0/1.",
                 ["E3_TEACH", "E4_TEACH"], "E1"),
             arm("CTX", "Real lungs, redrawn surroundings", "Generator redraws only what is outside the lungs, with one neutral "
                 "prompt: teaches the model that borders, markers and body outline do not matter.",
                 ["E3_CTX", "E4_CTX"], "E1"),
             arm("RTEACH", "Real images, teacher labels", "Control for TEACH: the same teacher labels on extra REAL images. "
                 "If this helps as much, the gain comes from copying the teacher, not from the synthetic images.",
                 ["E3_RTEACH", "E4_RTEACH"], "E1"),
             arm("CONF", "Only convincing synthetic images", "Prompt labels, but synthetic pneumonia images the teacher finds most "
                 "convincing come first, and 'normal' images it calls pneumonia are dropped.",
                 ["E3_CONF", "E4_CONF"], "E1"),
             arm("SRC_S08", "Counterfactual pairs (weaker edit), source label", "As SRC_S10 at edit strength 0.8 (more of the "
                 "original lungs kept).", ["E3_SRC_S08", "E4_SRC_S08"], "E1"),
         ]),
    dict(id="cx", tmux="cx", log="logs_cx.log", after="fix", title="Study 2: CheXpert instead of Kermany as the source",
         why="Kermany is children from one hospital; RSNA is adults. Training on adult CheXpert tests whether the source "
             "dataset was the main problem. CX = natural AP/PA mix, CXVB = view-balanced training sets (no AP/PA shortcut).",
         arms=[
             arm("CX_E1", "Baseline, CheXpert", "Same 1,000-image baseline as E1, from CheXpert. Δ is vs the Kermany E1.",
                 ["CX_E1"], "E1"),
             arm("CXVB_E1", "Baseline, CheXpert view-balanced", "As CX_E1 with half AP, half PA films in each class. Δ vs Kermany E1.",
                 ["CXVB_E1"], "E1"),
             arm("CX_E2_ERASE", "Strong augmentation, CheXpert", "Best Kermany recipe (random erasing + border masking); also "
                 "the future teacher for CheXpert synthetic images. Δ vs CX_E1.", ["CX_E2_ERASE"], "CX_E1"),
             arm("CXVB_E2_ERASE", "Strong augmentation, view-balanced", "Δ vs CXVB_E1.", ["CXVB_E2_ERASE"], "CXVB_E1"),
             arm("CX_R50", "+50% real images, CheXpert", "Reference for later synthetic arms. Δ vs CX_E1.", ["CX_R50"], "CX_E1"),
             arm("CXVB_R50", "+50% real images, view-balanced", "Δ vs CXVB_E1.", ["CXVB_R50"], "CXVB_E1"),
             arm("CX_N2000", "2,000 real images, CheXpert", "Scaling point. Δ vs CX_E1.", ["CX_N2000"], "CX_E1"),
             arm("CXVB_N2000", "2,000 real images, view-balanced", "Δ vs CXVB_E1.", ["CXVB_N2000"], "CXVB_E1"),
             arm("CX_VIT", "ViT-S teacher member", "Third teacher model, as for the Kermany teacher.", ["CX_E2_ERASE"], None,
                 archs=["vit_small"]),
         ],
         tail="then: every finished run scored on the CheXpert test set (--eval-only), tables rebuilt"),
]

def count_png(d):
    d = ROOT / d
    return sum(1 for sub in ("NORMAL", "PNEUMONIA") if (d / sub).is_dir() for f in os.scandir(d / sub)
               if f.name.endswith(".png") and not f.name.endswith(".tmp.png"))


LORA_RE = re.compile(r"loss=([\d.]+), step=(\d+)")


def lora_detail(tail):
    m = next((LORA_RE.search(l) for l in reversed(tail) if LORA_RE.search(l)), None)
    return dict(text=f"step {m[2]} / 2500, loss {m[1]}", frac=int(m[2]) / 2500) if m else dict(text="loading the base model")


def stage(title, why, est_s, done, detail=None):
    return dict(title=title, why=why, est_s=est_s, done=done, detail=detail)


PLAN.append(dict(
    id="gen", tmux="gen", log="logs_gen.log", after="cx", title="Study 2: our generator retrained on CheXpert",
    why="The same generator recipe (teammate's train_lora.py, same settings), trained on adult CheXpert images instead of "
        "Kermany, then the study-1 arms on top of the CheXpert baseline: does the teacher-label fix carry over?",
    stages=[
        stage("Prepare training images", "All 9,968 CheXpert train images at 512 px with the two class prompts as captions "
              "(val/test never reach the generator).", 120,
              lambda: (ROOT / "data/generator/chexpert/metadata.jsonl").exists(),
              lambda tail: dict(text=f"{count_png('data/generator/chexpert'):,} / 9,968 images",
                                frac=count_png("data/generator/chexpert") / 9968)),
        stage("Train the LoRA", "Teammate's script, 2,500 steps of batch 4 (about one pass over the images), r = 16.", 4500,
              lambda: (ROOT / "data/generator/lora_cx/adapter_model.safetensors").exists(), lora_detail),
        stage("Sample synthetic images", "1,000 per class with the study-1 prompts, negative prompt, CFG 4.5, 30 steps.", 7200,
              lambda: (ROOT / "data/synthetic/cx_sd/manifest.csv").exists(),
              lambda tail: dict(text=f"{count_png('data/synthetic/cx_sd'):,} / 2,000 images",
                                frac=count_png("data/synthetic/cx_sd") / 2000)),
        stage("Teacher labels", "CX_E2_ERASE models (EfficientNet-B0, DenseNet-121, ViT-S) score every synthetic image and "
              "the extra real images.", 300, lambda: (ROOT / "artifacts/manifests/real_teach_cx.csv").exists()),
    ],
    arms=[
        arm("CX_SYN_ONLY_SD", "Synthetic only, prompt labels", "Can our generator REPLACE the real data? 2,000 synthetic "
            "images, no real ones. Δ vs CX_N2000 (2,000 real images).", ["CX_SYN_ONLY_SD"], "CX_N2000"),
        arm("CX_SYN_ONLY_TEACH", "Synthetic only, teacher labels", "Same, labelled by the real-trained teacher (share a "
            "generator + a model instead of patient data). Δ vs CX_N2000.", ["CX_SYN_ONLY_TEACH"], "CX_N2000"),
        arm("CX_R25", "+25% real images, CheXpert", "Real-data reference for +250 images. Δ vs CX_E1.", ["CX_R25"], "CX_E1"),
        arm("CX_SD", "Synthetic images, prompt labels", "The study-1 setting that hurt on Kermany, now with a CheXpert "
            "generator. Δ vs CX_E1.", ["CX_E3_SD", "CX_E4_SD"], "CX_E1"),
        arm("CX_TEACH", "Synthetic images, teacher labels", "The fix that worked on Kermany. Δ vs CX_E1.",
            ["CX_E3_TEACH", "CX_E4_TEACH"], "CX_E1"),
        arm("CX_RTEACH", "Real images, teacher labels", "Distillation control for CX_TEACH. Δ vs CX_E1.",
            ["CX_E3_RTEACH", "CX_E4_RTEACH"], "CX_E1"),
    ]))

LATER = [
    ("Report + slides", "Write up pivot §8 (teacher labels fix the synthetic penalty) and §9 (CheXpert source: +0.12 to "
     "+0.16; generator retrained on it). Deadline ~11 Nov."),
]


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return ""


def gpu():
    q = sh(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits"]).strip().split(", ")
    return dict(zip(["name", "mem_used", "mem_total", "util", "temp"], q)) if len(q) == 5 else None


_score_cache = {}


def ensemble_score(exp, arch):
    """(view-balanced RSNA-test AUROC, C_test AUROC, n seeds) of the mean prediction over the finished seeds (as in the
    report), cached by file mtimes."""
    rds = [RUNS / exp / arch / f"seed{k}" for k in SEEDS]
    rds = [rd for rd in rds if run_state(rd)[0] == "done" and (rd / "preds_B_external.csv").exists()]
    if not rds:
        return None, None, 0
    key = tuple((rd / f).stat().st_mtime if (rd / f).exists() else 0 for rd in rds for f in ("preds_B_external.csv", "preds_C_test.csv"))
    hit = _score_cache.get((exp, arch))
    if hit and hit[0] == key:
        return hit[1]
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from cxr.analysis import add_b_role

    def ens(name, cols):
        ps = [pd.read_csv(rd / f"preds_{name}.csv", usecols=cols, dtype={"patient_id": str}) for rd in rds
              if (rd / f"preds_{name}.csv").exists()]
        if len(ps) < len(rds):
            return None
        return ps[0].assign(prob=np.mean([p.prob.to_numpy() for p in ps], axis=0))

    df = add_b_role(ens("B_external", ["image_id", "label", "view", "prob", "patient_id"]))
    df = df[df.label.notna() & (df.b_role == "test")]
    bal = sum(roc_auc_score(g.label, g.prob) for v, g in df.groupby("view") if v in ("AP", "PA")) / 2
    c = ens("C_test", ["image_id", "label", "prob"])
    c = roc_auc_score(c.label, c.prob) if c is not None else None
    _score_cache[(exp, arch)] = (key, (bal, c, len(rds)))
    return bal, c, len(rds)


def run_state(rd: Path):
    j = rd / "run.json"
    if j.exists():
        info = json.loads(j.read_text())
        if info.get("status") == "complete":
            return "done", j.stat().st_mtime, info
    if rd.exists():
        return "partial", max((f.stat().st_mtime for f in rd.iterdir()), default=rd.stat().st_mtime), None
    return "pending", None, None


STEP_RE = re.compile(r"step\s+(\d+) \| train ([\d.]+) \| val loss ([\d.]+) auroc ([\d.]+)")


def log_tail(path: Path, n=40):
    if not path.exists():
        return []
    with open(path, "rb") as f:
        f.seek(max(0, path.stat().st_size - 60_000))
        lines = f.read().decode(errors="replace").split("\n")
    return [l.split("\r")[-1].rstrip() for l in lines if l.strip()][-n:]


def status():
    sessions = {l.split(":")[0] for l in sh(["tmux", "ls"]).splitlines() if ":" in l}
    out, completions = [], []
    for q in PLAN:
        alive = q["tmux"] in sessions
        waiting = alive and q.get("after") in sessions
        arms, runs_all = [], []
        for a in q["arms"]:
            cells, results = [], []
            for arch in a["archs"]:
                for exp in a["exps"]:
                    for k in SEEDS:
                        st, t, info = run_state(RUNS / exp / arch / f"seed{k}")
                        r = dict(exp=exp, arch=arch, seed=k, state=st, t=t)
                        cells.append(r)
                        runs_all.append(r)
                        if st == "done":
                            completions.append(t)
                    bal, c, n = ensemble_score(exp, arch)
                    ref = ensemble_score(a["ref"], arch)[0] if a["ref"] else None
                    if n:
                        results.append(dict(exp=exp, arch=arch, n=n, bal=bal, c_test=c,
                                            delta=None if ref is None or bal is None else bal - ref))
            done = sum(c["state"] == "done" for c in cells)
            arms.append(dict(key=a["key"], title=a["title"], why=a["why"], ref=a["ref"], done=done, total=len(cells),
                             cells=cells, results=results))
        partial = [r for r in runs_all if r["state"] == "partial"]
        current = max(partial, key=lambda r: r["t"]) if partial and alive and not waiting else None
        tail = log_tail(ROOT / q["log"], 400)
        live = None
        if current:
            # step lines only after this run's start line (a resumed run prints none: then no live progress)
            tag = f"[{current['exp']}/{current['arch']} seed{current['seed']}]"
            start = next((i for i in range(len(tail) - 1, -1, -1) if tail[i].startswith(tag)), None)
            m = None if start is None else next(
                (STEP_RE.search(l) for l in reversed(tail[start:]) if STEP_RE.search(l)), None)
            live = dict(step=int(m[1]), train_loss=float(m[2]), val_loss=float(m[3]), val_auroc=float(m[4]),
                        max_steps=MAX_STEPS) if m else None
            current = {**current, "live": live, "age_s": time.time() - current["t"]}
        stages, stage_left, active_stage = [], 0, None
        for sg in q.get("stages", []):
            done = sg["done"]()
            row = dict(title=sg["title"], why=sg["why"], state="done" if done else "pending", detail=None)
            if not done:
                stage_left += sg["est_s"]
                if active_stage is None and alive and not waiting:
                    active_stage = row
                    row["state"] = "running"
                    row["detail"] = sg["detail"](tail) if sg["detail"] else None
            stages.append(row)
        state = "waiting" if waiting else "running" if alive else (
            "finished" if all(r["state"] == "done" for r in runs_all) and stage_left == 0 else "stopped")
        out.append(dict(id=q["id"], title=q["title"], why=q["why"], state=state, after=q.get("after"), tail_note=q.get("tail"),
                        arms=arms, current=current, stages=stages, stage=active_stage,
                        stage_left_s=stage_left, log=tail[-15:], done=sum(r["state"] == "done" for r in runs_all),
                        total=len(runs_all)))
    gaps = sorted(completions)
    gaps = [b - a for a, b in zip(gaps, gaps[1:]) if 0 < b - a < 1800][-30:]
    per_run = median(gaps) if gaps else 150.0
    left = 0
    for q in out:
        q["eta_s"] = None
        if q["state"] in ("running", "waiting"):
            left += (q["total"] - q["done"]) * per_run + q["stage_left_s"]
            q["eta_s"] = left
    return dict(now=time.time(), gpu=gpu(), sessions=sorted(sessions), per_run_s=per_run, queues=out,
                later=[dict(title=t, why=w) for t, w in LATER])


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/status":
            try:
                body, ctype = json.dumps(status()).encode(), "application/json"
            except Exception as e:  # keep the page alive; show the error instead
                body, ctype = json.dumps({"error": repr(e)}).encode(), "application/json"
        elif self.path in ("/", "/index.html"):
            body, ctype = (HERE / "index.html").read_bytes(), "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def tailnet_ip():
    ip = sh(["tailscale", "ip", "-4"]).strip().splitlines()
    return ip[0] if ip else "127.0.0.1"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None, help="default: this machine's Tailscale address (tailnet only)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    host = args.host or tailnet_ip()
    print(f"dashboard on http://{host}:{args.port}", flush=True)
    ThreadingHTTPServer((host, args.port), Handler).serve_forever()
