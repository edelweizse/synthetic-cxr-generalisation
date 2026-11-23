"""Train and evaluate experiment runs. Idempotent: finished runs are skipped, interrupted runs resume.

    python scripts/run.py --exp E1 E2 --arch convnext_tiny resnet50 --seeds 0 1 2
    python scripts/run.py --exp E3 E4 --arch convnext_tiny --synthetic artifacts/manifests/synthetic_lora.csv
    python scripts/run.py --exp E1 --arch efficientnet_b0 --resize-mode harmonize224 --tag HARM
    python scripts/run.py --exp E1 --arch efficientnet_b0 --resize-mode harmonize224 --hpo-variant harmonize224 --tag HARMT
    python scripts/run.py --exp E1 E2_ERASE --arch densenet121 --source chexpert     # study 2: runs/CX_E1, CX_E2_ERASE
    python scripts/run.py --eval-only            # (re)score every finished run on all available eval sets
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)  # queue logs go through tee; keep them live (dashboard)

import torch

from cxr.engine import RUNS_DIR, evaluate_run, train_run
from cxr.experiments import EVAL_SETS, EXPERIMENTS, SOURCE_PREFIX, SOURCES, make_cfg, run_dir


def finished_runs():
    for p in sorted(RUNS_DIR.glob("*/*/seed*/run.json")):
        if json.loads(p.read_text()).get("status") == "complete":
            yield p.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="*", default=[], choices=list(EXPERIMENTS))
    ap.add_argument("--arch", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--synthetic", default=None, help="synthetic manifest for E3/E4 (default artifacts/manifests/synthetic.csv)")
    ap.add_argument("--tag", default="", help="suffix for the experiment directory, e.g. a synthetic-data version")
    ap.add_argument("--resize-mode", default=None, help="override preprocessing, e.g. harmonize224 (see cxr/imaging.to_square)")
    ap.add_argument("--hpo-variant", default=None, help="use configs/hpo/<arch>_<variant>.json (scripts/hpo.py --resize-mode)")
    ap.add_argument("--loss", default=None, help="training loss (cxr/engine.LOSSES): focal, gce, qdro, qcons")
    ap.add_argument("--loss-params", default=None, help='JSON, e.g. \'{"lam": 0.5}\'')
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (memory only), e.g. ConvNeXt + qcons on 8 GB")
    ap.add_argument("--source", default="kermany", choices=list(SOURCES), help="real training source (study 2: chexpert)")
    ap.add_argument("--val", default=None, help="validation manifest (all rows) instead of the source's real val split, "
                    "e.g. held-out synthetic images for a training pipeline without real images")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    if args.eval_only:
        for rd in finished_runs():
            evaluate_run(rd, EVAL_SETS, n_boot=args.n_boot)
            print("evaluated", rd.relative_to(RUNS_DIR), flush=True)
        return

    failures = []
    for exp in args.exp:
        for arch in args.arch:
            for seed in args.seeds:
                name = SOURCE_PREFIX[args.source] + exp + (f"_{args.tag}" if args.tag else "")
                cfg = make_cfg(exp, arch, overrides={"name": f"{name}/{arch}",
                                                           **({"resize_mode": args.resize_mode} if args.resize_mode else {}),
                                                           **({"loss": args.loss} if args.loss else {}),
                                                           **({"grad_checkpointing": True} if args.grad_ckpt else {}),
                                                           **({"val": {"manifest": args.val, "where": None}} if args.val else {}),
                                                           **({"loss_params": json.loads(args.loss_params)} if args.loss_params else {})}, synthetic_manifest=args.synthetic,
                               hpo_variant=args.hpo_variant, source=args.source)
                rd = run_dir(name, arch, seed)
                try:
                    train_run(cfg, seed, rd, tags={"experiment": name, "stage": "train"})
                    res = evaluate_run(rd, EVAL_SETS, n_boot=args.n_boot)
                    print(f"[{name}/{arch}/seed{seed}] " + " | ".join(
                        f"{k} AUROC {v['auroc']:.3f}" for k, v in res.items()), flush=True)
                except Exception:  # keep the queue going; failures are reported at the end
                    traceback.print_exc()
                    failures.append(str(rd))
                finally:
                    torch.cuda.empty_cache()
    if failures:
        print("FAILED:", *failures, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
