"""Export a trained run for the demo: models/<exp>_<arch>/{model.pt, card.json}.

The seed is chosen by A-validation loss (never by external results).

    python scripts/export_model.py --exp E2 --arch convnext_tiny
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cxr import PROJECT_ROOT
from cxr.experiments import EXPERIMENTS, run_dir


def experiment_doc(name: str) -> str:
    """Doc of the registered experiment, also for tagged run names such as E3_sd_lora or E1_HARM."""
    for prefix in ("CXVB_", "CX_"):
        if name.startswith(prefix):
            return "CheXpert source: " + experiment_doc(name[len(prefix):])
    base = next((k for k in sorted(EXPERIMENTS, key=len, reverse=True) if name == k or name.startswith(k + "_")), None)
    if base is None:
        return ""
    tag = name[len(base) + 1:]
    return EXPERIMENTS[base]["doc"] + (f" (tag {tag})" if tag else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    seeds = [args.seed] if args.seed is not None else [0, 1, 2]
    infos = {s: json.loads((run_dir(args.exp, args.arch, s) / "run.json").read_text())
             for s in seeds if (run_dir(args.exp, args.arch, s) / "run.json").exists()}
    seed = min(infos, key=lambda s: infos[s]["best_val_loss"])
    rd = run_dir(args.exp, args.arch, seed)
    info = infos[seed]
    metrics = json.loads((rd / "metrics.json").read_text()) if (rd / "metrics.json").exists() else {}
    out = PROJECT_ROOT / "models" / f"{args.exp}_{args.arch}"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(rd / "best.pt", out / "model.pt")
    card = {
        "name": f"{args.exp}_{args.arch}", "experiment": args.exp, "source": info["cfg"].get("source", "kermany"),
        "experiment_doc": experiment_doc(args.exp), "arch": args.arch, "seed": seed,
        "threshold": info["threshold"], "image_size": info["cfg"]["image_size"],
        "resize_mode": info["cfg"]["resize_mode"], "n_train": info["n_train"],
        "best_val_loss": info["best_val_loss"], "best_val_auroc": info["best_val_auroc"],
        "metrics": {k: {m: v[m] for m in ("auroc", "sensitivity", "specificity", "f1", "n", "prevalence")}
                    for k, v in metrics.items()},
        "env": info.get("env", {}),
    }
    (out / "card.json").write_text(json.dumps(card, indent=2))
    print("exported", out)


if __name__ == "__main__":
    main()
