"""Context re-rendering (docs/RESEARCH_PIVOT.md §8): keep the REAL lungs of an E1 training image and let the teammate's
SD-1.5 LoRA redraw everything else (body outline, mediastinum edges, borders, markers) with ONE class-neutral prompt.

The label is the source's real label and the pathology stays real (lung pixels are pasted back). The generated texture
is outside the lungs and does not depend on the class, so the edit can only teach "the context does not matter": a
generative version of E2_ERASE's border masking, the strongest augmentation found so far. Sources are E1's own images
(subset_rank < --per-class), so E1 -> E3/E4_CTX adds no new real information.

    .venv-gen/bin/python scripts/synth_fix/ctx_generate.py --prototype          # 4 sources x 3 strengths -> grid PNG
    .venv-gen/bin/python scripts/synth_fix/ctx_generate.py --strength 1.0 --name ctx_e1
Output: data/synthetic/<name>/{NORMAL,PNEUMONIA}/*.png + manifest.csv, artifacts/manifests/synthetic_<name>.csv
(synth_rank = the source's per-class subset_rank, so E3 takes the first 125 per class and E4 the first 250)."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/synth_cf"))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cf_generate import A, CFG, NEG, STEPS, edit, load_pipe, lung_masks
from cxr.data import read_manifest, select
from cxr.imaging import load_gray, to_square

CTX_PROMPT = "a frontal cxr_scan, radiograph"


def sources(per_class, offset=0):
    pool = select(read_manifest(A), {"split": "train", "subset_rank": {"ge": offset, "lt": offset + per_class}})
    return pool.sort_values(["subset_rank", "label"]).reset_index(drop=True)  # interleaved: every prefix is balanced


def ctx_edit(pipe, emb, img, lung_hard, lung_soft, strength, seed):
    return edit(pipe, emb, img, ~lung_hard, 1 - lung_soft, "ctx", strength, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prototype", action="store_true")
    ap.add_argument("--per-class", type=int, default=250)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--name", default="ctx_e1")
    ap.add_argument("--mem-frac", type=float, default=0.35)
    args = ap.parse_args()

    pipe, emb = load_pipe(args.mem_frac, {"ctx": CTX_PROMPT})
    if args.prototype:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        src = sources(2, offset=400)
        imgs = [to_square(load_gray(ROOT / p), 512, "squash") for p in src.image_path]
        hard, soft = lung_masks(imgs)
        strengths = [0.6, 0.8, 1.0]
        fig, axes = plt.subplots(len(src), 1 + len(strengths), figsize=(2.6 * (1 + len(strengths)), 2.7 * len(src)))
        for i, r in src.iterrows():
            axes[i, 0].imshow(imgs[i], cmap="gray"); axes[i, 0].contour(hard[i], levels=[0.5], colors="cyan", linewidths=0.5)
            axes[i, 0].set_title(f"source ({'pneu' if r.label else 'norm'})", fontsize=7)
            for j, s in enumerate(strengths):
                axes[i, 1 + j].imshow(ctx_edit(pipe, emb, imgs[i], hard[i], soft[i], s, seed=2000 + i), cmap="gray", vmin=0, vmax=255)
                axes[i, 1 + j].set_title(f"context s={s}", fontsize=7)
        for ax in axes.flat:
            ax.axis("off")
        plt.tight_layout()
        out = ROOT / "reports/results/synth_fix/ctx_prototype.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=70)
        print("prototype written", out, "peak GPU MB", torch.cuda.max_memory_allocated() // 2**20)
        return

    out = ROOT / "data/synthetic" / args.name
    for c in ("NORMAL", "PNEUMONIA"):
        (out / c).mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.csv"
    rows = pd.read_csv(man_path).to_dict("records") if man_path.exists() else []
    have = {r["source_id"] for r in rows}
    src = sources(args.per_class)
    for k in range(0, len(src), 8):
        chunk = src.iloc[k:k + 8]
        if set(chunk.image_id) <= have:
            continue
        imgs = [to_square(load_gray(ROOT / p), 512, "squash") for p in chunk.image_path]
        hard, soft = lung_masks(imgs)
        for (i, r), img, h, s in zip(chunk.iterrows(), imgs, hard, soft):
            cname = "PNEUMONIA" if r.label else "NORMAL"
            fn = f"ctx_{cname.lower()}_{int(r.subset_rank):05d}.png"
            Image.fromarray(ctx_edit(pipe, emb, img, h, s, args.strength, seed=20_000 + i)).save(out / cname / fn)
            rows.append({"filename": fn, "relative_path": f"{cname}/{fn}", "label": cname, "prompt": CTX_PROMPT,
                         "negative_prompt": NEG, "seed": 20_000 + i, "guidance_scale": CFG, "steps": STEPS,
                         "strength": args.strength, "source_id": r.image_id, "source_label": int(r.label),
                         "source_patient": r.patient_id, "rank": int(r.subset_rank)})
        pd.DataFrame(rows).to_csv(man_path, index=False)
        print(f"{min(k + 8, len(src))}/{len(src)} images, peak GPU MB {torch.cuda.max_memory_allocated() // 2**20}", flush=True)
    m = pd.DataFrame(rows)
    m = pd.DataFrame({"label": (m.label == "PNEUMONIA").astype(int),
                      "image_id": [f"S/{args.name}/{r}" for r in m.relative_path.str.replace(".png", "")],
                      "image_path": [f"data/synthetic/{args.name}/{r}" for r in m.relative_path],
                      "source": f"synthetic:{args.name}", "split": "train", "patient_id": m.source_patient,
                      "synth_rank": m["rank"], "source_id": m.source_id, "source_label": m.source_label})
    m.to_csv(ROOT / f"artifacts/manifests/synthetic_{args.name}.csv", index=False)
    print("ctx done", m.label.value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
