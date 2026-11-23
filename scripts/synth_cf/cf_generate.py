"""Counterfactual synthetic pairs (docs/RESEARCH_PIVOT.md §7): edit only the lungs of a REAL Kermany image with the
teammate's SD-1.5 LoRA, once with the "normal" prompt and once with the "pneumonia" prompt.

Both images of a pair share the source image, lung mask, noise seed and strength; only the prompt differs. So everything
outside the lungs (patient, hospital look, quality, markers) and every processing artefact (VAE round trip, inpainting
texture) is identical in both classes, and nothing but the prompt-driven lung content can predict the label.
Sources: Kermany training images of patients outside E1 (subset_rank >= 500), half normal and half pneumonia, so the
patient population behind both classes is the same. Pixels outside the (dilated, feathered) lung mask are pasted back from
the source image.

Runs in the separate generation venv (.venv-gen: diffusers / peft), never in the training venv:
    .venv-gen/bin/python scripts/synth_cf/cf_generate.py --prototype            # 4 sources x 3 strengths -> grid PNG
    .venv-gen/bin/python scripts/synth_cf/cf_generate.py --pairs 250 --strength 0.8 --name cf_lung
GPU memory is capped (--mem-frac, default 0.35 of the card) so that a concurrent training run is not starved; text prompts
are encoded once on the CPU and the text encoder never touches the GPU.
Output: data/synthetic/<name>/{NORMAL,PNEUMONIA}/*.png + manifest.csv (columns as the sd_lora manifest, plus source_id, pair).
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter

from cxr.data import read_manifest, select
A, BASE_PER_CLASS = "artifacts/manifests/kermany.csv", 500  # = cxr.experiments (not imported: it pulls in mlflow)
from cxr.imaging import load_gray, to_square

SD_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
ADAPTER = ROOT / "data/synthetic/sd_lora_adapter"
# the teammate's sampling settings (data/synthetic/sd_lora/manifest.csv)
PROMPT = {0: "a frontal cxr_scan of normal lungs, clear lung fields, radiograph",
          1: "a frontal cxr_scan showing pneumonia, lung consolidation, radiograph"}
NEG = ("3d render, skeleton, glowing, neon, illustration, cartoon, drawing, artistic, transparent, perspective, "
       "bad anatomy, deformed")
CFG, STEPS = 4.5, 30


def load_pipe(mem_frac, prompts=None):
    from diffusers import StableDiffusionInpaintPipeline
    from huggingface_hub import snapshot_download
    from peft import PeftModel

    torch.cuda.set_per_process_memory_fraction(mem_frac)
    local = snapshot_download(SD_REPO, allow_patterns=[
        "model_index.json", "scheduler/*", "tokenizer/*", "feature_extractor/*",
        "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
        "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors"])
    pipe = StableDiffusionInpaintPipeline.from_pretrained(local, variant="fp16", torch_dtype=torch.float16,
                                                          safety_checker=None, requires_safety_checker=False)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, str(ADAPTER)).merge_and_unload()
    # prompts on the CPU, once (fp32 there), then only the embeddings go to the GPU
    te, tok = pipe.text_encoder.float(), pipe.tokenizer
    emb = {}
    with torch.no_grad():
        for k, text in {**(prompts or PROMPT), "neg": NEG}.items():
            ids = tok(text, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt")
            emb[k] = te(ids.input_ids)[0].half().cuda()
    pipe.text_encoder = None
    pipe.unet.cuda(); pipe.vae.cuda()
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe, emb


def lung_masks(imgs):
    """PSPNet (torchxrayvision) left + right lung on the CPU, dilated 12 px and feathered."""
    import torchxrayvision as xrv
    seg = xrv.baseline_models.chestx_det.PSPNet().eval()
    k = [seg.targets.index("Left Lung"), seg.targets.index("Right Lung")]
    x = torch.from_numpy(np.stack(imgs)).float().div(255).mul(2048).sub(1024).unsqueeze(1)
    with torch.no_grad():
        m = (torch.sigmoid(seg(x))[:, k] > 0.5).any(1).numpy()
    hard = [binary_dilation(mi, iterations=12) for mi in m]
    soft = [np.clip(gaussian_filter(h.astype(np.float32), 6), 0, 1) for h in hard]
    return hard, soft


def edit(pipe, emb, img, hard, soft, label, strength, seed):
    rgb = Image.fromarray(img).convert("RGB")
    out = pipe(prompt_embeds=emb[label], negative_prompt_embeds=emb["neg"], image=rgb,
               mask_image=Image.fromarray((hard * 255).astype(np.uint8)), strength=strength,
               num_inference_steps=STEPS, guidance_scale=CFG, generator=torch.Generator("cuda").manual_seed(seed)).images[0]
    gen = np.asarray(out.convert("L"), dtype=np.float32)
    return np.clip(img * (1 - soft) + gen * soft, 0, 255).round().astype(np.uint8)  # outside the lungs: source pixels


def sources(n_pairs, offset=0):
    pool = select(read_manifest(A), {"split": "train", "subset_rank": {"ge": BASE_PER_CLASS}})
    half = n_pairs // 2
    picks = [pool[pool.label == c].sort_values("subset_rank").iloc[offset:offset + n] for c, n in ((0, half),
                                                                                                   (1, n_pairs - half))]
    # interleave normal / pneumonia sources so every prefix (E3 = first 125 pairs) is balanced
    a, b = (p.reset_index(drop=True) for p in picks)
    return pd.concat([a, b]).sort_index(kind="stable").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prototype", action="store_true")
    ap.add_argument("--pairs", type=int, default=250)
    ap.add_argument("--strength", type=float, default=0.8)
    ap.add_argument("--name", default="cf_lung")
    ap.add_argument("--mem-frac", type=float, default=0.35)
    args = ap.parse_args()

    pipe, emb = load_pipe(args.mem_frac)
    if args.prototype:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        src = sources(4, offset=300)  # not the images a full run uses first
        imgs = [to_square(load_gray(ROOT / p), 512, "squash") for p in src.image_path]
        hard, soft = lung_masks(imgs)
        strengths = [0.6, 0.8, 1.0]
        fig, axes = plt.subplots(len(src), 1 + 2 * len(strengths), figsize=(2.2 * (1 + 2 * len(strengths)), 2.4 * len(src)))
        for i, r in src.iterrows():
            axes[i, 0].imshow(imgs[i], cmap="gray"); axes[i, 0].contour(hard[i], levels=[0.5], colors="cyan", linewidths=0.5)
            axes[i, 0].set_title(f"source ({'pneu' if r.label else 'norm'})", fontsize=7)
            for j, s in enumerate(strengths):
                for c in (0, 1):
                    ax = axes[i, 1 + 2 * j + c]
                    ax.imshow(edit(pipe, emb, imgs[i], hard[i], soft[i], c, s, seed=1000 + i), cmap="gray", vmin=0, vmax=255)
                    ax.set_title(f"s={s} -> {'pneumonia' if c else 'normal'}", fontsize=7)
        for ax in axes.flat:
            ax.axis("off")
        plt.tight_layout()
        out = ROOT / "reports/results/synth_probe/cf_prototype.png"
        plt.savefig(out, dpi=70)
        print("prototype written", out, "peak GPU MB", torch.cuda.max_memory_allocated() // 2**20)
        return

    out = ROOT / "data/synthetic" / args.name
    for c in ("NORMAL", "PNEUMONIA"):
        (out / c).mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.csv"
    done = pd.read_csv(man_path) if man_path.exists() else pd.DataFrame()
    src = sources(args.pairs)
    rows = done.to_dict("records")
    for k in range(0, len(src), 8):
        chunk = src.iloc[k:k + 8]
        if len(done) and set(chunk.index) <= set(done.pair):
            continue
        imgs = [to_square(load_gray(ROOT / p), 512, "squash") for p in chunk.image_path]
        hard, soft = lung_masks(imgs)
        for (pair, r), img, h, s in zip(chunk.iterrows(), imgs, hard, soft):
            for c, cname in ((0, "NORMAL"), (1, "PNEUMONIA")):
                fn = f"cf_{cname.lower()}_{pair:05d}.png"
                Image.fromarray(edit(pipe, emb, img, h, s, c, args.strength, seed=10_000 + pair)).save(out / cname / fn)
                rows.append({"filename": fn, "relative_path": f"{cname}/{fn}", "label": cname, "prompt": PROMPT[c],
                             "negative_prompt": NEG, "seed": 10_000 + pair, "guidance_scale": CFG, "steps": STEPS,
                             "strength": args.strength, "source_id": r.image_id, "source_label": int(r.label), "pair": pair})
        pd.DataFrame(rows).to_csv(man_path, index=False)
        print(f"{min(k + 8, len(src))}/{len(src)} pairs, peak GPU MB {torch.cuda.max_memory_allocated() // 2**20}", flush=True)
    # classifier manifest (as scripts/prepare_synthetic.py), but synth_rank = pair: pair r is rank r in BOTH classes, so E3
    # (rank < 125) and E4 (rank < 250) always take complete pairs
    m = pd.DataFrame(rows)
    m = pd.DataFrame({"label": (m.label == "PNEUMONIA").astype(int),
                      "image_id": [f"S/{args.name}/{r}" for r in m.relative_path.str.replace(".png", "")],
                      "image_path": [f"data/synthetic/{args.name}/{r}" for r in m.relative_path],
                      "source": f"synthetic:{args.name}", "split": "train", "patient_id": m.source_id,
                      "synth_rank": m.pair, "source_id": m.source_id, "source_label": m.source_label})
    m.to_csv(ROOT / f"artifacts/manifests/synthetic_{args.name}.csv", index=False)
    print("cf done", m.label.value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
