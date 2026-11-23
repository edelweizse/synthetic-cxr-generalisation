"""Text-to-image sampling from an SD-1.5 LoRA with the generative component's settings (one prompt per class, the shared
negative prompt, CFG 4.5, 30 steps, 512 px; see data/synthetic/sd_lora/manifest.csv), resumable.

    .venv-gen/bin/python scripts/generator/sample.py --adapter data/generator/lora_cx --name cx_sd --per-class 1000
    .venv/bin/python scripts/prepare_synthetic.py data/synthetic/cx_sd --name cx_sd   # -> artifacts/manifests/synthetic_cx_sd.csv
Output: data/synthetic/<name>/{NORMAL,PNEUMONIA}/synth_<class>_<i>.png (grayscale) + manifest.csv (prompt, seed, ...).
Seeds are fixed per (class, index), so an interrupted run continues with the same images.
--captions <metadata.jsonl> (generator v3): image i of a class is drawn with the i-th caption of that class's training
captions, shuffled with --seed (cycled if more images than captions), i.e. the real label / view / finding mix.
--balance-view (view-captioned LoRA, prepare_chexpert.py view): the class prompt + ", AP view" / ", PA view" alternately, so
both classes are 50/50 AP/PA.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/synth_cf"))

import numpy as np
import pandas as pd
import torch

from cf_generate import CFG, NEG, PROMPT, SD_REPO, STEPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--per-class", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg", type=float, default=CFG, help="classifier-free guidance scale")
    ap.add_argument("--captions", default=None, help="metadata.jsonl of the generator's training captions")
    ap.add_argument("--balance-view", action="store_true", help="alternate ', AP view' / ', PA view' in each class prompt")
    args = ap.parse_args()

    from diffusers import StableDiffusionPipeline
    from huggingface_hub import snapshot_download
    from peft import PeftModel

    out = ROOT / "data/synthetic" / args.name
    names = {0: "NORMAL", 1: "PNEUMONIA"}
    prompt = {y: [f"{PROMPT[y]}, {v} view" for v in ("AP", "PA")] if args.balance_view else [PROMPT[y]] for y in (0, 1)}
    if args.captions:
        cap = pd.read_json(ROOT / args.captions, lines=True)
        rng = np.random.default_rng(args.seed)
        prompt = {y: list(rng.permutation(cap.text[cap.file_name.str.startswith(names[y])].to_numpy())) for y in (0, 1)}

    def text(y, i):
        return prompt[y][(i - 1) % len(prompt[y])]
    todo = [(y, i) for i in range(1, args.per_class + 1) for y in (0, 1)  # classes alternate: partial sets are balanced
            if not (out / names[y] / f"synth_{names[y].lower()}_{i:05d}.png").exists()]
    if todo:
        local = snapshot_download(SD_REPO, allow_patterns=[
            "model_index.json", "scheduler/*", "tokenizer/*", "feature_extractor/*",
            "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
            "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors"])
        pipe = StableDiffusionPipeline.from_pretrained(local, variant="fp16", torch_dtype=torch.float16,
                                                       safety_checker=None, requires_safety_checker=False)
        pipe.unet = PeftModel.from_pretrained(pipe.unet, str(ROOT / args.adapter)).merge_and_unload()
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
    print(f"{len(todo)} of {2 * args.per_class} images to generate", flush=True)
    for k in range(0, len(todo), args.batch):
        chunk = [t for t in todo[k:k + args.batch]]
        for y in (0, 1):  # one class per call
            part = [i for yy, i in chunk if yy == y]
            if not part:
                continue
            seeds = [args.seed * 1_000_000 + y * 100_000 + i for i in part]
            imgs = pipe([text(y, i) for i in part], negative_prompt=[NEG] * len(part), guidance_scale=args.cfg,
                        num_inference_steps=STEPS, height=512, width=512,
                        generator=[torch.Generator("cuda").manual_seed(s) for s in seeds]).images
            (out / names[y]).mkdir(parents=True, exist_ok=True)
            for i, img in zip(part, imgs):
                img.convert("L").save(out / names[y] / f"synth_{names[y].lower()}_{i:05d}.png")
        if k % 100 < args.batch:
            print(f"{min(k + args.batch, len(todo))}/{len(todo)}", flush=True)
    rows = [{"filename": f"synth_{names[y].lower()}_{i:05d}.png",
             "relative_path": f"{names[y]}/synth_{names[y].lower()}_{i:05d}.png", "label": names[y], "prompt": text(y, i),
             "negative_prompt": NEG, "seed": args.seed * 1_000_000 + y * 100_000 + i, "guidance_scale": args.cfg,
             "steps": STEPS, "adapter": args.adapter} for y in (0, 1) for i in range(1, args.per_class + 1)]
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    print("sampling done", out, flush=True)


if __name__ == "__main__":
    main()
