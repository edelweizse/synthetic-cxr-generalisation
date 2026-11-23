"""Does the SD-1.5 VAE erase the finding? (docs/RESEARCH_PIVOT.md §13): real CheXpert test images, original vs VAE encode + decode,
both scored by the CX_E2_ERASE teacher. If p(pneumonia) and AUROC hold, the latent space is not what limits the generator.

    HF_HUB_OFFLINE=1 .venv-gen/bin/python scripts/generator/vae_check.py encode [n per class]   # GPU, writes data/vae_check
    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/generator/vae_check.py score                 # CPU
Output: reports/results/generator/vae_check.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data/vae_check"
SD_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"  # = cf_generate.SD_REPO


@torch.no_grad()
def encode(n="200"):
    from diffusers import AutoencoderKL
    from huggingface_hub import snapshot_download
    m = pd.read_csv(ROOT / "artifacts/manifests/chexpert.csv")
    m = m[m.split == "test"].groupby("label").sample(int(n), random_state=0)
    local = snapshot_download(SD_REPO, allow_patterns=["vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors"])
    vae = AutoencoderKL.from_pretrained(local, subfolder="vae", variant="fp16", torch_dtype=torch.float16).cuda().eval()
    rows = []
    for d in ("orig", "recon"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    for _, r in m.iterrows():
        img = Image.open(ROOT / r.image_path).convert("L").resize((512, 512), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None].half().cuda()
        y = vae.decode(vae.encode(x).latent_dist.mode()).sample[0].float().mean(0).clamp(-1, 1)
        name = f"{r.patient_id}.png"
        img.save(OUT / "orig" / name)
        Image.fromarray(((y + 1) * 127.5).round().byte().cpu().numpy()).save(OUT / "recon" / name)
        rows.append(r)
    m = pd.DataFrame(rows)
    for d in ("orig", "recon"):
        m.assign(image_path=[str(OUT / d / f"{i}.png") for i in m.patient_id]).to_csv(OUT / f"{d}.csv", index=False)
    print("encoded", len(m), m.label.value_counts().to_dict())


def score():
    from sklearn.metrics import roc_auc_score
    from cxr.data import load_set
    from cxr.engine import RUNS_DIR, load_model, predict_logits, sigmoid
    torch.set_num_threads(4)
    dev = torch.device("cpu")
    teachers = [load_model(RUNS_DIR / "CX_E2_ERASE" / a / "seed0", dev)[0] for a in ("efficientnet_b0", "densenet121", "vit_small")]
    rows = []
    for d in ("orig", "recon"):
        ts = load_set(str(OUT / f"{d}.csv"), None, 224, "squash", device=None)
        y = ts.labels.numpy()
        p = np.mean([sigmoid(predict_logits(t, ts, dev, batch_size=32, amp=False)) for t in teachers], axis=0)
        rows.append(dict(set=d, n=len(y), p_normal=p[y == 0].mean(), p_pneumonia=p[y == 1].mean(), auroc=roc_auc_score(y, p)))
    r = pd.DataFrame(rows).round(4)
    r.to_csv(ROOT / "reports/results/generator/vae_check.csv", index=False)
    print(r.to_string(index=False))


if __name__ == "__main__":
    {"encode": encode, "score": score}[sys.argv[1]](*sys.argv[2:])
