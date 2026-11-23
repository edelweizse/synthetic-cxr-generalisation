"""Research demo: pneumonia score for a chest X-ray, with Grad-CAM. NOT a medical device.

    streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import torch

from cxr.explain import gradcam, overlay
from cxr.imaging import load_gray, to_square
from cxr.models import Classifier
from cxr.quality import contrast, gauss

MODELS_DIR = ROOT / "models"

st.set_page_config(page_title="CXR pneumonia (research demo)", layout="wide")


@st.cache_resource
def load(model_dir: str):
    ckpt = torch.load(Path(model_dir) / "model.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = Classifier(cfg["arch"], pretrained=False, drop_rate=cfg.get("drop_rate", 0.0),
                       drop_path_rate=cfg.get("drop_path_rate", 0.0))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def card_of(d: Path) -> dict:
    return json.loads((d / "card.json").read_text())


def model_input(img: np.ndarray, card: dict, sigma: float = 0.0, factor: float = 1.0) -> np.ndarray:
    """224-px model input with a controlled quality change applied first (same code as the causal test, cxr/quality.py).
    The change is applied to the plain resized image; models with other preprocessing (e.g. equalisation) then get
    their own transform on top, as at training time."""
    size = card["image_size"]
    x = torch.from_numpy(to_square(img, size, "squash")).float().div(255)[None, None]
    x = contrast(gauss(x, sigma), factor)
    x = (x[0, 0].numpy() * 255).round().astype(np.uint8)
    return x if card["resize_mode"] == "squash" else to_square(x, size, card["resize_mode"])


@torch.no_grad()
def score(model, x_img: np.ndarray) -> float:
    return float(torch.sigmoid(model(torch.from_numpy(x_img).float().div(255)[None, None])).reshape(-1)[0])


st.error("**Research prototype, not for clinical use.** The score is the output of an experimental model "
         "trained on one public source dataset (see the sidebar). It is not a diagnosis and not a calibrated probability.", icon="⚠️")
st.title("Chest X-ray pneumonia classifier")

models = sorted(d for d in MODELS_DIR.glob("*") if (d / "model.pt").exists() and (d / "card.json").exists())
if not models:
    st.warning("No exported models in `models/`. Run `python scripts/export_model.py --exp E1 --arch <arch>`.")
    st.stop()

with st.sidebar:
    # default: the CheXpert-trained model (study 2), else the best Kermany configuration; seeds chosen on source validation
    names = [d.name for d in models]
    default = next((names.index(n) for n in ("CX_E2_ERASE_efficientnet_b0", "E2_ERASE_efficientnet_b0") if n in names), 0)
    choice = st.selectbox("Model", models, index=default, format_func=lambda d: d.name)
    card = card_of(choice)
    st.markdown(f"**{card['arch']}**, {card['experiment']}: {card['experiment_doc']}  \n"
                f"trained on {card['n_train']} images · threshold {card['threshold']:.3f} (chosen on source validation)")
    source = card.get("source", "kermany")
    st.markdown("Training data: **" + ("CheXpert, adults (Stanford)" if source.startswith("chexpert") else "Kermany, children 1–5 y (Guangzhou)") + "**")
    rows = [{"test set": k.replace("A_test:label", "Kermany test").replace("C_test:label", "CheXpert test")
                         .replace("B_external:label_any", "RSNA: opacity vs all others")
                         .replace("B_external:label", "RSNA adults: opacity vs normal"),
             "AUROC": v["auroc"], "sens": v["sensitivity"], "spec": v["specificity"]}
            for k, v in card["metrics"].items()]
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("test set").round(3))
    show_cam = st.checkbox("Grad-CAM heatmap", value=True)

up = st.file_uploader("Chest X-ray (PNG, JPEG or DICOM)", type=["png", "jpg", "jpeg", "dcm"])
examples = sorted((ROOT / "app/examples").glob("*.*"))
ex = st.selectbox("…or an example image", [None] + examples, format_func=lambda p: "-" if p is None else p.name)

src = None
if up is not None:
    tmp = ROOT / "app" / ".upload" / up.name
    tmp.parent.mkdir(exist_ok=True)
    tmp.write_bytes(up.getvalue())
    src = tmp
elif ex is not None:
    src = ex

if src is not None:
    try:
        img = load_gray(src)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read image: {e}")
        st.stop()
    h, w = img.shape
    if not 0.6 <= w / h <= 2.2:
        st.warning(f"Unusual aspect ratio ({w}x{h}). Is this a frontal chest radiograph?")
    x_img = to_square(img, card["image_size"], card["resize_mode"])
    x = torch.from_numpy(x_img).float().div(255)[None, None]
    model = load(str(choice))
    cam, prob = gradcam(model, x)
    p = float(prob[0])
    positive = p >= card["threshold"]

    c1, c2 = st.columns(2)
    c1.image(img, caption=f"input {w}x{h}", width="stretch", clamp=True)
    if show_cam:
        c2.image(overlay(x_img, cam[0]), caption="Grad-CAM: regions that raised the score", width="stretch")
    else:
        c2.image(x_img, caption="model input (224 px)", width="stretch", clamp=True)

    st.metric("Model score (pneumonia)", f"{p:.3f}", help="Not a calibrated probability")
    st.progress(min(max(p, 0.0), 1.0))
    if positive:
        st.warning(f"Score ≥ threshold {card['threshold']:.3f}: **pattern the model associates with pneumonia**")
    else:
        st.success(f"Score < threshold {card['threshold']:.3f}: **no pneumonia pattern detected by the model**")
    trained_on = "adult CheXpert images (Stanford)" if source.startswith("chexpert") else "pediatric images (Guangzhou, 1–5 y)"
    st.caption(f"Known limitations: trained on {trained_on}; performance drops on radiographs from other "
               "hospitals and populations (see sidebar). Sensitive to acquisition differences "
               "(device, positioning, AP vs PA). Grad-CAM is a coarse attribution, not a lesion map.")

    st.divider()
    st.subheader("Image-quality stress test")
    st.markdown("The same image, only its **quality** changed. A model that learned *pneumonia* should not care; "
                "models trained on Kermany partly learned *blurry / low-contrast = pneumonia*, because Kermany "
                "pneumonia films are softer than its normal films (docs/FINDINGS_SYNTHETIC.md §3–4). "
                "Try it on a **normal** example.")
    s1, s2 = st.columns(2)
    sigma = s1.slider("Blur σ (pixels at 224 px)", 0.0, 3.0, 1.0, 0.25)
    factor = s2.slider("Contrast factor", 0.4, 1.0, 1.0, 0.05, help="1 = unchanged, 0.7 = 30% less contrast")
    x_deg = model_input(img, card, sigma, factor)
    p_deg = score(model, x_deg)
    d1, d2 = st.columns(2)
    d1.image(x_deg, caption="degraded model input", width="stretch", clamp=True)
    d2.metric("Score after degradation", f"{p_deg:.3f}", delta=f"{p_deg - p:+.3f} vs original",
              delta_color="inverse")
    st.markdown("**Score vs blur** at the chosen contrast, for every exported model:")
    sigmas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    curves = {d.name.removesuffix("_" + card_of(d)["arch"]): [score(load(str(d)), model_input(img, card_of(d), s_, factor)) for s_ in sigmas] for d in models}
    st.line_chart(pd.DataFrame(curves, index=pd.Index(sigmas, name="blur σ")), y_label="score")
    st.caption("In the study (healthy images, EfficientNet-B0, 2 seeds), blur σ = 1 raised the mean score by +0.33 for "
               "the baseline E1 but only +0.02 for E2_ERASE on Kermany; at σ = 2 every model was pushed to ≈ 0.9. "
               "Models trained with synthetic images (E3_sd_lora) also react strongly to lower contrast.")
