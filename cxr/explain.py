"""Grad-CAM for the classifier, covering CNN (NCHW), Swin (NHWC) and ViT (token) feature maps.

Heatmaps show which regions raised the pneumonia logit. They are a qualitative sanity check (is the
model looking at the lungs, or at text markers and borders?), not a localisation of disease.
"""

import numpy as np
import torch
import torch.nn.functional as F


def target_layer(model):
    net, arch = model.net, model.arch
    if arch.startswith("resnet"):
        return net.layer4[-1], "nchw"
    if arch.startswith("densenet"):
        return net.features, "nchw"
    if arch.startswith("efficientnet"):
        return net.bn2, "nchw"
    if arch.startswith("convnext"):
        return net.stages[-1], "nchw"
    if arch.startswith("swin"):
        return net.layers[-1], "nhwc"
    if arch.startswith(("vit", "deit")):
        return net.blocks[-1].norm1, "tokens"
    raise ValueError(f"no Grad-CAM target for {arch}")


def _to_nchw(t, layout, model):
    if layout == "nchw":
        return t
    if layout == "nhwc":
        return t.permute(0, 3, 1, 2)
    n_prefix = getattr(model.net, "num_prefix_tokens", 1)
    t = t[:, n_prefix:]
    side = int(round(t.shape[1] ** 0.5))
    return t.reshape(t.shape[0], side, side, t.shape[2]).permute(0, 3, 1, 2)


def gradcam(model, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """x: [B,1,H,W] in [0,1]. Returns (heatmaps [B,H,W] in [0,1], probabilities [B])."""
    layer, layout = target_layer(model)
    store = {}
    handle = layer.register_forward_hook(lambda _m, _i, out: store.__setitem__("act", out))
    model.eval()
    try:
        with torch.enable_grad():
            logits = model(x.clone().requires_grad_(True))  # input grad keeps a graph even if the backbone is frozen
            # autograd.grad instead of a tensor hook + backward: no reference cycle and no parameter
            # .grad buffers, so nothing stays on the GPU between calls
            grad = torch.autograd.grad(logits.sum(), store["act"])[0]
        act = store.pop("act").detach()
    finally:
        handle.remove()
        store.clear()
    act, grad = _to_nchw(act.float(), layout, model), _to_nchw(grad.float(), layout, model)
    cam = F.relu((grad.mean(dim=(2, 3), keepdim=True) * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam - cam.amin(dim=(1, 2), keepdim=True)
    cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return cam.detach().cpu().numpy(), torch.sigmoid(logits.detach().float()).cpu().numpy()


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """uint8 grayscale image + [0,1] heatmap -> uint8 RGB overlay (jet colormap)."""
    import matplotlib

    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    base = np.repeat((img.astype(np.float32) / 255)[..., None], 3, axis=2)
    return (np.clip((1 - alpha) * base + alpha * heat, 0, 1) * 255).astype(np.uint8)
