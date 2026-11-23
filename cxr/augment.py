"""Batched, on-device training augmentation.

conventional (E2): mild, anatomically plausible perturbations: small rotation/translation/scale and
    brightness/contrast/gamma. No horizontal flips (they move the heart to the right side and the
    laterality marker) and no elastic warps (unrealistic anatomy).
acquisition: conventional + random blur, sensor-like noise and down/up-sampling to a random lower
    resolution. Targets A's acquisition shortcut (classes differ in original resolution and JPEG
    quality, so sharpness and noise texture leak the label).
erase: conventional + random erasing of a rectangle and random masking of a border frame. Targets
    reliance on image periphery (text markers, collimation edges), seen in Grad-CAM.
robust: acquisition + erase.
"""

import math

import torch
import torch.nn.functional as F

CONVENTIONAL = {"rotate_deg": 10.0, "translate": 0.06, "scale": 0.10, "brightness": 0.15,
                "contrast": 0.15, "gamma": 0.15, "p": 0.9}
_ACQ = {"blur_p": 0.5, "blur_sigma": (0.3, 1.5), "noise_p": 0.5, "noise_std": (0.005, 0.03),
        "down_p": 0.5, "down_scales": (0.35, 0.5, 0.7)}
_ERASE = {"erase_p": 0.5, "erase_area": (0.02, 0.12), "border_p": 0.5, "border_frac": (0.03, 0.12)}
PRESETS = {
    "conventional": CONVENTIONAL,
    "acquisition": {**CONVENTIONAL, **_ACQ},
    "erase": {**CONVENTIONAL, **_ERASE},
    "robust": {**CONVENTIONAL, **_ACQ, **_ERASE},
}


def _u(n, lo, hi, gen, device):
    return torch.rand(n, generator=gen, device=device) * (hi - lo) + lo


def _bernoulli(n, p, gen, device):
    return torch.rand(n, generator=gen, device=device) < p


def _gaussian_blur(x, sigma):
    """Per-sample Gaussian blur; sigma: [B] tensor (0 = identity)."""
    b, _, h, w = x.shape
    radius = 4
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    s = sigma.clamp_min(1e-3).view(b, 1)
    k = torch.exp(-0.5 * (t.view(1, -1) / s) ** 2)
    k = k / k.sum(1, keepdim=True)
    k[sigma <= 0] = (t == 0).to(x.dtype)                       # identity kernel where no blur
    xs = x.view(1, b, h, w)
    xs = F.conv2d(F.pad(xs, (radius, radius, 0, 0), mode="replicate"), k.view(b, 1, 1, -1), groups=b)
    xs = F.conv2d(F.pad(xs, (0, 0, radius, radius), mode="replicate"), k.view(b, 1, -1, 1), groups=b)
    return xs.view(b, 1, h, w)


def augment_batch(x: torch.Tensor, params: dict, gen: torch.Generator) -> torch.Tensor:
    """x: float [B,1,H,W] in [0,1] on any device; `gen` must live on the same device."""
    b, dev = x.shape[0], x.device
    h, w = x.shape[-2:]
    apply = (torch.rand(b, generator=gen, device=dev) < params.get("p", 1.0)).float()

    # geometry
    ang = _u(b, -1, 1, gen, dev) * math.radians(params["rotate_deg"]) * apply
    sc = 1 + _u(b, -1, 1, gen, dev) * params["scale"] * apply
    tx = _u(b, -1, 1, gen, dev) * params["translate"] * 2 * apply   # affine_grid coords span [-1, 1]
    ty = _u(b, -1, 1, gen, dev) * params["translate"] * 2 * apply
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

    # acquisition: resolution loss, blur, noise
    if params.get("down_p", 0) > 0:
        do = _bernoulli(b, params["down_p"], gen, dev)
        choice = torch.randint(len(params["down_scales"]), (b,), generator=gen, device=dev)
        for k, s in enumerate(params["down_scales"]):
            idx = (do & (choice == k)).nonzero().flatten()
            if len(idx):
                small = F.interpolate(x[idx], size=(max(8, int(h * s)), max(8, int(w * s))), mode="area")
                x[idx] = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    if params.get("blur_p", 0) > 0:
        lo, hi = params["blur_sigma"]
        sigma = _u(b, lo, hi, gen, dev) * _bernoulli(b, params["blur_p"], gen, dev)
        x = _gaussian_blur(x, sigma)
    if params.get("noise_p", 0) > 0:
        lo, hi = params["noise_std"]
        std = _u(b, lo, hi, gen, dev) * _bernoulli(b, params["noise_p"], gen, dev)
        x = x + torch.randn(x.shape, generator=gen, device=dev) * std.view(b, 1, 1, 1)

    # photometric
    shape = (b, 1, 1, 1)
    x = x.clamp(1e-6, 1)
    gamma = torch.exp(_u(b, -1, 1, gen, dev) * params["gamma"] * apply).view(shape)
    x = x.pow(gamma)
    mean = x.mean(dim=(2, 3), keepdim=True)
    contrast = (1 + _u(b, -1, 1, gen, dev) * params["contrast"] * apply).view(shape)
    bright = (1 + _u(b, -1, 1, gen, dev) * params["brightness"] * apply).view(shape)
    x = ((x - mean) * contrast + mean).mul(bright)

    # occlusion: random rectangle + random border frame
    if params.get("erase_p", 0) > 0 or params.get("border_p", 0) > 0:
        yy = torch.linspace(0, 1, h, device=dev).view(1, h, 1)
        xx = torch.linspace(0, 1, w, device=dev).view(1, 1, w)
        mask = torch.zeros(b, h, w, dtype=torch.bool, device=dev)
        if params.get("erase_p", 0) > 0:
            lo, hi = params["erase_area"]
            area = _u(b, lo, hi, gen, dev)
            aspect = torch.exp(_u(b, math.log(0.5), math.log(2.0), gen, dev))
            eh, ew = (area * aspect).sqrt().clamp(max=1), (area / aspect).sqrt().clamp(max=1)
            y0, x0 = _u(b, 0, 1, gen, dev) * (1 - eh), _u(b, 0, 1, gen, dev) * (1 - ew)
            rect = (yy >= y0.view(b, 1, 1)) & (yy <= (y0 + eh).view(b, 1, 1)) & \
                   (xx >= x0.view(b, 1, 1)) & (xx <= (x0 + ew).view(b, 1, 1))
            mask |= rect & _bernoulli(b, params["erase_p"], gen, dev).view(b, 1, 1)
        if params.get("border_p", 0) > 0:
            lo, hi = params["border_frac"]
            f = _u(b, lo, hi, gen, dev).view(b, 1, 1)
            frame = (yy < f) | (yy > 1 - f) | (xx < f) | (xx > 1 - f)
            mask |= frame & _bernoulli(b, params["border_p"], gen, dev).view(b, 1, 1)
        fill = _u(b, 0, 1, gen, dev).view(b, 1, 1)
        x = torch.where(mask.unsqueeze(1), fill.unsqueeze(1).expand(b, 1, h, w), x)
    return x.clamp(0, 1)
