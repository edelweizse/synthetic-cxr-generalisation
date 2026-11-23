"""Controlled image-quality degradations, used by the causal shortcut test (scripts/synth_probe/degrade.py)
and the demo's stress test. Input: float tensor (N, 1, H, W) in [0, 1]."""

import torch
import torch.nn.functional as F


def gauss(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur, reflect padding; sigma in pixels of the model input."""
    if sigma <= 0:
        return x
    k = int(2 * round(3 * sigma) + 1)
    t = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-t ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1))


def contrast(x: torch.Tensor, f: float) -> torch.Tensor:
    """Scale deviations from each image's mean by f (f < 1 lowers contrast)."""
    m = x.mean((2, 3), keepdim=True)
    return (m + f * (x - m)).clamp(0, 1)
