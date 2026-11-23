"""ImageNet-pretrained backbones (via timm) with a single-logit head.

Only ImageNet weights are used. Chest-X-ray-pretrained checkpoints (e.g. torchxrayvision) are
avoided on purpose: they were trained on NIH ChestX-ray14 / RSNA, i.e. on our external test domain B.
"""

import torch
from torch import nn

# short name -> timm model id. Mix of classic CNNs, a modern CNN and two vision transformers.
ARCHS = {
    "resnet18": "resnet18.a1_in1k",
    "resnet50": "resnet50.a1_in1k",
    "densenet121": "densenet121.tv_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "vit_small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
}
# architectures that accept stochastic depth
DROP_PATH = {"efficientnet_b0", "convnext_tiny", "vit_small", "swin_tiny", "resnet50", "resnet18"}


class Classifier(nn.Module):
    """Takes grayscale [B,1,H,W] in [0,1]; replicates to RGB and applies the backbone's normalisation."""

    def __init__(self, arch: str = "convnext_tiny", pretrained: bool = True, drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0):
        super().__init__()
        import timm

        kw = {"drop_path_rate": drop_path_rate} if arch in DROP_PATH and drop_path_rate > 0 else {}
        self.arch = arch
        self.net = timm.create_model(ARCHS.get(arch, arch), pretrained=pretrained, num_classes=1,
                                     drop_rate=drop_rate, **kw)
        cfg = self.net.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        return self.net(x).squeeze(1)

    def param_groups(self, lr: float, weight_decay: float, head_lr_mult: float = 1.0):
        """AdamW groups: no weight decay on norms/biases/embeddings; optional larger LR for the new head."""
        head = set(id(p) for p in self.net.get_classifier().parameters())
        groups = {(h, d): [] for h in (0, 1) for d in (0, 1)}
        for name, p in self.net.named_parameters():
            if not p.requires_grad:
                continue
            no_decay = p.ndim <= 1 or name.endswith(".bias") or "pos_embed" in name or "cls_token" in name \
                or "relative_position" in name
            groups[(int(id(p) in head), int(not no_decay))].append(p)
        return [{"params": ps, "lr": lr * (head_lr_mult if h else 1.0), "weight_decay": weight_decay if d else 0.0}
                for (h, d), ps in groups.items() if ps]
