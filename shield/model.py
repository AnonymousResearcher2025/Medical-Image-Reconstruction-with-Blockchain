"""Four residual stages [2,2,2,2], U-Net skips, and a normalized 256-D ISV.

Default channel width is ResNet-18's 64. GroupNorm supports singleton tile
batches; no pretrained weights or undocumented attention module is assumed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def norm(channels):
    return nn.GroupNorm(min(8, channels), channels)


class Residual(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            norm(cout),
            nn.ReLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            norm(cout),
        )
        self.skip = (
            nn.Identity()
            if cin == cout and stride == 1
            else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), norm(cout))
        )

    def forward(self, x):
        return F.relu(self.body(x) + self.skip(x))


class Up(nn.Module):
    def __init__(self, cin, skip, cout):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin + skip, cout, 3, padding=1),
            norm(cout),
            nn.ReLU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            norm(cout),
            nn.ReLU(),
        )

    def forward(self, x, skip):
        return self.body(
            torch.cat(
                [F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], dim=1
            )
        )


class HybridUNet(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 64, embedding_dim: int = 256):
        super().__init__()
        if num_classes < 1 or base_channels < 8 or base_channels % 8 or embedding_dim < 1:
            raise ValueError("Require classes >=1, embedding_dim >=1, and base_channels a multiple of 8")
        self.config = {
            "num_classes": num_classes,
            "base_channels": base_channels,
            "embedding_dim": embedding_dim,
        }
        b = base_channels
        self.stem = nn.Sequential(nn.Conv2d(1, b, 7, 2, 3, bias=False), norm(b), nn.ReLU())
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.stages = nn.ModuleList()
        cin = b
        for i, cout in enumerate([b, b * 2, b * 4, b * 8]):
            self.stages.append(nn.Sequential(Residual(cin, cout, 1 if i == 0 else 2), Residual(cout, cout)))
            cin = cout
        self.projection = nn.Linear(b * 8, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.ups = nn.ModuleList(
            [Up(b * 8, b * 4, b * 4), Up(b * 4, b * 2, b * 2), Up(b * 2, b, b), Up(b, b, b)]
        )
        self.output = nn.Conv2d(b, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def encode(self, x):
        features = [self.stem(x)]
        x = self.pool(features[0])
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        v = F.normalize(self.projection(F.adaptive_avg_pool2d(x, 1).flatten(1)), dim=1)
        return features, v

    def embed(self, x):
        return self.encode(x)[1]

    def forward(self, x):
        features, v = self.encode(x)
        decoded = features[-1]
        for up, skip in zip(self.ups, reversed(features[:-1])):
            decoded = up(decoded, skip)
        residual = self.output(
            F.interpolate(decoded, size=x.shape[-2:], mode="bilinear", align_corners=False)
        )
        # A residual image predictor provides an explicit identity initialization path.
        reconstruction = (x + residual).clamp(0, 1)
        return reconstruction, v
