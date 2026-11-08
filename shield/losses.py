"""Differentiable Eq. 2 objective, local-window SSIM, and batch-hard triplets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def ssim(x, y, window=7):
    """Uniform local window SSIM, unit data range, no padded boundary samples."""
    window = min(window, x.shape[-2], x.shape[-1])
    pool = lambda value: F.avg_pool2d(value, window, stride=1)
    mx, my = pool(x), pool(y)
    vx, vy, cov = pool(x * x) - mx * mx, pool(y * y) - my * my, pool(x * y) - mx * my
    score = ((2 * mx * my + 0.01**2) * (2 * cov + 0.03**2)) / (
        (mx * mx + my * my + 0.01**2) * (vx + vy + 0.03**2)
    ).clamp_min(1e-12)
    return score.flatten(1).mean(1)


def batch_hard_triplet(v, labels, margin=0.2):
    distances = torch.cdist(v, v)
    same = labels[:, None].eq(labels[None, :])
    same.fill_diagonal_(False)
    different = labels[:, None].ne(labels[None, :])
    valid = same.any(1) & different.any(1)
    if not valid.any():
        return v.sum() * 0
    positive = distances.masked_fill(~same, -float("inf")).max(1).values
    negative = distances.masked_fill(~different, float("inf")).min(1).values
    return F.relu(positive[valid] - negative[valid] + margin).mean()


@dataclass
class LossWeights:
    pixel: float = 8.0
    ssim: float = 1.5
    semantic: float = 1.0  # unspecified in the paper, explicitly selected here
    classification: float = 15.0
    embedding: float = 8.0
    triplet: float = 5.0


def composite_loss(model, noisy, clean, labels, tiles_per_image=1, weights=None):
    w = weights or LossWeights()
    reconstructed, _ = model(noisy)
    reconstructed_v = model.embed(reconstructed)
    with torch.no_grad():
        original_v = model.embed(clean)
    semantic = (1 - F.cosine_similarity(reconstructed_v, original_v)).mean()
    # Each identity is represented by its mean tile vector before supervised losses.
    v = F.normalize(reconstructed_v.reshape(len(labels), tiles_per_image, -1).mean(1), dim=1)
    pixel, structural = F.l1_loss(reconstructed, clean), 1 - ssim(reconstructed, clean).mean()
    classification = F.cross_entropy(model.classifier(v), labels)
    same = labels[:, None].eq(labels[None, :])
    same.fill_diagonal_(False)
    embedding = (1 - v @ v.T)[same].mean() if same.any() else v.sum() * 0
    triplet = batch_hard_triplet(v, labels)
    parts = {
        "pixel": pixel,
        "ssim": structural,
        "semantic": semantic,
        "classification": classification,
        "embedding": embedding,
        "triplet": triplet,
    }
    total = sum(getattr(w, name) * value for name, value in parts.items())
    return total, {name: float(value.detach()) for name, value in parts.items()}
