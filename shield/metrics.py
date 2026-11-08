"""Reconstruction and patient retrieval metrics without optional sklearn dependencies."""

import numpy as np
import torch.nn.functional as F

from .losses import ssim


def reconstruction_metrics(original, reconstructed):
    mse = float(F.mse_loss(original, reconstructed))
    return {
        "psnr_db": float(-10 * np.log10(max(mse, 1e-12))),
        "ssim": float(ssim(original, reconstructed).mean()),
    }


def binary_auc(labels, scores):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))
