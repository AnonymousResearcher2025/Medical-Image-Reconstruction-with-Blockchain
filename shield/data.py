"""Explicit patient/split manifests, paired synthetic data, and NIH/BraTS preparation."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

SPLITS = {"train", "val", "enroll", "query"}


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"path", "patient_id", "split"} <= set(reader.fieldnames or []):
            raise ValueError("Manifest requires path,patient_id,split columns")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty data manifest")
    seen, patients = set(), {name: set() for name in SPLITS}
    for row in rows:
        if row["split"] not in SPLITS or not row["patient_id"]:
            raise ValueError("Unknown split or missing patient identifier")
        resolved = (path.parent / row["path"]).resolve()
        if not resolved.is_file():
            raise ValueError(f"Missing image: {resolved}")
        if resolved in seen:
            raise ValueError("An image appears multiple times; enrollment/query leakage is forbidden")
        seen.add(resolved)
        row["path"] = str(resolved)
        patients[row["split"]].add(row["patient_id"])
    if patients["train"] & patients["val"]:
        raise ValueError("Training and validation patients must be disjoint")
    retrieval = patients["enroll"] | patients["query"]
    if (patients["train"] | patients["val"]) & retrieval:
        raise ValueError("Retrieval patients must be disjoint from reconstruction train/val")
    if patients["query"] - patients["enroll"]:
        raise ValueError("Every query patient must have an enrollment image")
    return rows


def load_image(path, image_size=256):
    with Image.open(path) as image:
        image = image.convert("F").resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.array(image, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Non-finite image intensities")
    lo, hi = float(array.min()), float(array.max())
    if hi > 1 or lo < 0:
        array = (array - lo) / max(hi - lo, 1e-8)
    return torch.from_numpy(array.copy()).unsqueeze(0).clamp(0, 1)


class ImageDataset(Dataset):
    def __init__(self, rows, image_size, class_map=None):
        self.rows, self.image_size = list(rows), image_size
        self.class_map = class_map or {
            name: i for i, name in enumerate(sorted({r["patient_id"] for r in rows}))
        }
        self.labels = [self.class_map[r["patient_id"]] for r in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return load_image(row["path"], self.image_size), self.labels[index]


class PatientBatchSampler(Sampler):
    """Two images per patient when available; valid triplets need >=2 patients."""

    def __init__(self, labels, batch_size=8, seed=0):
        if batch_size < 2 or batch_size % 2 or not labels:
            raise ValueError("Patient batch size must be a positive even number >= 2")
        self.groups = {label: np.where(np.array(labels) == label)[0] for label in sorted(set(labels))}
        self.batch_size, self.seed, self.length = batch_size, seed, len(labels)

    def __len__(self):
        return math.ceil(self.length / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        keys = list(self.groups)
        for _ in range(len(self)):
            selected = rng.choice(keys, min(len(keys), self.batch_size // 2), replace=False)
            batch = []
            for label in selected:
                candidates = self.groups[int(label)]
                batch.extend(rng.choice(candidates, 2, replace=len(candidates) < 2).tolist())
            yield batch


def corrupt(clean, sigma=0.05, mask_fraction=0.0, generator=None):
    if sigma < 0 or not 0 <= mask_fraction < 1:
        raise ValueError("Require nonnegative sigma and 0 <= mask_fraction < 1")
    noisy = clean + sigma * torch.randn(
        clean.shape, generator=generator, device=clean.device, dtype=clean.dtype
    )
    if mask_fraction:
        h, w = clean.shape[-2:]
        mh, mw = max(1, round(h * math.sqrt(mask_fraction))), max(1, round(w * math.sqrt(mask_fraction)))
        # One contiguous region per image/tile, rather than independent dropout pixels.
        for i in range(len(clean)):
            y = int(torch.randint(h - mh + 1, (1,), generator=generator, device=clean.device))
            x = int(torch.randint(w - mw + 1, (1,), generator=generator, device=clean.device))
            noisy[i, :, y : y + mh, x : x + mw] = 0
    return noisy.clamp(0, 1)


def make_synthetic(directory, seed=0, image_size=64):
    """Synthetic patterns only; no medical images, patient data or clinical claims."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng, rows = np.random.default_rng(seed), []
    yy, xx = np.mgrid[-1 : 1 : complex(image_size), -1 : 1 : complex(image_size)]
    definitions = [(f"train-{p}", "train", 4) for p in range(4)]
    definitions += [("val-0", "val", 3)]
    definitions += [(f"retrieval-{p}", "retrieval", 4) for p in range(3)]
    for patient, split, count in definitions:
        centers = rng.uniform(-0.55, 0.55, (4, 2))
        base = np.zeros_like(xx) + 0.15
        for x, y in centers:
            base += rng.uniform(0.15, 0.45) * np.exp(
                -((xx - x) ** 2 + (yy - y) ** 2) / rng.uniform(0.03, 0.12)
            )
        for j in range(count):
            array = np.clip(base + rng.normal(0, 0.008, base.shape), 0, 1)
            filename = f"{patient}-{j}.png"
            Image.fromarray((array * 255).astype(np.uint8)).save(directory / filename)
            current = ("enroll" if j < 2 else "query") if split == "retrieval" else split
            rows.append({"path": filename, "patient_id": patient, "split": current})
    path = directory / "manifest.csv"
    write_manifest(path, rows)
    return path


def write_manifest(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "patient_id", "split"])
        writer.writeheader()
        writer.writerows(rows)


def prepare_nih(metadata, images, output, train_patients=500, retrieval_patients=220, seed=0):
    """Select patient-disjoint cohorts from user-provided NIH metadata and images.

    This does not assume the paper's unpublished 10,000-image sample selection.
    """
    with Path(metadata).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row["Patient ID"]), []).append(Path(images).resolve() / row["Image Index"])
    eligible = sorted(p for p, paths in grouped.items() if len(paths) >= 2)
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)
    needed = train_patients + retrieval_patients
    if train_patients < 2 or retrieval_patients < 1 or len(eligible) < needed:
        raise ValueError("Insufficient patients with >=2 images for requested cohorts")
    reconstruction, retrieval = eligible[:train_patients], eligible[train_patients:needed]
    nval = max(1, round(0.15 * len(reconstruction)))
    validation = set(reconstruction[:nval])
    result = []
    for patient in reconstruction + retrieval:
        paths = sorted(grouped[patient])
        for i, path in enumerate(paths):
            split = (
                ("enroll" if i < len(paths) - 1 else "query")
                if patient in retrieval
                else ("val" if patient in validation else "train")
            )
            result.append({"path": str(path), "patient_id": patient, "split": split})
    write_manifest(output, result)
    read_manifest(output)


def prepare_brats(root, output, cases=50, seed=0):
    import nibabel as nib

    root, output = Path(root), Path(output)
    paths = sorted(root.rglob("*_t1ce.nii*"))
    if cases < 3 or len(paths) < cases:
        raise ValueError("Need >=3 available T1ce cases for train/val/retrieval")
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(paths), cases, replace=False)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    nval, nretrieval = max(1, round(cases * 0.15)), max(1, round(cases * 0.2))
    for j, index in enumerate(selected):
        path = paths[index]
        patient = path.name.split("_t1ce")[0]
        volume = np.asarray(nib.as_closest_canonical(nib.load(path)).dataobj, dtype=np.float32)
        if volume.ndim != 3 or volume.shape[2] < 80 or not np.isfinite(volume).all():
            raise ValueError(f"Invalid volume for slices 60..79: {path}")
        low, high = np.percentile(volume, [1, 99.5])
        for z in range(60, 80):
            array = np.clip((volume[:, :, z] - low) / max(high - low, 1e-8), 0, 1)
            filename = f"{patient}-{z}.png"
            Image.fromarray((array * 255).astype(np.uint8)).save(output / filename)
            split = (
                ("enroll" if z < 70 else "query")
                if j < nretrieval
                else ("val" if j < nretrieval + nval else "train")
            )
            rows.append({"path": filename, "patient_id": patient, "split": split})
    path = output / "manifest.csv"
    write_manifest(path, rows)
    read_manifest(path)
    return path
