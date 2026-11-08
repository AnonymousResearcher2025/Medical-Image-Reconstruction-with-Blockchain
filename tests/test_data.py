import csv

import pytest

from shield.data import make_synthetic, read_manifest, write_manifest


def test_patient_disjoint_and_query_leakage(tmp_path):
    path = make_synthetic(tmp_path / "data")
    rows = read_manifest(path)
    assert {row["split"] for row in rows} == {"train", "val", "enroll", "query"}
    with path.open() as stream:
        raw = list(csv.DictReader(stream))
    raw.append(dict(raw[0], split="query"))
    write_manifest(path, raw)
    with pytest.raises(ValueError, match="multiple times"):
        read_manifest(path)


def test_patient_overlap_rejected(tmp_path):
    path = make_synthetic(tmp_path / "data")
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    rows[-1]["patient_id"] = "train-0"
    write_manifest(path, rows)
    with pytest.raises(ValueError, match="Retrieval patients"):
        read_manifest(path)


def test_nih_preparation(tmp_path):
    from PIL import Image

    from shield.data import prepare_nih

    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Patient ID", "Image Index"])
        writer.writeheader()
        for patient in range(5):
            for index in range(2):
                name = f"{patient}-{index}.png"
                Image.new("L", (32, 32), 100 + patient).save(tmp_path / name)
                writer.writerow({"Patient ID": patient, "Image Index": name})
    output = tmp_path / "manifest.csv"
    prepare_nih(metadata, tmp_path, output, train_patients=3, retrieval_patients=2)
    rows = read_manifest(output)
    assert len(rows) == 10
    assert sum(row["split"] == "query" for row in rows) == 2


def test_brats_preparation(tmp_path):
    import numpy as np

    from shield.data import prepare_brats

    nib = pytest.importorskip("nibabel")
    root = tmp_path / "brats"
    root.mkdir()
    for patient in range(3):
        volume = np.random.default_rng(patient).random((8, 8, 80)).astype(np.float32)
        nib.save(nib.Nifti1Image(volume, np.eye(4)), root / f"patient-{patient}_t1ce.nii.gz")
    manifest = prepare_brats(root, tmp_path / "slices", cases=3)
    rows = read_manifest(manifest)
    assert len(rows) == 60
    assert sum(row["split"] == "enroll" for row in rows) == 10
    assert {row["split"] for row in rows} == {"train", "val", "enroll", "query"}
