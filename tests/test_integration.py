import json

import torch

from shield.data import make_synthetic
from shield.engine import evaluate, load_checkpoint, train
from shield.ledger import verify_ledger


def test_training_federation_retrieval_and_reload(tmp_path):
    manifest = make_synthetic(tmp_path / "data", image_size=32)
    config = {
        "seed": 0,
        "image_size": 32,
        "tile_size": 32,
        "base_channels": 8,
        "batch_size": 8,
        "tiles_per_image": 1,
        "epochs": 1,
        "nodes": 4,
        "faults": 1,
        "threads": 2,
        "federated": {"clients": 2, "rounds": 1, "local_epochs": 1, "partition": "dirichlet"},
    }
    run = tmp_path / "central"
    training = train(manifest, run, config)
    model, checkpoint = load_checkpoint(run / "model.pt")
    assert checkpoint["model_hash"] == training["model_hash"]
    report = evaluate(manifest, run, sigmas=[0.05])
    assert len(report["results"]) == 1 and 0 <= report["results"][0]["rank1"] <= 1
    assert verify_ledger(run / "ledger", training["ledger"]["manifest_hash"])["blocks"] == 16
    assert torch.isfinite(model.embed(torch.rand(1, 1, 32, 32))).all()
    fl = train(manifest, tmp_path / "fl", config, federated=True)
    assert len(fl["history"]) == 1
    assert fl["history"][0]["proof"]["height"] == 0
    assert json.loads((tmp_path / "fl/partitions.json").read_text())
