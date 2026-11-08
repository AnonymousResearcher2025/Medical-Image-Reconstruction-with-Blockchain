"""Train, evaluate, reconstruct, enroll, and retrieve with checkpoint provenance."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from .crypto import array_hash, merkle_root, state_hash
from .data import ImageDataset, PatientBatchSampler, corrupt, load_image, read_manifest
from .federated import fedavg, partition_clients
from .gft import GFT
from .ledger import ProvenanceLedger, verify_ledger
from .losses import LossWeights, composite_loss
from .metrics import binary_auc, reconstruction_metrics
from .model import HybridUNet
from .overlay import Overlay
from .shards import ShardStore
from .store import VectorStore


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_for(name="auto"):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def new_ledger(directory, config):
    nodes = int(config.get("nodes", 20))
    return ProvenanceLedger(
        directory,
        Overlay(nodes, min(3, nodes - 1), int(config.get("seed", 0))),
        int(config.get("faults", max(0, (nodes - 1) // 3))),
    )


def training_epoch(model, dataset, config, epoch_seed, optimizer):
    model.train()
    device = next(model.parameters()).device
    sampler = PatientBatchSampler(dataset.labels, int(config.get("batch_size", 8)), epoch_seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    weights = LossWeights(**config.get("loss", {}))
    report, batches = {}, 0
    for clean, labels in loader:
        clean, labels = clean.to(device), labels.to(device)
        layout = GFT(*clean.shape[-2:], tile_size=int(config.get("tile_size", 32)))
        shards = torch.stack([layout.fragment(image) for image in clean])
        count = int(config.get("tiles_per_image", 4))
        if count > 0 and count < shards.shape[1]:
            # Shared tile locations across identities; never use patient IDs in the mapping.
            indices = torch.randperm(shards.shape[1], device=device)[:count]
            shards = shards[:, indices]
        tiles_per_image = shards.shape[1]
        shards = shards.flatten(0, 1)
        noisy = corrupt(shards, float(config.get("sigma", 0.05)), float(config.get("mask_fraction", 0)))
        optimizer.zero_grad(set_to_none=True)
        loss, parts = composite_loss(model, noisy, shards, labels, tiles_per_image, weights)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        parts["total"] = float(loss.detach())
        for name, value in parts.items():
            report[name] = report.get(name, 0) + value
        batches += 1
    return {name: value / max(batches, 1) for name, value in report.items()}


def make_optimizer(model, config):
    identity, reconstruction = [], []
    for name, parameter in model.named_parameters():
        (identity if name.startswith(("projection.", "classifier.")) else reconstruction).append(parameter)
    return torch.optim.Adam(
        [
            {"params": reconstruction, "lr": float(config.get("lr_reconstruction", 1e-4))},
            {"params": identity, "lr": float(config.get("lr_identity", 1e-5))},
        ]
    )


def save_checkpoint(model, path, config, class_map):
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    checkpoint = {
        "state_dict": state,
        "model_config": model.config,
        "config": config,
        "class_map": class_map,
        "model_hash": state_hash(state),
    }
    torch.save(checkpoint, path)
    return checkpoint["model_hash"]


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if state_hash(checkpoint["state_dict"]) != checkpoint["model_hash"]:
        raise ValueError("Checkpoint tensor hash mismatch")
    model = HybridUNet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def reconstruct(model, image, config, nodes=20):
    layout = GFT(
        *image.shape[-2:],
        int(config.get("tile_size", 32)),
        nodes,
        config.get("placement", "hilbert"),
        int(config.get("seed", 0)),
    )
    shards = layout.fragment(image.to(next(model.parameters()).device))
    batch_size = int(config.get("inference_batch_size", 32))
    if batch_size < 1:
        raise ValueError("Inference batch size must be positive")
    reconstructed, vectors = [], []
    for batch in shards.split(batch_size):
        output, _ = model(batch)
        reconstructed.append(output)
        vectors.append(model.embed(output))
    outputs = torch.cat(reconstructed)
    vectors = torch.cat(vectors)
    return layout.reconstruct(outputs), vectors, outputs, layout


@torch.inference_mode()
def validate_reconstruction(model, rows, config):
    values = []
    model.eval()
    for index, row in enumerate(rows):
        clean = load_image(row["path"], int(config.get("image_size", 256))).to(
            next(model.parameters()).device
        )
        generator = torch.Generator(device=clean.device).manual_seed(int(config.get("seed", 0)) + index)
        noisy = corrupt(clean[None], float(config.get("sigma", 0.05)), generator=generator)[0]
        output, _, _, _ = reconstruct(model, noisy, config)
        values.append(reconstruction_metrics(clean[None], output[None]))
    return {key: float(np.mean([value[key] for value in values])) for key in values[0]} if values else {}


def train(manifest, output, config, federated=False):
    if int(config.get("image_size", 256)) < 16 or int(config.get("tile_size", 32)) < 16:
        raise ValueError("Image size and tile size must be at least 16 for this encoder")
    if int(config.get("batch_size", 8)) < 2 or int(config.get("batch_size", 8)) % 2:
        raise ValueError("Training batch size must be even and at least two")
    if int(config.get("tiles_per_image", 4)) < 0:
        raise ValueError("tiles_per_image must be nonnegative")
    if federated:
        fl = config.get("federated", {})
        if min(int(fl.get("rounds", 50)), int(fl.get("local_epochs", 5))) < 1:
            raise ValueError("Federated rounds and local_epochs must be positive")
    elif int(config.get("epochs", 10)) < 1:
        raise ValueError("Training epochs must be positive")
    seed = int(config.get("seed", 0))
    seed_all(seed)
    torch.set_num_threads(int(config.get("threads", 2)))
    rows = read_manifest(manifest)
    training = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "val"]
    if not training or not validation:
        raise ValueError("Training requires train and patient-disjoint val rows")
    class_map = {patient: i for i, patient in enumerate(sorted({r["patient_id"] for r in training}))}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model.pt").exists() or (output / "ledger/manifest.json").exists():
        raise ValueError("Training output must be fresh; choose another run directory")
    (output / "config.json").write_text(json.dumps(config, indent=2))
    ledger = new_ledger(output / "ledger", config)
    model = HybridUNet(
        len(class_map), int(config.get("base_channels", 64)), int(config.get("embedding_dim", 256))
    ).to(device_for(config.get("device", "auto")))
    history = []
    image_size = int(config.get("image_size", 256))
    if federated:
        fl = config.get("federated", {})
        dataset = ImageDataset(training, image_size, class_map)
        partitions = partition_clients(
            dataset.labels,
            int(fl.get("clients", 5)),
            fl.get("partition", "iid"),
            float(fl.get("alpha", 0.5)),
            seed,
        )
        (output / "partitions.json").write_text(json.dumps(partitions))
        for round_index in range(int(fl.get("rounds", 50))):
            states, counts, client_metrics, client_hashes = [], [], [], []
            for client_id, indices in enumerate(partitions):
                local = copy.deepcopy(model)
                optimizer = make_optimizer(local, config)
                local_dataset = ImageDataset([training[i] for i in indices], image_size, class_map)
                for epoch in range(int(fl.get("local_epochs", 5))):
                    metric = training_epoch(
                        local,
                        local_dataset,
                        config,
                        seed + round_index * 1000 + client_id * 10 + epoch,
                        optimizer,
                    )
                state = {name: value.detach().cpu().clone() for name, value in local.state_dict().items()}
                states.append(state)
                counts.append(len(indices))
                client_metrics.append(metric)
                client_hashes.append(state_hash(state))
            # Uniform is Algorithm 2. Sample-weighting is an explicit optional extension.
            model.load_state_dict(fedavg(states, counts if fl.get("sample_weighted", False) else None))
            address = save_checkpoint(model, output / "model.pt", config, class_map)
            proof = ledger.commit(
                {
                    "kind": "ANCHOR",
                    "round": round_index + 1,
                    "model_hash": address,
                    "client_hashes": client_hashes,
                    "partition": fl.get("partition", "iid"),
                }
            )
            metrics = validate_reconstruction(model, validation, config)
            record = {
                "round": round_index + 1,
                "clients": client_metrics,
                "validation": metrics,
                "proof": proof,
                "communication": {
                    "mode": "simulated_dense_fp32_payloads_not_network_measurement",
                    "client_upload_bytes": sum(t.numel() * t.element_size() for t in states[0].values())
                    * len(states),
                    "anchor_certificate_bytes": proof["certificate_bytes"],
                },
            }
            history.append(record)
            print(json.dumps(record), flush=True)
    else:
        dataset, optimizer = ImageDataset(training, image_size, class_map), make_optimizer(model, config)
        for epoch in range(int(config.get("epochs", 10))):
            metrics = training_epoch(model, dataset, config, seed + epoch, optimizer)
            address = save_checkpoint(model, output / "model.pt", config, class_map)
            proof = ledger.commit({"kind": "ANCHOR", "epoch": epoch + 1, "model_hash": address})
            record = {
                "epoch": epoch + 1,
                "training": metrics,
                "validation": validate_reconstruction(model, validation, config),
                "proof": proof,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
    if not history:
        raise ValueError("Training epochs/rounds must be positive")
    report = {
        "model_hash": address,
        "history": history,
        "ledger": verify_ledger(ledger.directory, ledger.genesis),
        "parameters": sum(p.numel() for p in model.parameters()),
        "data_counts": {
            split: sum(r["split"] == split for r in rows) for split in ["train", "val", "enroll", "query"]
        },
    }
    (output / "training.json").write_text(json.dumps(report, indent=2))
    return report


def evaluate(manifest, run, sigmas=(0, 0.02, 0.05, 0.10), mask_fraction=0.0):
    run = Path(run)
    model, checkpoint = load_checkpoint(run / "model.pt")
    config, address = checkpoint["config"], checkpoint["model_hash"]
    torch.set_num_threads(int(config.get("threads", 2)))
    seed_all(int(config.get("seed", 0)))
    rows = read_manifest(manifest)
    ledger = new_ledger(run / "ledger", config)
    # Require the loaded checkpoint to be anchored, not merely self-hashed.
    if not any(
        b["header"]["transaction"].get("kind") == "ANCHOR"
        and b["header"]["transaction"].get("model_hash") == address
        for b in ledger.blocks
    ):
        raise ValueError("Model checkpoint has no matching ledger anchor")
    store = VectorStore(run / "vectors", ledger)
    shard_store = ShardStore(run / "shards")
    grouped = {}
    for row in rows:
        if row["split"] == "enroll":
            image = load_image(row["path"], int(config.get("image_size", 256)))
            _, vectors, _, _ = reconstruct(model, image, config)
            grouped.setdefault(row["patient_id"], []).append(vectors.cpu().numpy())
    for patient, vectors in grouped.items():
        store.enroll(patient, np.concatenate(vectors), address)
    query_rows = [r for r in rows if r["split"] == "query"]
    if not grouped or not query_rows:
        raise ValueError("Retrieval evaluation requires enroll and query rows")
    results = []
    preview = None
    for sigma in sigmas:
        correct, scores, labels, quality, cosines = 0, [], [], [], []
        for query_index, row in enumerate(query_rows):
            clean = load_image(row["path"], int(config.get("image_size", 256)))
            generator = torch.Generator().manual_seed(int(config.get("seed", 0)) + query_index)
            noisy = corrupt(clean[None], float(sigma), mask_fraction, generator)[0]
            output, vectors, shards, layout = reconstruct(model, noisy, config)
            query = F.normalize(vectors.mean(0), dim=0).cpu().numpy()
            shard_proof = shard_store.put(shards, layout)
            ledger.commit(
                {
                    "kind": "RECONSTRUCT",
                    "input_hash": array_hash(noisy.numpy()),
                    "shard_merkle_root": shard_proof["merkle_root"],
                    "shard_manifest_hash": shard_proof["manifest_hash"],
                    "isv_merkle_root": merkle_root([array_hash(v.cpu().numpy()) for v in vectors]),
                    "model_hash": address,
                    "sigma": float(sigma),
                    "mask_fraction": float(mask_fraction),
                    "placement": config.get("placement", "hilbert"),
                    "tile_nodes": [tile.node for tile in layout.tiles],
                }
            )
            ranking = store.query(query, address, top_k=len(grouped))["ranking"]
            correct += ranking[0]["patient_id"] == row["patient_id"]
            for item in ranking:
                scores.append(item["score"])
                labels.append(item["patient_id"] == row["patient_id"])
            quality.append(reconstruction_metrics(clean[None], output.cpu()[None]))
            with torch.inference_mode():
                clean_v = model.embed(layout.fragment(clean))
            cosines.append(
                float(F.cosine_similarity(vectors.mean(0)[None], clean_v.mean(0)[None]).clamp(-1, 1))
            )
            if preview is None and sigma > 0:
                preview = (clean, noisy, output.cpu())
        results.append(
            {
                "sigma": float(sigma),
                "mask_fraction": float(mask_fraction),
                "rank1": correct / len(query_rows),
                "roc_auc": binary_auc(labels, scores),
                "queries": len(query_rows),
                "patients": len(grouped),
                "embedding_cosine": float(np.mean(cosines)),
                **{key: float(np.mean([v[key] for v in quality])) for key in quality[0]},
            }
        )
    suffix = f"-mask-{mask_fraction:g}" if mask_fraction else ""
    if preview:
        array = torch.cat(preview, dim=-1).squeeze(0).clamp(0, 1).numpy()
        Image.fromarray((array * 255).astype(np.uint8)).save(run / f"reconstruction{suffix}.png")
    report = {
        "model_hash": address,
        "results": results,
        "ledger": verify_ledger(ledger.directory, ledger.genesis),
    }
    (run / f"evaluation{suffix}.json").write_text(json.dumps(report, indent=2))
    return report
