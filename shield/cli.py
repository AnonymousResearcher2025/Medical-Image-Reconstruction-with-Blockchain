from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import yaml
from PIL import Image

from .crypto import state_hash
from .data import load_image, make_synthetic, prepare_brats, prepare_nih
from .engine import evaluate, load_checkpoint, reconstruct, train
from .ledger import ProvenanceLedger, QuorumError, verify_ledger
from .overlay import Overlay


def read_config(path):
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError("Configuration must be a mapping")
    return config


def build_parser():
    parser = argparse.ArgumentParser(description="SHIELD research reconstruction and signed provenance")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["train", "federated"]:
        item = commands.add_parser(name)
        item.add_argument("--manifest", required=True)
        item.add_argument("--config", required=True)
        item.add_argument("--output", required=True)
    item = commands.add_parser("evaluate")
    item.add_argument("--manifest", required=True)
    item.add_argument("--run", required=True)
    item.add_argument("--sigmas", nargs="+", type=float, default=[0, 0.02, 0.05, 0.10])
    item.add_argument("--mask-fraction", type=float, default=0)
    item = commands.add_parser("infer")
    item.add_argument("--checkpoint", required=True)
    item.add_argument("--image", required=True)
    item.add_argument("--output", required=True)
    item = commands.add_parser("verify-ledger")
    item.add_argument("--ledger", required=True)
    item.add_argument(
        "--manifest-hash", required=True, help="Manifest fingerprint retained outside the ledger"
    )
    item.add_argument("--head", help="Retained ledger head to detect removal of the chain suffix")
    item = commands.add_parser("verify-checkpoint")
    item.add_argument("--checkpoint", required=True)
    item.add_argument("--ledger", required=True)
    item.add_argument("--manifest-hash", required=True)
    item = commands.add_parser("synthetic")
    item.add_argument("--output", required=True)
    item.add_argument("--seed", type=int, default=0)
    item = commands.add_parser("demo")
    item.add_argument("--output", default="runs/demo")
    item.add_argument("--config", default="configs/demo.yaml")
    item = commands.add_parser("prepare-nih")
    item.add_argument("--metadata", required=True)
    item.add_argument("--images", required=True)
    item.add_argument("--output", required=True)
    item.add_argument("--train-patients", type=int, default=500)
    item.add_argument("--retrieval-patients", type=int, default=220)
    item.add_argument("--seed", type=int, default=0)
    item = commands.add_parser("prepare-brats")
    item.add_argument("--root", required=True)
    item.add_argument("--output", required=True)
    item.add_argument("--cases", type=int, default=50)
    item.add_argument("--seed", type=int, default=0)
    item = commands.add_parser("benchmark-consensus")
    item.add_argument("--output", required=True)
    item.add_argument("--nodes", type=int, nargs="+", default=[20, 50, 100])
    item.add_argument("--transactions", type=int, default=10)
    item.add_argument("--fault-rate", type=float, default=0.1)
    item.add_argument("--fault", choices=["none", "crash", "byzantine"], default="none")
    return parser


def benchmark(args):
    if args.transactions < 1 or not 0 <= args.fault_rate <= 1:
        raise ValueError("Invalid transaction count or fault rate")
    output = Path(args.output)
    if output.exists():
        raise ValueError("Benchmark output must not exist")
    output.mkdir(parents=True)
    results = []
    for nodes in args.nodes:
        overlay = Overlay(nodes, min(3, nodes - 1))
        ledger = ProvenanceLedger(output / f"nodes-{nodes}", overlay, (nodes - 1) // 3)
        faults = list(range(int(nodes * args.fault_rate)))
        durations, failures = [], 0
        for i in range(args.transactions):
            started = time.perf_counter()
            try:
                ledger.commit(
                    {"kind": "BENCHMARK", "index": i},
                    crashed=faults if args.fault == "crash" else (),
                    byzantine=faults if args.fault == "byzantine" else (),
                )
            except QuorumError:
                failures += 1
            durations.append((time.perf_counter() - started) * 1000)
        results.append(
            {
                "nodes": nodes,
                "transactions": args.transactions,
                "fault": args.fault,
                "fault_rate": args.fault_rate,
                "commit_success": 1 - failures / args.transactions,
                "mean_commit_ms": statistics.mean(durations),
                "mode": "single_process_serial_signing",
                "ledger_bytes": (ledger.directory / "blocks.jsonl").stat().st_size,
            }
        )
    (output / "benchmark.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    args = build_parser().parse_args()
    if args.command in {"train", "federated"}:
        result = train(args.manifest, args.output, read_config(args.config), args.command == "federated")
    elif args.command == "evaluate":
        result = evaluate(args.manifest, args.run, args.sigmas, args.mask_fraction)
    elif args.command == "infer":
        model, checkpoint = load_checkpoint(args.checkpoint)
        torch.set_num_threads(int(checkpoint["config"].get("threads", 2)))
        image = load_image(args.image, int(checkpoint["config"].get("image_size", 256)))
        output, vectors, _, _ = reconstruct(model, image, checkpoint["config"])
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((output.cpu().squeeze().numpy() * 255).astype("uint8")).save(destination)
        result = {
            "output": str(destination),
            "embedding_dim": vectors.shape[1],
            "model_hash": checkpoint["model_hash"],
            "anchored": False,
        }
    elif args.command == "verify-ledger":
        result = verify_ledger(args.ledger, args.manifest_hash, args.head)
    elif args.command == "verify-checkpoint":
        result = verify_ledger(args.ledger, args.manifest_hash)
        model, checkpoint = load_checkpoint(args.checkpoint)
        address = state_hash(model.state_dict())
        blocks = [json.loads(line) for line in (Path(args.ledger) / "blocks.jsonl").read_text().splitlines()]
        anchors = [
            block["hash"]
            for block in blocks
            if block["header"]["transaction"].get("kind") == "ANCHOR"
            and block["header"]["transaction"].get("model_hash") == address
        ]
        if not anchors:
            raise ValueError("Checkpoint is not anchored in the verified ledger")
        result = {
            "valid": True,
            "model_hash": address,
            "anchor_blocks": anchors,
            "manifest_hash": result["manifest_hash"],
        }
    elif args.command == "synthetic":
        result = {"manifest": str(make_synthetic(args.output, args.seed))}
    elif args.command == "demo":
        output, config = Path(args.output), read_config(args.config)
        if output.exists():
            raise ValueError("Demo output must not exist; choose a fresh directory")
        manifest = make_synthetic(
            output / "synthetic", int(config.get("seed", 0)), int(config.get("image_size", 64))
        )
        centralized = train(manifest, output / "centralized", config)
        retrieval = evaluate(manifest, output / "centralized")
        federated = train(manifest, output / "federated", config, federated=True)
        result = {
            "data": "synthetic_only",
            "centralized": centralized,
            "evaluation": retrieval,
            "federated": federated,
        }
        (output / "summary.json").write_text(json.dumps(result, indent=2))
    elif args.command == "prepare-nih":
        prepare_nih(
            args.metadata, args.images, args.output, args.train_patients, args.retrieval_patients, args.seed
        )
        result = {"manifest": args.output}
    elif args.command == "prepare-brats":
        result = {"manifest": str(prepare_brats(args.root, args.output, args.cases, args.seed))}
    else:
        result = benchmark(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
