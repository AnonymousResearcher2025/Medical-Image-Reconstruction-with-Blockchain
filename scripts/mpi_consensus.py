"""Real MPI Allgather/Bcast certificate pipeline; one ephemeral key per rank.

Run: mpiexec -n 4 python scripts/mpi_consensus.py --output runs/mpi --transactions 5
No timeout/view-change protocol: process crashes can stall MPI collectives.
--fault injects missing votes/conflicting signatures, not OS process termination.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from mpi4py import MPI

from shield.crypto import canonical, digest
from shield.ledger import certificate_weight, manifest_hash, proposal_bytes, reaches_quorum, validate_block
from shield.overlay import Overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--transactions", type=int, default=5)
    parser.add_argument("--fault", choices=["none", "crash", "byzantine"], default="none")
    parser.add_argument("--fault-rate", type=float, default=0.1)
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    rank, nodes = comm.Get_rank(), comm.Get_size()
    if nodes < 4 or args.transactions < 1 or not 0 <= args.fault_rate <= 1:
        raise ValueError("Require >=4 ranks, positive transactions, fault rate in [0,1]")
    output = Path(args.output) / f"rank-{rank}"
    if comm.allreduce(int(output.exists())):
        raise ValueError("Output must be fresh on every rank")
    output.mkdir(parents=True)
    key = Ed25519PrivateKey.generate()
    public = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    publics = comm.allgather(public)
    manifest = {
        "version": 1,
        "overlay": Overlay(nodes, min(3, nodes - 1)).metadata(),
        "faults": (nodes - 1) // 3,
        "public_keys": publics,
    }
    (output / "manifest.json").write_bytes(canonical(manifest))
    (output / "blocks.jsonl").touch()
    genesis, previous, height = manifest_hash(manifest), manifest_hash(manifest), 0
    durations = []
    for transaction in range(args.transactions):
        started = MPI.Wtime()
        header = (
            {
                "height": height,
                "previous": previous,
                "timestamp_ns": time.time_ns(),
                "transaction": {"kind": "MPI_BENCHMARK", "index": transaction},
            }
            if rank == 0
            else None
        )
        header = comm.bcast(header, root=0)
        fault = rank >= nodes - int(nodes * args.fault_rate)
        message = (
            dict(header, transaction={"conflict": True}) if fault and args.fault == "byzantine" else header
        )
        vote = (
            None
            if fault and args.fault == "crash"
            else {"node": rank, "signature": key.sign(proposal_bytes(message, genesis)).hex()}
        )
        votes = comm.allgather(vote)
        valid = []
        for item in votes:
            if item is not None:
                try:
                    certificate_weight(manifest, header, [item])
                    valid.append(item)
                except ValueError:
                    pass
        weight = certificate_weight(manifest, header, valid)
        block = None
        if rank == 0 and reaches_quorum(manifest, weight):
            block = {"header": header, "signatures": valid}
            block["hash"] = digest(b"SHIELD_BLOCK_V1\0" + canonical(block))
        block = comm.bcast(block, root=0)
        if block is not None:
            validate_block(manifest, block, height, previous)
            with (output / "blocks.jsonl").open("ab") as stream:
                stream.write(canonical(block) + b"\n")
            previous, height = block["hash"], height + 1
        durations.append((MPI.Wtime() - started) * 1000)
    summary = {
        "mode": "real_mpi_collectives_full_proof_replication",
        "ranks": nodes,
        "blocks": height,
        "manifest_hash": genesis,
        "fault": args.fault,
        "mean_commit_ms": sum(durations) / len(durations),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    if rank == 0:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
