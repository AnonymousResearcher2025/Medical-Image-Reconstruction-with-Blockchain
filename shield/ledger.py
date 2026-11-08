"""Hash-linked, Ed25519-signed provenance with the paper's weighted threshold.

The local backend emulates validators in one process. Certificates establish
signature/chain integrity against a pinned manifest, not Byzantine agreement.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

from .crypto import canonical, digest
from .overlay import Overlay


class QuorumError(ValueError):
    pass


def manifest_hash(manifest):
    return digest(b"SHIELD_MANIFEST_V1\0" + canonical(manifest))


def proposal_bytes(header, genesis):
    return b"SHIELD_PROPOSAL_V1\0" + bytes.fromhex(genesis) + canonical(header)


def certificate_weight(manifest, header, signatures):
    seen, weight = set(), 0
    message = proposal_bytes(header, manifest_hash(manifest))
    for item in signatures:
        signer = item["node"]
        if type(signer) is not int or not 0 <= signer < len(manifest["public_keys"]) or signer in seen:
            raise ValueError("Invalid or duplicated signer")
        seen.add(signer)
        try:
            key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(manifest["public_keys"][signer]))
            key.verify(bytes.fromhex(item["signature"]), message)
        except (InvalidSignature, ValueError) as error:
            raise ValueError("Invalid proposal signature") from error
        weight += manifest["overlay"]["degrees"][signer]
    return weight


def reaches_quorum(manifest, weight):
    degrees = manifest["overlay"]["degrees"]
    return weight * len(degrees) >= (2 * manifest["faults"] + 1) * sum(degrees)


def validate_manifest(manifest):
    overlay = Overlay(**manifest["overlay"]["config"])
    if manifest["overlay"]["degrees"] != overlay.degrees:
        raise ValueError("Overlay degree table does not match seeded topology")
    if not 0 <= manifest["faults"] <= (overlay.nodes - 1) // 3:
        raise ValueError("Invalid fault parameter")
    if len(manifest["public_keys"]) != overlay.nodes or len(set(manifest["public_keys"])) != overlay.nodes:
        raise ValueError("Expected unique public keys for every validator")
    for key in manifest["public_keys"]:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(key))
    if manifest["version"] != 1:
        raise ValueError("Unsupported ledger version")


def validate_block(manifest, block, height, previous):
    header = block["header"]
    if header["height"] != height or header["previous"] != previous:
        raise ValueError("Broken chain height or previous hash")
    if type(header["timestamp_ns"]) is not int or header["timestamp_ns"] < 0:
        raise ValueError("Invalid timestamp")
    if block["hash"] != digest(
        b"SHIELD_BLOCK_V1\0" + canonical({"header": header, "signatures": block["signatures"]})
    ):
        raise ValueError("Block hash mismatch")
    weight = certificate_weight(manifest, header, block["signatures"])
    if not reaches_quorum(manifest, weight):
        raise QuorumError("Insufficient valid degree weight")
    return weight


def verify_ledger(directory, expected_manifest_hash=None, expected_head=None):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    validate_manifest(manifest)
    genesis = manifest_hash(manifest)
    if expected_manifest_hash is not None and genesis != expected_manifest_hash:
        raise ValueError("Pinned manifest hash mismatch")
    previous, count = genesis, 0
    path = directory / "blocks.jsonl"
    if not path.exists():
        raise ValueError("Missing ledger blocks file")
    for line in path.read_text().splitlines():
        block = json.loads(line)
        validate_block(manifest, block, count, previous)
        previous, count = block["hash"], count + 1
    if expected_head is not None and previous != expected_head:
        raise ValueError("Ledger head differs from retained checkpoint (possible truncation)")
    return {
        "valid": True,
        "blocks": count,
        "head": previous,
        "manifest_hash": genesis,
        "manifest_pinned": expected_manifest_hash is not None,
    }


class ProvenanceLedger:
    """Single-writer local research backend; never use its co-located keys in production."""

    def __init__(self, directory, overlay=None, faults=1):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keys_dir = self.directory / ".keys"
        self.keys_dir.mkdir(mode=0o700, exist_ok=True)
        os.chmod(self.keys_dir, 0o700)
        path = self.directory / "manifest.json"
        self.keys = []
        if path.exists():
            self.manifest = json.loads(path.read_text())
            if overlay is not None and self.manifest["overlay"] != overlay.metadata():
                raise ValueError("Ledger topology differs from requested topology")
            if self.manifest["faults"] != faults:
                raise ValueError("Ledger fault parameter differs from requested parameter")
            for i, expected in enumerate(self.manifest["public_keys"]):
                key = Ed25519PrivateKey.from_private_bytes((self.keys_dir / f"{i}.key").read_bytes())
                if key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex() != expected:
                    raise ValueError("Validator key mismatch")
                self.keys.append(key)
        else:
            overlay = overlay or Overlay()
            if not 0 <= faults <= (overlay.nodes - 1) // 3:
                raise ValueError("Require 0 <= f <= (N-1)//3")
            self.keys = [Ed25519PrivateKey.generate() for _ in range(overlay.nodes)]
            self.manifest = {
                "version": 1,
                "overlay": overlay.metadata(),
                "faults": faults,
                "public_keys": [
                    k.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex() for k in self.keys
                ],
            }
            for i, key in enumerate(self.keys):
                raw = key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
                with os.fdopen(
                    os.open(self.keys_dir / f"{i}.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
                ) as stream:
                    stream.write(raw)
            path.write_bytes(canonical(self.manifest))
            (self.directory / "blocks.jsonl").touch(exist_ok=False)
        verify_ledger(self.directory)
        self.blocks = [
            json.loads(line) for line in (self.directory / "blocks.jsonl").read_text().splitlines()
        ]
        self.genesis = manifest_hash(self.manifest)

    def prepare(self, transaction):
        return {
            "height": len(self.blocks),
            "previous": self.blocks[-1]["hash"] if self.blocks else self.genesis,
            "timestamp_ns": time.time_ns(),
            "transaction": transaction,
        }

    def sign(self, node, header):
        previous = self.blocks[-1]["hash"] if self.blocks else self.genesis
        if header["height"] != len(self.blocks) or header["previous"] != previous:
            raise ValueError("Validator refuses a proposal that does not extend its ledger")
        # Persist a per-height lock to reject local equivocation across restarts.
        lock = self.keys_dir / f"vote-{node}-{header['height']}.json"
        fingerprint = digest(proposal_bytes(header, self.genesis))
        if lock.exists() and lock.read_text() != fingerprint:
            raise ValueError("Validator refuses a conflicting proposal at this height")
        lock.write_text(fingerprint)
        return {"node": node, "signature": self.keys[node].sign(proposal_bytes(header, self.genesis)).hex()}

    def assemble(self, header, signatures):
        signatures = sorted(signatures, key=lambda item: item["node"])
        block = {"header": header, "signatures": signatures}
        block["hash"] = digest(b"SHIELD_BLOCK_V1\0" + canonical(block))
        return block

    def append(self, block):
        previous = self.blocks[-1]["hash"] if self.blocks else self.genesis
        validate_block(self.manifest, block, len(self.blocks), previous)
        with (self.directory / "blocks.jsonl").open("ab") as stream:
            stream.write(canonical(block) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.blocks.append(block)
        return {
            "block_hash": block["hash"],
            "height": block["header"]["height"],
            "timestamp_ns": block["header"]["timestamp_ns"],
            "manifest_hash": self.genesis,
            "signatures": len(block["signatures"]),
            "certificate_bytes": len(canonical(block)),
        }

    def commit(self, transaction, crashed=(), byzantine=()):
        header = self.prepare(transaction)
        # Exact digest check models rejection of conflicting messages before honest signatures.
        available = [i for i in range(len(self.keys)) if i not in set(crashed) | set(byzantine)]
        if not reaches_quorum(self.manifest, sum(self.manifest["overlay"]["degrees"][i] for i in available)):
            raise QuorumError("Available validators cannot reach the paper's weighted threshold")
        signatures = [self.sign(i, header) for i in available]
        # Inject signatures on a conflicting payload: invalid for this proposal and excluded.
        for i in byzantine:
            alternate = dict(header, transaction={"conflict": True})
            invalid = {
                "node": i,
                "signature": self.keys[i].sign(proposal_bytes(alternate, self.genesis)).hex(),
            }
            try:
                certificate_weight(self.manifest, header, [invalid])
            except ValueError:
                continue
            raise AssertionError("Conflicting signature unexpectedly verified")
        return self.append(self.assemble(header, signatures))
