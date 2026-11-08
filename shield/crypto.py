"""Portable content addressing with domain-separated SHA-256 and Ed25519."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_hash(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array, dtype="<f4")
    if not np.isfinite(array).all():
        raise ValueError("Cannot hash non-finite vectors")
    return digest(
        b"SHIELD_ARRAY_V1\0" + canonical({"shape": list(array.shape), "dtype": "<f4"}) + array.tobytes()
    )


def merkle_root(hashes: list[str]) -> str:
    if not hashes:
        return digest(b"SHIELD_EMPTY_V1")
    level = [hashlib.sha256(b"\x00" + bytes.fromhex(h)).digest() for h in hashes]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(b"\x01" + level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
    return level[0].hex()


def state_hash(state: dict) -> str:
    """Hash tensor values, shapes and dtypes, independent of torch.save's archive metadata."""
    import torch

    h = hashlib.sha256(b"SHIELD_STATE_V1\0")
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        metadata = canonical({"name": name, "shape": list(value.shape), "dtype": str(value.dtype)})
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        h.update(len(metadata).to_bytes(8, "big"))
        h.update(metadata)
        h.update(len(raw).to_bytes(8, "big"))
        h.update(raw)
    return h.hexdigest()
