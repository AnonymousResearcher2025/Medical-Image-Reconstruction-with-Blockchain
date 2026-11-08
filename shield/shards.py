"""Content-addressed off-ledger shard placement, emulating one directory per node."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .crypto import array_hash, canonical, digest, merkle_root
from .gft import GFT


class ShardStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def put(self, shards, layout):
        if len(shards) != len(layout.tiles):
            raise ValueError("Shard count differs from layout")
        addresses = []
        for shard, tile in zip(shards, layout.tiles):
            array = shard.detach().cpu().numpy()
            address = array_hash(array)
            directory = self.directory / f"node-{tile.node}"
            directory.mkdir(exist_ok=True)
            np.save(directory / f"{address}.npy", array, allow_pickle=False)
            addresses.append({"hash": address, "node": tile.node, "row": tile.row, "col": tile.col})
        manifest = {
            "height": layout.height,
            "width": layout.width,
            "tile_size": layout.tile_size,
            "nodes": layout.nodes,
            "shards": addresses,
            "merkle_root": merkle_root([item["hash"] for item in addresses]),
        }
        address = digest(canonical(manifest))
        (self.directory / f"{address}.json").write_bytes(canonical(manifest))
        return {"manifest_hash": address, "merkle_root": manifest["merkle_root"]}

    def get(self, manifest_address, expected_merkle_root):
        raw = (self.directory / f"{manifest_address}.json").read_bytes()
        if digest(raw) != manifest_address:
            raise ValueError("Shard manifest hash mismatch")
        manifest = json.loads(raw)
        layout = GFT(manifest["height"], manifest["width"], manifest["tile_size"], manifest["nodes"])
        by_coordinate = {}
        hashes = []
        for item in manifest["shards"]:
            if type(item["node"]) is not int or not 0 <= item["node"] < layout.nodes:
                raise ValueError("Invalid shard node")
            # A content address is a fixed-length hex digest, never a filesystem path.
            if len(item["hash"]) != 64 or any(c not in "0123456789abcdef" for c in item["hash"]):
                raise ValueError("Invalid shard content address")
            array = np.load(
                self.directory / f"node-{item['node']}" / f"{item['hash']}.npy", allow_pickle=False
            )
            if array_hash(array) != item["hash"]:
                raise ValueError("Shard content hash mismatch")
            coordinate = (item["row"], item["col"])
            if coordinate in by_coordinate:
                raise ValueError("Duplicated shard coordinate")
            by_coordinate[coordinate] = torch.from_numpy(array)
            hashes.append(item["hash"])
        if merkle_root(hashes) != expected_merkle_root or manifest["merkle_root"] != expected_merkle_root:
            raise ValueError("Shard Merkle root differs from ledger commitment")
        if set(by_coordinate) != {(tile.row, tile.col) for tile in layout.tiles}:
            raise ValueError("Shard coordinates do not match image")
        return layout.reconstruct(torch.stack([by_coordinate[tile.row, tile.col] for tile in layout.tiles]))
