"""Equation 1: Hilbert index modulo node count, with exact padded reassembly.

A Hilbert traversal has adjacent consecutive cells. The converse and graph
locality guarantees claimed in the paper do not follow; see docs/PAPER_NOTES.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


def hilbert_xy(side: int, distance: int) -> tuple[int, int]:
    if side < 1 or side & (side - 1) or not 0 <= distance < side * side:
        raise ValueError("Hilbert side must be a power of two and distance must lie in the square")
    x = y = 0
    scale, remaining = 1, distance
    while scale < side:
        rx, ry = (remaining // 2) & 1, (remaining ^ (remaining // 2)) & 1
        if ry == 0:
            if rx == 1:
                x, y = scale - 1 - x, scale - 1 - y
            x, y = y, x
        x += scale * rx
        y += scale * ry
        remaining //= 4
        scale *= 2
    return x, y


@dataclass(frozen=True)
class Tile:
    row: int
    col: int
    hilbert_index: int
    node: int


class GFT:
    def __init__(
        self,
        height: int,
        width: int,
        tile_size: int = 32,
        nodes: int = 20,
        placement: str = "hilbert",
        seed: int = 0,
    ):
        if min(height, width, tile_size, nodes) < 1:
            raise ValueError("Image dimensions, tile size, and nodes must be positive")
        if placement not in {"hilbert", "random"}:
            raise ValueError("Unknown placement")
        self.height, self.width, self.tile_size, self.nodes = height, width, tile_size, nodes
        self.rows, self.cols = (height + tile_size - 1) // tile_size, (width + tile_size - 1) // tile_size
        side = 1 << (max(self.rows, self.cols) - 1).bit_length()
        self.tiles = []
        rng = np.random.default_rng(seed)
        for distance in range(side * side):
            x, y = hilbert_xy(side, distance)
            if y < self.rows and x < self.cols:
                node = distance % nodes if placement == "hilbert" else int(rng.integers(nodes))
                self.tiles.append(Tile(y, x, distance, node))

    def fragment(self, image: torch.Tensor) -> torch.Tensor:
        """CHW -> KCHW, ordered by Hilbert traversal."""
        if image.ndim != 3 or tuple(image.shape[-2:]) != (self.height, self.width):
            raise ValueError("Expected CHW matching this GFT layout")
        t = self.tile_size
        padded = F.pad(
            image, (0, self.cols * t - self.width, 0, self.rows * t - self.height), mode="replicate"
        )
        return torch.stack(
            [padded[:, p.row * t : (p.row + 1) * t, p.col * t : (p.col + 1) * t] for p in self.tiles]
        )

    def reconstruct(self, shards: torch.Tensor) -> torch.Tensor:
        t = self.tile_size
        if shards.ndim != 4 or shards.shape[0] != len(self.tiles) or tuple(shards.shape[-2:]) != (t, t):
            raise ValueError("Shard shape does not match layout")
        image = shards.new_zeros(shards.shape[1], self.rows * t, self.cols * t)
        for shard, p in zip(shards, self.tiles):
            image[:, p.row * t : (p.row + 1) * t, p.col * t : (p.col + 1) * t] = shard
        return image[:, : self.height, : self.width]
