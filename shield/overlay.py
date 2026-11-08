"""Seeded trust-weighted Barabasi-Albert attachment (m=3 by default)."""

from __future__ import annotations

import numpy as np


class Overlay:
    def __init__(self, nodes: int = 20, m: int = 3, seed: int = 0, trust=None):
        if nodes < 2 or not 1 <= m < nodes:
            raise ValueError("Require nodes >= 2 and 1 <= m < nodes")
        self.nodes, self.m, self.seed = nodes, m, seed
        self.trust = np.ones(nodes) if trust is None else np.asarray(trust, dtype=float)
        if self.trust.shape != (nodes,) or not np.isfinite(self.trust).all() or (self.trust <= 0).any():
            raise ValueError("Trust must have one finite positive value per node")
        self.adjacency = [set() for _ in range(nodes)]
        for i in range(m + 1):
            for j in range(i):
                self._edge(i, j)
        rng = np.random.default_rng(seed)
        for new in range(m + 1, nodes):
            weights = np.array([len(self.adjacency[i]) for i in range(new)]) * self.trust[:new]
            for target in rng.choice(new, size=m, replace=False, p=weights / weights.sum()):
                self._edge(new, int(target))

    def _edge(self, a, b):
        self.adjacency[a].add(b)
        self.adjacency[b].add(a)

    @property
    def degrees(self) -> list[int]:
        return [len(n) for n in self.adjacency]

    def config(self) -> dict:
        return {"nodes": self.nodes, "m": self.m, "seed": self.seed, "trust": self.trust.tolist()}

    def metadata(self) -> dict:
        return {"config": self.config(), "degrees": self.degrees}
