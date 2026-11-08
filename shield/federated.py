"""Deterministic IID/Dirichlet partitions and Eq. 2 Algorithm 2 aggregation."""

from __future__ import annotations

import numpy as np
import torch


def partition_clients(labels, clients=5, mode="iid", alpha=0.5, seed=0):
    if clients < 1 or len(labels) < clients or alpha <= 0 or mode not in {"iid", "dirichlet"}:
        raise ValueError("Invalid federated partition parameters or insufficient samples")
    rng = np.random.default_rng(seed)
    if mode == "iid":
        return [part.tolist() for part in np.array_split(rng.permutation(len(labels)), clients)]
    result = [[] for _ in range(clients)]
    labels = np.array(labels)
    for label in sorted(set(labels.tolist())):
        indices = np.where(labels == label)[0]
        rng.shuffle(indices)
        counts = rng.multinomial(len(indices), rng.dirichlet(np.full(clients, alpha)))
        for target, part in zip(result, np.split(indices, np.cumsum(counts)[:-1])):
            target.extend(part.tolist())
    # Explicit deterministic repair guarantees participation without dropping/duplicating data.
    for target in result:
        if not target:
            largest = max(result, key=len)
            target.append(largest.pop())
    return result


def fedavg(states, counts=None):
    if not states:
        raise ValueError("No client states")
    counts = np.ones(len(states)) if counts is None else np.asarray(counts, dtype=float)
    if counts.shape != (len(states),) or not np.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("Positive finite aggregation counts required")
    weights = counts / counts.sum()
    result = {}
    if any(state.keys() != states[0].keys() for state in states):
        raise ValueError("Client state keys differ")
    for name, first in states[0].items():
        if any(state[name].shape != first.shape or state[name].dtype != first.dtype for state in states):
            raise ValueError("Client tensor metadata differs")
        if first.is_floating_point():
            result[name] = sum(
                state[name].to(torch.float64) * float(weight) for state, weight in zip(states, weights)
            ).to(first.dtype)
        else:
            if any(not torch.equal(state[name], first) for state in states):
                raise ValueError("Cannot average different non-floating buffers")
            result[name] = first.clone()
    return result
