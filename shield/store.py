"""Local content-addressed templates checked against authenticated enrollment events."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .crypto import array_hash, canonical
from .ledger import verify_ledger


class VectorStore:
    def __init__(self, directory, ledger):
        self.directory, self.ledger = Path(directory), ledger
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / "index.json"
        self.index = json.loads(self.index_path.read_text()) if self.index_path.exists() else {}

    def enroll(self, patient_id, shard_embeddings, model_hash):
        vectors = np.asarray(shard_embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] < 1 or not np.isfinite(vectors).all():
            raise ValueError("Expected a finite nonempty matrix of shard embeddings")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1, atol=1e-4):
            raise ValueError("Shard ISVs must be L2 normalized")
        # Eq. 3 stores the arithmetic mean, without changing its hash by normalization.
        template = vectors.mean(axis=0)
        if np.linalg.norm(template) < 1e-8:
            raise ValueError("Degenerate mean template")
        address = array_hash(template)
        np.save(self.directory / f"{address}.npy", template, allow_pickle=False)
        proof = self.ledger.commit(
            {
                "kind": "ENROLL",
                "patient_id": str(patient_id),
                "template_hash": address,
                "model_hash": model_hash,
                "embedding_dim": int(template.size),
            }
        )
        self.index[str(patient_id)] = {"hash": address, "height": proof["height"], "model_hash": model_hash}
        temporary = self.index_path.with_suffix(".tmp")
        temporary.write_bytes(canonical(self.index))
        temporary.replace(self.index_path)
        return proof

    def _verified_templates(self, model_hash):
        verify_ledger(self.ledger.directory, self.ledger.genesis)
        # Derive authoritative patient membership and model binding from the ledger.
        authoritative = {}
        for block in self.ledger.blocks:
            tx = block["header"]["transaction"]
            if tx.get("kind") == "ENROLL":
                authoritative[tx["patient_id"]] = {
                    "hash": tx["template_hash"],
                    "height": block["header"]["height"],
                    "model_hash": tx["model_hash"],
                }
        if self.index != authoritative:
            raise ValueError("Template index differs from ledger enrollments")
        result = {}
        for patient, item in self.index.items():
            if item["model_hash"] != model_hash:
                continue
            value = np.load(self.directory / f"{item['hash']}.npy", allow_pickle=False)
            if value.ndim != 1 or array_hash(value) != item["hash"] or np.linalg.norm(value) < 1e-8:
                raise ValueError("Template content-address verification failed")
            result[patient] = value / np.linalg.norm(value)
        return result

    def query(self, embedding, model_hash, top_k=5, anchor=True):
        query = np.asarray(embedding, dtype=np.float32)
        if query.ndim != 1 or not np.isfinite(query).all() or np.linalg.norm(query) < 1e-8 or top_k < 1:
            raise ValueError("Expected a finite nonzero query vector and positive top_k")
        templates = self._verified_templates(model_hash)
        if not templates:
            raise ValueError("No enrolled templates for this model checkpoint")
        if any(value.shape != query.shape for value in templates.values()):
            raise ValueError("Query dimension does not match templates")
        normalized = query / np.linalg.norm(query)
        ranking = sorted(
            (
                {"patient_id": patient, "score": float(normalized @ value)}
                for patient, value in templates.items()
            ),
            key=lambda item: (-item["score"], item["patient_id"]),
        )[:top_k]
        proof = (
            self.ledger.commit(
                {
                    "kind": "VERIFY",
                    "query_hash": array_hash(query),
                    "model_hash": model_hash,
                    "ranking": ranking,
                }
            )
            if anchor
            else None
        )
        return {"ranking": ranking, "proof": proof}
