import copy
import json

import numpy as np
import pytest

from shield.ledger import ProvenanceLedger, QuorumError, validate_block, verify_ledger
from shield.overlay import Overlay
from shield.store import VectorStore


def ledger(tmp_path):
    return ProvenanceLedger(tmp_path / "ledger", Overlay(8), 1)


def test_chain_persistence_and_signature_tampering(tmp_path):
    chain = ledger(tmp_path)
    proof = chain.commit({"kind": "ANCHOR", "model_hash": "a" * 64})
    assert verify_ledger(chain.directory, chain.genesis)["blocks"] == 1
    reloaded = ledger(tmp_path)
    reloaded.commit({"kind": "ANCHOR", "model_hash": "b" * 64})
    assert verify_ledger(chain.directory, chain.genesis)["blocks"] == 2
    forged = copy.deepcopy(chain.blocks[0])
    forged["header"]["transaction"]["model_hash"] = "c" * 64
    forged = chain.assemble(forged["header"], forged["signatures"])
    with pytest.raises(ValueError, match="signature"):
        validate_block(chain.manifest, forged, 0, chain.genesis)
    with pytest.raises(ValueError, match="Pinned"):
        verify_ledger(chain.directory, "0" * 64)
    assert proof["manifest_hash"] == chain.genesis


def test_quorum_duplicates_crashes_conflicts_and_equivocation(tmp_path):
    chain = ledger(tmp_path)
    with pytest.raises(QuorumError):
        chain.commit({"kind": "X"}, crashed=range(8))
    proof = chain.commit({"kind": "X"}, byzantine=[7])
    assert proof["signatures"] == 7
    header = chain.prepare({"kind": "Y"})
    vote = chain.sign(0, header)
    with pytest.raises(ValueError, match="duplicated"):
        chain.append(chain.assemble(header, [vote, vote]))
    with pytest.raises(ValueError, match="conflicting"):
        chain.sign(0, dict(header, transaction={"kind": "Z"}))


def test_template_authenticity_and_index_binding(tmp_path):
    chain = ledger(tmp_path)
    store = VectorStore(tmp_path / "vectors", chain)
    vectors = np.eye(256, dtype=np.float32)
    store.enroll("p1", vectors[:1], "checkpoint")
    store.enroll("p2", vectors[1:2], "checkpoint")
    result = store.query(vectors[0], "checkpoint")
    assert result["ranking"][0]["patient_id"] == "p1"
    assert result["ranking"][0]["score"] == pytest.approx(1)
    item = store.index["p1"]
    np.save(store.directory / f"{item['hash']}.npy", vectors[1], allow_pickle=False)
    with pytest.raises(ValueError, match="verification"):
        store.query(vectors[0], "checkpoint")
    store.index.pop("p1")
    with pytest.raises(ValueError, match="index"):
        store.query(vectors[0], "checkpoint")


def test_raw_templates_not_in_ledger(tmp_path):
    chain = ledger(tmp_path)
    store = VectorStore(tmp_path / "vectors", chain)
    store.enroll("p", np.eye(256, dtype=np.float32)[:1], "model")
    tx = json.loads((chain.directory / "blocks.jsonl").read_text())["header"]["transaction"]
    assert set(tx) == {"kind", "patient_id", "template_hash", "model_hash", "embedding_dim"}


def test_paper_quorum_rule_allows_disjoint_certificates(tmp_path):
    """A regression documenting the paper rule's non-property, not a safety claim."""
    chain = ledger(tmp_path)
    first = chain.prepare({"choice": "A"})
    second = dict(first, transaction={"choice": "B"})
    a = chain.assemble(first, [chain.sign(i, first) for i in [0, 1, 2]])
    b = chain.assemble(second, [chain.sign(i, second) for i in [3, 4, 5, 6, 7]])
    # Both meet Equation 4 without any validator equivocating.
    assert validate_block(chain.manifest, a, 0, chain.genesis) == 15
    assert validate_block(chain.manifest, b, 0, chain.genesis) == 21


def test_retained_head_detects_truncation(tmp_path):
    chain = ledger(tmp_path)
    chain.commit({"kind": "A"})
    chain.commit({"kind": "B"})
    head = verify_ledger(chain.directory, chain.genesis)["head"]
    path = chain.directory / "blocks.jsonl"
    lines = path.read_text().splitlines()
    path.write_text(lines[0] + "\n")
    with pytest.raises(ValueError, match="truncation"):
        verify_ledger(chain.directory, chain.genesis, head)
