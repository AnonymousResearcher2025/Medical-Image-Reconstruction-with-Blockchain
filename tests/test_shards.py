import json

import numpy as np
import pytest
import torch

from shield.gft import GFT
from shield.shards import ShardStore


def test_sharded_roundtrip_and_tampering(tmp_path):
    image = torch.rand(1, 39, 65)
    layout = GFT(39, 65, 16, 5)
    store = ShardStore(tmp_path)
    proof = store.put(layout.fragment(image), layout)
    assert torch.equal(store.get(proof["manifest_hash"], proof["merkle_root"]), image)
    manifest = json.loads((tmp_path / f"{proof['manifest_hash']}.json").read_text())
    item = manifest["shards"][0]
    np.save(tmp_path / f"node-{item['node']}" / f"{item['hash']}.npy", np.zeros((1, 16, 16), np.float32))
    with pytest.raises(ValueError, match="content hash"):
        store.get(proof["manifest_hash"], proof["merkle_root"])
