import numpy as np
import pytest
import torch

from shield.crypto import state_hash
from shield.federated import fedavg, partition_clients
from shield.losses import composite_loss, ssim
from shield.metrics import binary_auc
from shield.model import HybridUNet


def test_model_loss_backward_and_checkpoint_hash(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = HybridUNet(2, 8)
    clean = torch.rand(4, 1, 32, 32)
    noisy = (clean + torch.randn_like(clean) * 0.05).clamp(0, 1)
    before = state_hash(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss, parts = composite_loss(model, noisy, clean, torch.tensor([0, 0, 1, 1]))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert parts["triplet"] >= 0
    optimizer.step()
    assert state_hash(model.state_dict()) != before
    model.eval()
    output, v = model(clean[:1])
    assert output.shape == clean[:1].shape and v.shape == (1, 256)
    assert torch.allclose(v.norm(dim=1), torch.ones(1), atol=1e-6)
    assert torch.allclose(ssim(clean, clean), torch.ones(4), atol=1e-5)


def test_fedavg_and_partition_coverage():
    states = [
        {"w": torch.tensor([1.0, 2.0]), "counter": torch.tensor(0)},
        {"w": torch.tensor([3.0, 6.0]), "counter": torch.tensor(0)},
    ]
    assert torch.equal(fedavg(states)["w"], torch.tensor([2.0, 4.0]))
    assert torch.equal(fedavg(states, [1, 3])["w"], torch.tensor([2.5, 5.0]))
    labels = [0] * 12 + [1] * 12 + [2] * 12
    for mode in ["iid", "dirichlet"]:
        partitions = partition_clients(labels, 5, mode, seed=1)
        assert all(partitions)
        assert sorted(i for part in partitions for i in part) == list(range(36))
        assert partitions == partition_clients(labels, 5, mode, seed=1)
    with pytest.raises(ValueError):
        fedavg(states, [0, 1])


def test_auc_ties_and_perfect_ranking():
    assert binary_auc([0, 1], [0.2, 0.8]) == 1
    assert binary_auc([0, 1], [0.5, 0.5]) == 0.5
    assert binary_auc([1], [0.5]) is None
    assert np.isfinite(binary_auc([0, 1, 0, 1], [0.4, 0.8, 0.3, 0.7]))
