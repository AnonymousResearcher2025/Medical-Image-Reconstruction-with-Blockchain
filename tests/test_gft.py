import itertools

import pytest
import torch

from shield.gft import GFT, hilbert_xy


@pytest.mark.parametrize("height,width,tile", [(256, 256, 32), (39, 65, 16), (9, 7, 32)])
def test_exact_reassembly(height, width, tile):
    image = torch.rand(1, height, width)
    layout = GFT(height, width, tile, 7)
    assert torch.equal(image, layout.reconstruct(layout.fragment(image)))
    assert all(p.node == p.hilbert_index % 7 for p in layout.tiles)


def test_hilbert_traversal_and_counterexample():
    cells = [hilbert_xy(16, i) for i in range(256)]
    assert len(set(cells)) == 256
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in itertools.pairwise(cells))
    index = {cell: i for i, cell in enumerate(cells)}
    maximum = max(abs(index[x, y] - index[x + 1, y]) for x in range(15) for y in range(16))
    assert maximum > 100  # adjacent grid cells need not have nearby Hilbert indices


def test_invalid_parameters():
    with pytest.raises(ValueError):
        GFT(0, 32)
    with pytest.raises(ValueError):
        hilbert_xy(3, 0)
