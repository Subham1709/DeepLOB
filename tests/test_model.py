import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deeplob.model import DeepLOB  # noqa: E402


def test_shapes_and_size():
    m = DeepLOB()
    assert m(torch.randn(8, 100, 40)).shape == (8, 3)
    assert m(torch.randn(8, 1, 100, 40)).shape == (8, 3)
    assert sum(p.numel() for p in m.parameters()) == 143_907


def test_can_overfit_a_tiny_batch():
    torch.manual_seed(0)
    m = DeepLOB()
    x, y = torch.randn(16, 100, 40), torch.randint(0, 3, (16,))
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(60):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
