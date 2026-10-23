import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deeplob import fi2010 as F  # noqa: E402


def fake_raw(n_per_stock=(300, 400), levels=(1.0, 3.0)):
    """Two 'stocks' at very different price levels, FI-2010 row layout."""
    cols = []
    rng = np.random.default_rng(0)
    for n, p in zip(n_per_stock, levels):
        raw = np.zeros((149, n), np.float32)
        mid = p * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
        for i in range(10):
            raw[4 * i] = mid + 0.001 * (i + 1)
            raw[4 * i + 2] = mid - 0.001 * (i + 1)
            raw[4 * i + 1] = raw[4 * i + 3] = 1.0
        raw[144:] = rng.integers(1, 4, (5, n))
        cols.append(raw)
    return np.concatenate(cols, axis=1)


def test_segments_split_at_stock_boundary():
    raw = fake_raw()
    assert F.stock_segments(raw) == [(0, 300), (300, 700)]


def test_windows_stay_inside_one_stock_and_end_at_label_row():
    raw = fake_raw()
    s = F.make_split(raw, 10)
    assert len(s.ends) == (300 - 99) + (400 - 99)
    assert not np.any((s.ends >= 300) & (s.ends < 399))   # no window crosses row 300
    w = F.windows(s, np.array([0, len(s.ends) - 1]))
    assert w.shape == (2, 100, 40)
    np.testing.assert_array_equal(w[0, -1], raw[:40, 99])
    np.testing.assert_array_equal(w[1, 0], raw[:40, 600])
    assert s.y[s.ends[0]] == 3 - raw[144, 99]


def test_train_val_split_is_per_stock_with_gap():
    raw = fake_raw((1000, 2000))
    tr, va = F.train_val_segments(raw, val_frac=0.2, gap=100)
    assert tr == [(0, 700), (1000, 2500)] and va == [(800, 1000), (2600, 3000)]
