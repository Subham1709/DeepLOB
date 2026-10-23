import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deeplob.backtest import backtest  # noqa: E402


def test_perfect_foresight_pays_only_costs():
    n, h = 1000, 10
    mid = 100 * np.exp(np.cumsum(np.r_[0, np.random.default_rng(0).normal(0, 1e-4, n - 1)]))
    bid, ask = mid - 0.005, mid + 0.005
    ends = np.arange(0, n - h)
    up = mid[ends + h] > mid[ends]
    prob = np.where(up[:, None], [0, 0, 1], [1, 0, 0]).astype(float)
    free = backtest(mid, mid, mid, ends, prob, h, fee_bps=0)
    assert free["hit_rate"] == 1.0 and free["net_bps_total"] > 0
    assert np.isclose(free["net_bps_total"], free["gross_bps_total"])
    costly = backtest(mid, bid, ask, ends, prob, h, fee_bps=10)
    assert costly["net_bps_total"] < free["net_bps_total"]
    assert costly["decisions"] == len(range(0, n - h, h))


def test_flat_signal_costs_nothing():
    mid = np.linspace(100, 101, 200)
    prob = np.tile([0.2, 0.6, 0.2], (190, 1))
    r = backtest(mid, mid - 0.01, mid + 0.01, np.arange(190), prob, 10, fee_bps=10)
    assert r["active"] == 0 and r["net_bps_total"] == 0
