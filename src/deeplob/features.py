"""Hand-made order book features for the baselines.

All features use only snapshots up to and including the prediction time t:
  - past mid-price returns over 1, 2, 3, 5, 10, 20, 50, 99 events (bps)
  - spread, in bps of the mid
  - volume imbalance (bid - ask) / (bid + ask), cumulative over the top 1, 3, 5, 10 levels
  - the 40 raw book numbers at time t
"""
from __future__ import annotations

import numpy as np

LAGS = (1, 2, 3, 5, 10, 20, 50, 99)


def book_features(lob: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """lob (N, 40) in FI-2010 order; returns features for rows `ends`."""
    ask_p, ask_v, bid_p, bid_v = (lob[:, j::4].astype(np.float64) for j in range(4))
    mid = (ask_p[:, 0] + bid_p[:, 0]) / 2
    lm = np.log(mid)
    t = ends
    cols, names = [], []
    for g in LAGS:
        cols.append((lm[t] - lm[t - g]) * 1e4)
        names.append(f"ret_{g}")
    cols.append((ask_p[t, 0] - bid_p[t, 0]) / mid[t] * 1e4)
    names.append("spread_bps")
    for n in (1, 3, 5, 10):
        b, a = bid_v[t, :n].sum(1), ask_v[t, :n].sum(1)
        cols.append((b - a) / (b + a + 1e-12))
        names.append(f"imb_{n}")
    X = np.column_stack(cols + [lob[t].astype(np.float64)])
    names += [f"{s}{i + 1}" for i in range(10) for s in ("ask_p", "ask_v", "bid_p", "bid_v")]
    return X, names
