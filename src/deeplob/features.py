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


BINANCE_LAGS = (1, 2, 5, 10, 20, 50, 99)   # snapshots (100 ms each), all inside the window


def binance_features(book, ends: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Features at prediction times `ends` from a binance.Book (all from data up to t).

    Past mid returns (bps), spread (bps), cumulative volume imbalance over the top 1/3/5/10
    levels, each level's distance from the mid (bps) and log(1 + qty) for levels 1-10.
    """
    lm = np.log(book.mid)
    t = ends
    cols, names = [], []
    for g in BINANCE_LAGS:
        cols.append((lm[t] - lm[t - g]) * 1e4)
        names.append(f"ret_{g}")
    cols.append((book.ask1[t] - book.bid1[t]) / book.mid[t] * 1e4)
    names.append("spread_bps")
    q = np.expm1(book.log_q[t])
    ask_q, bid_q = q[:, :10], q[:, 10:]
    for n in (1, 3, 5, 10):
        b, a = bid_q[:, :n].sum(1), ask_q[:, :n].sum(1)
        cols.append((b - a) / (b + a + 1e-12))
        names.append(f"imb_{n}")
    dist = (book.log_p[t] - lm[t][:, None]) * 1e4
    X = np.column_stack(cols + [dist, book.log_q[t]])
    names += [f"{s}_dist{i + 1}" for s in ("ask", "bid") for i in range(10)]
    names += [f"{s}_logq{i + 1}" for s in ("ask", "bid") for i in range(10)]
    return X, names
