"""A deliberately simple taker backtest for direction forecasts.

Every h snapshots (non-overlapping decisions) the strategy looks at the model's class
probabilities and holds +1 (up), -1 (down) or 0 units for the next h snapshots:
  - go long if P(up) is the largest and above `min_prob`, short likewise for P(down)
  - otherwise flat
Position changes are executed at the touch (buy at the ask, sell at the bid), so each unit
traded costs half the spread relative to the mid, plus the exchange fee on notional.
Everything is in basis points of notional.

This ignores queue position, latency and market impact, all of which make reality worse, so
a strategy that loses money here would lose more in practice.
"""
from __future__ import annotations

import numpy as np


def decision_rows(ends: np.ndarray, h: int) -> np.ndarray:
    """Indices of the decision times: ends[0], then the first prediction time >= previous + h."""
    take, last = [], -np.inf
    for i, t in enumerate(ends):
        if t >= last + h:
            take.append(i)
            last = t
    return np.asarray(take, dtype=np.int64)


def backtest(mid: np.ndarray, bid1: np.ndarray, ask1: np.ndarray, ends: np.ndarray,
             prob: np.ndarray, h: int, fee_bps: float, min_prob: float = 0.0) -> dict:
    """`ends` are sorted prediction times with probabilities `prob` (n, 3: down, flat, up).
    Decisions are taken at `decision_rows(ends, h)`."""
    take = decision_rows(ends, h)
    t = ends[take]
    p = prob[take]
    signal = np.where((p[:, 2] >= p.max(1)) & (p[:, 2] > min_prob), 1,
                      np.where((p[:, 0] >= p.max(1)) & (p[:, 0] > min_prob), -1, 0))
    # A gap between decisions (session break) means we were flat in between.
    contiguous = np.r_[False, np.diff(t) == h]
    prev = np.where(contiguous, np.r_[0, signal[:-1]], 0)
    gross = signal * (mid[t + h] / mid[t] - 1) * 1e4
    half_spread = (ask1[t] - bid1[t]) / 2 / mid[t] * 1e4
    traded = np.abs(signal - prev)
    cost = traded * (half_spread + fee_bps)
    # close the position at the end of each contiguous run
    run_end = np.r_[~contiguous[1:], True]
    t_close = t + h
    half_spread_close = (ask1[t_close] - bid1[t_close]) / 2 / mid[t_close] * 1e4
    cost = cost + run_end * np.abs(signal) * (half_spread_close + fee_bps)
    net = gross - cost
    active = signal != 0
    return dict(decisions=int(len(t)), active=int(active.sum()), units_traded=float(traded.sum()
                + (run_end * np.abs(signal)).sum()),
                hit_rate=float(np.mean(gross[active] > 0)) if active.any() else float("nan"),
                gross_bps_per_active=float(gross[active].mean()) if active.any() else 0.0,
                cost_bps_per_active=float(cost[active].mean()) if active.any() else 0.0,
                net_bps_total=float(net.sum()), gross_bps_total=float(gross.sum()),
                breakeven_fee_bps=float((gross.sum() - (traded * half_spread).sum())
                                        / max(traded.sum() + (run_end * np.abs(signal)).sum(), 1)))
