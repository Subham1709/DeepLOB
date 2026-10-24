"""Turn recorded Binance chunks into DeepLOB inputs and future-only labels.

Snapshots arrive every 100 ms. A *session* is a stretch with no gap longer than
`max_gap_ms` (runner restarts, dropped connections); windows and labels never cross one.

Inputs, built per window so nothing is computed from data after the prediction time:
  prices   log(p / mid_t), where t is the window's last snapshot, divided by a per-column
           scale fitted on training windows (BTC's spread is ~0.001 bp, so raw bps would be tiny
           numbers). Prices then don't drift with the BTC level, but the model still sees
           how the book moved within the window.
  volumes  log(1 + qty), z-scored with the training days' mean and std.
Column order matches FI-2010: ask price, ask volume, bid price, bid volume for levels 1-10.

Label at t for horizon h (snapshots): r = mean(mid[t+1 .. t+h]) / mid[t] - 1, using only
the future (unlike FI-2010, see the learning log). Classes: down if r < -alpha, up if
r > alpha, else flat; alpha is set on the training days so the three classes are about
equally common. Around 40% of 1-second horizons have no mid move at all, so alpha has a
floor (`MIN_ALPHA`): "flat" then means "exactly unchanged".
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .recorder import LEVELS, load_chunk

MIN_ALPHA = 1e-8   # 0.0001 bp: below float noise in the mean of mids, far below a tick


@dataclass
class Book:
    t: np.ndarray        # ms since epoch
    log_p: np.ndarray    # (n, 20): log ask prices L1-10, log bid prices L1-10
    log_q: np.ndarray    # (n, 20): log(1+qty), same order
    mid: np.ndarray
    ask1: np.ndarray
    bid1: np.ndarray


def load_symbol(root: Path, symbol: str) -> Book:
    parts = [load_chunk(p) for p in sorted((root / symbol).glob("*.npz"))]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.argsort(cat["t"], kind="stable")
    cat = {k: v[order] for k, v in cat.items()}
    ok = (cat["ask_p"][:, 0] > cat["bid_p"][:, 0]) & (cat["bid_p"][:, -1] > 0)  # drop broken rows
    cat = {k: v[ok] for k, v in cat.items()}
    mid = (cat["ask_p"][:, 0] + cat["bid_p"][:, 0]) / 2
    return Book(t=cat["t"], log_p=np.log(np.hstack([cat["ask_p"], cat["bid_p"]])),
                log_q=np.log1p(np.hstack([cat["ask_q"], cat["bid_q"]])),
                mid=mid, ask1=cat["ask_p"][:, 0], bid1=cat["bid_p"][:, 0])


def sessions(t: np.ndarray, max_gap_ms: int = 1000) -> list[tuple[int, int]]:
    cuts = np.where(np.diff(t) > max_gap_ms)[0] + 1
    e = np.r_[0, cuts, len(t)]
    return list(zip(e[:-1].tolist(), e[1:].tolist()))


def utc_day(t_ms: np.ndarray) -> np.ndarray:
    return np.array([datetime.fromtimestamp(x / 1000, timezone.utc).strftime("%Y-%m-%d") for x in t_ms])


def future_return(mid: np.ndarray, h: int) -> np.ndarray:
    """r[t] = mean(mid[t+1..t+h]) / mid[t] - 1 (NaN where the future runs off the end)."""
    c = np.cumsum(np.r_[0.0, mid])
    r = np.full(len(mid), np.nan)
    t = np.arange(len(mid) - h)
    r[t] = (c[t + 1 + h] - c[t + 1]) / h / mid[t] - 1
    return r


def window_ends(book: Book, h: int, T: int = 100, days: set[str] | None = None,
                stride: int = 1) -> np.ndarray:
    """Valid prediction times: a full T-window behind and h snapshots ahead, inside one
    session, and (optionally) on the given UTC days."""
    day = utc_day(book.t) if days is not None else None
    ends = []
    for s, e in sessions(book.t):
        idx = np.arange(s + T - 1, e - h, stride)
        if day is not None:
            idx = idx[np.isin(day[idx], list(days))]
        ends.append(idx)
    return np.concatenate(ends) if ends else np.zeros(0, np.int64)


def labels(r: np.ndarray, alpha: float) -> np.ndarray:
    y = np.ones(len(r), np.int64)
    y[r > alpha] = 2
    y[r < -alpha] = 0
    return y


def balanced_alpha(r_train: np.ndarray) -> float:
    """alpha with about a third of training labels flat (|r| <= alpha)."""
    return max(float(np.quantile(np.abs(r_train[~np.isnan(r_train)]), 1 / 3)), MIN_ALPHA)


def fit_scales(book: Book, ends: np.ndarray, T: int = 100, n: int = 5000, seed: int = 0):
    """(price scales (20,), volume mean (20,), volume std (20,)) from a sample of training windows."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(ends, size=min(n, len(ends)), replace=False)
    offs = book.log_p[pick[:, None] - np.arange(T)[None, :]] - np.log(book.mid[pick])[:, None, None]
    return offs.reshape(-1, offs.shape[-1]).std(0) + 1e-12, book.log_q[ends].mean(0), book.log_q[ends].std(0) + 1e-8


class BinanceWindows(torch.utils.data.Dataset):
    """Same interface as train.Windows: (window (T, 40) float32, label)."""

    def __init__(self, book: Book, y: np.ndarray, ends: np.ndarray, p_scale: np.ndarray,
                 q_mean: np.ndarray, q_std: np.ndarray, T: int = 100):
        self.lp = torch.from_numpy(book.log_p.astype(np.float64))
        self.p_scale = torch.from_numpy(np.asarray(p_scale, np.float64))
        self.lmid = torch.from_numpy(np.log(book.mid))
        self.q = torch.from_numpy(((book.log_q - q_mean) / q_std).astype(np.float32))
        self.y = torch.from_numpy(y)
        self.ends = ends
        self.T = T
        L = LEVELS
        # FI-2010 column order: for level i -> ask p, ask q, bid p, bid q
        self.p_cols = torch.tensor([j for i in range(L) for j in (i, L + i)])

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, i):
        t = int(self.ends[i])
        sl = slice(t - self.T + 1, t + 1)
        p = ((self.lp[sl] - self.lmid[t]) / self.p_scale).float()   # (T, 20) relative to mid_t
        q = self.q[sl]
        L = LEVELS
        x = torch.empty(self.T, 4 * L)
        x[:, 0::4], x[:, 1::4] = p[:, :L], q[:, :L]
        x[:, 2::4], x[:, 3::4] = p[:, L:], q[:, L:]
        return x, self.y[t]


SYMBOLS = ("BTCUSDT", "ETHUSDT")


@dataclass
class Prepared:
    """One symbol at one horizon: the book, labels and window ends for train / val / test."""
    symbol: str
    h: int
    book: Book
    y: np.ndarray
    r: np.ndarray
    alpha: float
    ends: dict
    scales: tuple

    def windows(self, split: str, stride: int = 1, T: int = 100) -> BinanceWindows:
        return BinanceWindows(self.book, self.y, self.ends[split][::stride], *self.scales, T=T)


def time_split(book: Book, h: int, T: int = 100, fractions=(0.6, 0.2, 0.2)) -> dict:
    """Chronological train / val / test split of the prediction times.

    Cut points are set on recording time. Around each cut, `T + h` snapshots are dropped
    (an embargo), so that no training label looks into the validation period and no
    validation window starts inside the training period.
    """
    ends = window_ends(book, h, T)
    t = book.t[ends]
    c1, c2 = np.quantile(t, [fractions[0], fractions[0] + fractions[1]])
    gap_ms = (T + h) * 100
    return {"train": ends[t < c1 - gap_ms],
            "val": ends[(t >= c1) & (t < c2 - gap_ms)],
            "test": ends[t >= c2]}


def prepare(root: Path, symbol: str, h: int, T: int = 100) -> Prepared:
    book = load_symbol(root, symbol)
    ends = time_split(book, h, T)
    r = future_return(book.mid, h)
    alpha = balanced_alpha(r[ends["train"]])
    y = labels(np.nan_to_num(r), alpha)
    return Prepared(symbol, h, book, y, r, alpha, ends, fit_scales(book, ends["train"], T))
