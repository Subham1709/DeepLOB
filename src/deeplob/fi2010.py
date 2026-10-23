"""FI-2010 limit order book benchmark (Ntakaris et al., 2018).

Five Finnish stocks on NASDAQ Nordic, 10 trading days in June 2010, one row per order book
event. We use the version shipped with the DeepLOB authors' code: no auction periods,
"decimal precision" normalisation, setup 2 (train on days 1-7, test on days 8, 9, 10).

The text files are stored *transposed*: 149 rows x N events.
  rows   0-39  the order book: for level i = 0..9, rows 4i..4i+3 are
                ask price, ask volume, bid price, bid volume
  rows  40-143 hand-crafted features from the original paper (unused by DeepLOB)
  rows 144-148 labels for horizons k = 10, 20, 30, 50, 100 events:
                1 = up, 2 = stationary, 3 = down

Each file is the five stocks one after another (stock 1 all days, then stock 2, ...). The
price levels of different stocks differ a lot, so the four boundaries show up as huge
jumps in the mid-price; ``stock_segments`` finds them so that no input window or label
straddles two stocks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

import numpy as np

URL = ("https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-"
       "Networks-for-Limit-Order-Books/master/data/data.zip")
FILES = {"train": "Train_Dst_NoAuction_DecPre_CF_7", "test8": "Test_Dst_NoAuction_DecPre_CF_7",
         "test9": "Test_Dst_NoAuction_DecPre_CF_8", "test10": "Test_Dst_NoAuction_DecPre_CF_9"}
HORIZONS = (10, 20, 30, 50, 100)
N_LOB = 40


def download(root: Path) -> None:
    """Fetch the zip (56 MB) once, unpack, and cache each file as float32 .npy."""
    root.mkdir(parents=True, exist_ok=True)
    if all((root / f"{f}.npy").exists() for f in FILES.values()):
        return
    z = root / "data.zip"
    if not z.exists():
        urlretrieve(URL, z)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(root)
    for f in FILES.values():
        np.save(root / f"{f}.npy", np.loadtxt(root / f"{f}.txt", dtype=np.float32))
        (root / f"{f}.txt").unlink()


def load_raw(root: Path, name: str) -> np.ndarray:
    """(149, N) array for one of FILES' keys."""
    download(root)
    return np.load(root / f"{FILES[name]}.npy")


def mid_price(raw: np.ndarray) -> np.ndarray:
    return (raw[0].astype(np.float64) + raw[2]) / 2


def stock_segments(raw: np.ndarray, jump: float = 0.05) -> list[tuple[int, int]]:
    """[start, end) row ranges of the five stocks, split where |log mid change| > jump."""
    m = mid_price(raw)
    cuts = np.where(np.abs(np.diff(np.log(m))) > jump)[0] + 1
    edges = np.r_[0, cuts, len(m)]
    return list(zip(edges[:-1].tolist(), edges[1:].tolist()))


@dataclass
class Split:
    """Order book (N, 40) and class labels (N,) in {0 down, 1 stationary, 2 up}, plus the
    row indices that may end an input window (the window must lie inside one stock)."""
    lob: np.ndarray
    y: np.ndarray
    ends: np.ndarray


def to_class(label_row: np.ndarray) -> np.ndarray:
    """FI-2010 codes 1 up / 2 stationary / 3 down -> 2 / 1 / 0, so 'bigger = higher'."""
    return (3 - label_row).astype(np.int64)


def make_split(raw: np.ndarray, horizon: int, window: int = 100,
               segments: list[tuple[int, int]] | None = None) -> Split:
    """Windows of `window` events ending at row t, labelled with the label stored at row t.

    `segments` restricts which rows can end a window (default: every stock's whole range).
    """
    h = HORIZONS.index(horizon)
    segments = segments or stock_segments(raw)
    ends = np.concatenate([np.arange(s + window - 1, e) for s, e in segments])
    return Split(lob=np.ascontiguousarray(raw[:N_LOB].T), y=to_class(raw[N_LOB + 104 + h]), ends=ends)


def train_val_segments(raw: np.ndarray, val_frac: float = 0.2, gap: int = 100):
    """Time-ordered split *within each stock*: first 80% of each stock's events to train,
    last 20% to validation, with `gap` events dropped in between so that no training label
    (which looks up to 100 events ahead) overlaps a validation window.

    The DeepLOB reference notebook instead takes the last 20% of the file, which is almost
    entirely stock 5: its 'validation set' is a different stock, not later data.
    """
    tr, va = [], []
    for s, e in stock_segments(raw):
        cut = s + int((e - s) * (1 - val_frac))
        tr.append((s, cut - gap))
        va.append((cut, e))
    return tr, va


def windows(split: Split, idx: np.ndarray, window: int = 100) -> np.ndarray:
    """Materialise input windows for the given positions in split.ends -> (n, window, 40)."""
    t = split.ends[idx]
    return split.lob[t[:, None] - np.arange(window - 1, -1, -1)[None, :]]
