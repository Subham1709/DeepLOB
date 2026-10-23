"""Record Binance spot order book snapshots (top 10 levels, every 100 ms).

Binance's partial-depth stream ``<symbol>@depth20@100ms`` pushes the top 20 bid and ask
levels every 100 ms. We keep the top 10 (what DeepLOB uses) and store them exactly, as
integers: price in ticks and quantity in lot steps, so nothing is lost to float rounding.

Files are written every ``chunk_minutes`` to ``<out>/<SYMBOL>/<YYYYMMDD_HHMM>.npz``. Prices
and quantities are delta-encoded along time before compression: between two 100 ms
snapshots most levels don't change, so the differences are mostly zero and compress well.
``load_chunk`` undoes this.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np

# Binance's public market-data mirror. api.binance.com / stream.binance.com refuse US IPs
# (HTTP 451), which is where GitHub's runners live; these mirror hosts serve the same data.
REST = "https://data-api.binance.vision/api/v3"
WS = "wss://data-stream.binance.vision/stream?streams="
LEVELS = 10


def symbol_filters(symbol: str) -> tuple[float, float]:
    """(tick size, lot step) of a spot symbol, from exchangeInfo."""
    with urlopen(f"{REST}/exchangeInfo?symbol={symbol}", timeout=30) as r:
        info = json.load(r)["symbols"][0]
    f = {x["filterType"]: x for x in info["filters"]}
    return float(f["PRICE_FILTER"]["tickSize"]), float(f["LOT_SIZE"]["stepSize"])


def encode_levels(levels: list[list[str]], tick: float, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Top LEVELS (price, qty) string pairs -> integer ticks and lot steps.

    A side with fewer than LEVELS levels is padded with price 0 / qty 0 (never happens for
    BTC or ETH, but keeps the shapes fixed).
    """
    p = np.zeros(LEVELS, np.int64)
    q = np.zeros(LEVELS, np.int64)
    for i, (ps, qs) in enumerate(levels[:LEVELS]):
        p[i] = round(float(ps) / tick)
        q[i] = round(float(qs) / step)
    return p, q


@dataclass
class Buffer:
    symbol: str
    tick: float
    step: float
    t: list = field(default_factory=list)    # local receive time, ms since epoch
    uid: list = field(default_factory=list)  # Binance lastUpdateId
    rows: list = field(default_factory=list)  # (ask_p, ask_q, bid_p, bid_q), each LEVELS ints

    def add(self, t_ms: int, msg: dict) -> None:
        ap, aq = encode_levels(msg["asks"], self.tick, self.step)
        bp, bq = encode_levels(msg["bids"], self.tick, self.step)
        self.t.append(t_ms)
        self.uid.append(msg["lastUpdateId"])
        self.rows.append(np.concatenate([ap, aq, bp, bq]))

    def flush(self, out: Path) -> Path | None:
        if not self.t:
            return None
        t = np.asarray(self.t, np.int64)
        start = datetime.fromtimestamp(t[0] / 1000, timezone.utc).strftime("%Y%m%d_%H%M")
        path = out / self.symbol / f"{start}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_chunk(path, t, np.asarray(self.uid, np.int64), np.vstack(self.rows), self.tick, self.step)
        self.t, self.uid, self.rows = [], [], []
        return path


def save_chunk(path: Path, t, uid, book, tick, step) -> None:
    """book: (n, 4*LEVELS) int64 in the order ask_p, ask_q, bid_p, bid_q."""
    d = np.diff(book, axis=0, prepend=np.zeros((1, book.shape[1]), np.int64))
    np.savez_compressed(path, t=np.diff(t, prepend=0), uid=np.diff(uid, prepend=0),
                        book=d, tick=tick, step=step)


def load_chunk(path: Path) -> dict:
    """Decode one chunk: times (ms), update ids, and float prices/quantities (n, LEVELS)."""
    z = np.load(path)
    book = np.cumsum(z["book"], axis=0)
    tick, step = float(z["tick"]), float(z["step"])
    L = LEVELS
    return dict(t=np.cumsum(z["t"]), uid=np.cumsum(z["uid"]),
                ask_p=book[:, :L] * tick, ask_q=book[:, L:2 * L] * step,
                bid_p=book[:, 2 * L:3 * L] * tick, bid_q=book[:, 3 * L:] * step)


async def record(symbols: list[str], minutes: float, out: Path, chunk_minutes: float = 60) -> None:
    import websockets  # only the recorder needs it

    bufs = {s: Buffer(s, *symbol_filters(s)) for s in symbols}
    url = WS + "/".join(f"{s.lower()}@depth20@100ms" for s in symbols)
    stop = time.time() + 60 * minutes
    next_flush = time.time() + 60 * chunk_minutes
    while time.time() < stop:
        try:
            async with websockets.connect(url, open_timeout=20, ping_interval=60) as ws:
                print("connected", url, flush=True)
                while time.time() < stop:
                    raw = await asyncio.wait_for(ws.recv(), 30)
                    t_ms = int(time.time() * 1000)
                    msg = json.loads(raw)
                    bufs[msg["stream"].split("@")[0].upper()].add(t_ms, msg["data"])
                    if time.time() >= next_flush:
                        for b in bufs.values():
                            print("wrote", b.flush(out), flush=True)
                        next_flush += 60 * chunk_minutes
        except Exception as e:  # dropped connection: log, wait, reconnect (a gap in t shows it)
            print("connection error:", repr(e), flush=True)
            await asyncio.sleep(5)
    for b in bufs.values():
        print("wrote", b.flush(out), flush=True)
