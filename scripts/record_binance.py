"""Record Binance top-10 order book snapshots. Run by the record-binance workflow.

    python scripts/record_binance.py --minutes 340 --symbols BTCUSDT ETHUSDT
"""
import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deeplob.recorder import record  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
ap.add_argument("--minutes", type=float, default=340)
ap.add_argument("--chunk-minutes", type=float, default=60)
ap.add_argument("--out", type=Path, default=ROOT / "data" / "binance")
a = ap.parse_args()
asyncio.run(record(a.symbols, a.minutes, a.out, a.chunk_minutes))
