"""Cut the small slices of data the web app needs out of the recorded order books.

    python app/precompute.py

The full recording is millions of 100 ms snapshots (and keeps growing), too much for a
free web host's memory. The app only needs the test period, so this writes
app/assets/binance_<SYMBOL>.npz with:
  t, mid, bid1, ask1     the test period (plus the last horizon), indices shifted to start at 0
  book_*                 the full top-10 book for the first 10 minutes of the test period
  pred_<family>_h<h>     prediction times (shifted) and class probabilities of the model
                         chosen on validation in each family (as in scripts/07_backtest_binance.py)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deeplob import binance as B  # noqa: E402
from deeplob.recorder import load_chunk  # noqa: E402

R = ROOT / "results" / "binance"
OUT = ROOT / "app" / "assets"
HORIZONS = (100, 300, 600)
FAMILIES = {"logreg_past": "logreg_past", "logreg_book": "logreg_all",
            "lightgbm": "lightgbm"}
BOOK_ROWS = 6000  # 10 minutes at 100 ms


def chosen(h):
    """Prediction file stem of the validation-best config in each family."""
    res = json.loads((R / "baselines.json").read_text())[str(h)]["res"]
    out = {fam: max((n for n in res if n.startswith(prefix)),
                    key=lambda n: res[n]["val"]["macro_f1"])
           for fam, prefix in FAMILIES.items()}
    dl = {p.stem.split("_")[-1]: json.loads(p.read_text())
          for p in sorted(R.glob(f"deeplob_h{h}_*.json"))}
    out["deeplob"] = "deeplob_" + max(dl, key=lambda c: dl[c]["val"]["macro_f1"])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    picks = {h: chosen(h) for h in HORIZONS}
    for sym in B.SYMBOLS:
        book = B.load_symbol(ROOT / "data" / "binance", sym)
        preds = {}
        for h, fams in picks.items():
            for fam, stem in fams.items():
                parts = stem.split("_")
                fname = f"pred_{'_'.join(parts)}_h{h}_{sym}.npz"
                if stem.startswith("deeplob_"):
                    fname = f"pred_deeplob_{parts[1]}_h{h}_{sym}.npz"
                z = np.load(R / fname)
                preds[(fam, h)] = (z["ends"], z["prob"])
        lo = min(e.min() for e, _ in preds.values())
        hi = max(e.max() for e, _ in preds.values()) + max(HORIZONS) + 1
        arrays = dict(t=book.t[lo:hi], mid=book.mid[lo:hi], bid1=book.bid1[lo:hi],
                      ask1=book.ask1[lo:hi])
        for (fam, h), (e, p) in preds.items():
            arrays[f"ends_{fam}_h{h}"] = (e - lo).astype(np.int32)
            arrays[f"prob_{fam}_h{h}"] = p.astype(np.float32)
        # raw top-10 book for the depth chart: re-read the chunks (Book keeps logs only)
        t0 = book.t[lo]
        for path in sorted((ROOT / "data" / "binance" / sym).glob("*.npz")):
            c = load_chunk(path)
            k = np.searchsorted(c["t"], t0)
            if k < len(c["t"]):
                for key in ("t", "ask_p", "ask_q", "bid_p", "bid_q"):
                    arrays[f"book_{key}"] = c[key][k:k + BOOK_ROWS]
                break
        np.savez_compressed(OUT / f"binance_{sym}.npz", **arrays)
        print(sym, {k: v.shape for k, v in arrays.items() if not k.startswith(("ends", "prob"))},
              (OUT / f"binance_{sym}.npz").stat().st_size // 1024, "KB")
    (OUT / "chosen.json").write_text(json.dumps({str(h): f for h, f in picks.items()}, indent=1))


if __name__ == "__main__":
    main()
