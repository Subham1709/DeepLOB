"""Classification and fee-aware backtest summary for Binance. Writes results/binance_summary.md.

For each model family (logistic regression on past returns, logistic regression on book
features, LightGBM, DeepLOB) the config is chosen on validation macro F1; that config's test
predictions are then backtested at several fee levels (bps per side, on notional).
Binance spot fees: 10 bp per side at the base tier, 7.5 with the BNB discount; only
high-volume tiers and market makers get close to 0-1 bp.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deeplob import binance as B  # noqa: E402
from deeplob.backtest import backtest  # noqa: E402

R = ROOT / "results" / "binance"
HORIZONS = (100, 300, 600)
FEES = (0.0, 1.0, 7.5, 10.0)
base = json.loads((R / "baselines.json").read_text())
books = {s: B.load_symbol(ROOT / "data" / "binance", s) for s in B.SYMBOLS}


def families(h):
    res = base[str(h)]["res"]
    fam = {}
    for label, prefix in (("Logistic regression, past returns only", "logreg_past"),
                          ("Logistic regression, book features", "logreg_all"),
                          ("LightGBM, book features", "lightgbm")):
        name = max((n for n in res if n.startswith(prefix)), key=lambda n: res[n]["val"]["macro_f1"])
        fam[label] = (name, res[name])
    dl = {p.stem.split("_")[-1]: json.loads(p.read_text()) for p in sorted(R.glob(f"deeplob_h{h}_*.json"))}
    if dl:
        c = max(dl, key=lambda c: dl[c]["val"]["macro_f1"])
        fam["DeepLOB"] = (f"deeplob_{c}", dl[c])
    return fam, dl


out = ["# Binance results", "",
       "BTCUSDT and ETHUSDT spot, top 10 levels every 100 ms, recorded from Binance's public "
       "WebSocket. Chronological split 60 / 20 / 20 with an embargo; labels from future mids only, "
       "classes balanced on the training period. Configs chosen on validation macro F1.", ""]
# The recorder keeps appending chunks; the models saw data up to `data_end_ms`. Later rows
# don't shift earlier indices (chunks are appended in time order), so saved ends stay valid.
end_ms = max(json.loads(p.read_text())["data_end_ms"] for p in R.glob("deeplob_h*_*.json"))
t = next(iter(books.values())).t
t = t[t <= end_ms]
hours = sum((t[e - 1] - t[s]) / 3.6e6 for s, e in B.sessions(t))
out.append(f"Data: {len(t):,} BTC snapshots, {hours:.1f} h of recording "
           f"({datetime.fromtimestamp(t[0] / 1000, timezone.utc):%Y-%m-%d %H:%M} to "
           f"{datetime.fromtimestamp(t[-1] / 1000, timezone.utc):%Y-%m-%d %H:%M} UTC).")
out += ["", "## Classification (test macro F1, both symbols)", "",
        "| model | 10 s | 30 s | 60 s |", "|---|---|---|---|"]
table, chosen = {}, {}
for h in HORIZONS:
    fam, dl = families(h)
    chosen[h] = fam
    for label, (name, r) in fam.items():
        table.setdefault(label, {})[h] = f"{r['test']['macro_f1']:.3f}" + (f" ({name.split('_')[-1]})" if label == "DeepLOB" else "")
for label, row in table.items():
    out.append(f"| {label} | " + " | ".join(row.get(h, "-") for h in HORIZONS) + " |")
out.append("| Chance (balanced classes) | 0.333 | 0.333 | 0.333 |")

dl_rows = []
for h in HORIZONS:
    _, dl = families(h)
    for c, r in dl.items():
        dl_rows.append(f"| {h // 10} s | {c} | {len(r['history'])} ({r['best_epoch']}) | "
                       f"{r['val']['macro_f1']:.3f} | {r['test']['macro_f1']:.3f} |")
if dl_rows:
    out += ["", "## DeepLOB configs tried", "", "B: lr 1e-3, batch 128. C: lr 1e-2, Adam eps 1, batch 32 "
            "(paper). E: lr 3e-4, batch 128, AdamW weight decay 1e-4.", "",
            "| horizon | config | epochs (best) | val macro F1 | test macro F1 |", "|---|---|---|---|---|"] + dl_rows

out += ["", "## Backtest (test period, taker orders, one unit, decisions every horizon)", "",
        "Net P&L summed over the test period in bps of notional. Break-even fee: the fee per side "
        "at which net P&L would be zero, after paying the spread.", "",
        "| horizon | model | symbol | trades | hit rate | gross bps / trade | break-even fee | "
        + " | ".join(f"net @ {f:g} bp" for f in FEES) + " |",
        "|---|---|---|---|---|---|---|" + "---|" * len(FEES)]
for h in HORIZONS:
    for label, (name, _) in chosen[h].items():
        if label.startswith("Logistic regression, past"):
            continue
        for s, b in books.items():
            z = np.load(R / f"pred_{name}_h{h}_{s}.npz")
            bts = [backtest(b.mid, b.bid1, b.ask1, z["ends"], z["prob"], h, f) for f in FEES]
            bt = bts[0]
            out.append(f"| {h // 10} s | {label} | {s[:3]} | {bt['active']} | {bt['hit_rate']:.3f} | "
                       f"{bt['gross_bps_per_active']:.2f} | {bt['breakeven_fee_bps']:.2f} | "
                       + " | ".join(f"{x['net_bps_total']:,.0f}" for x in bts) + " |")

(ROOT / "results" / "binance_summary.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
