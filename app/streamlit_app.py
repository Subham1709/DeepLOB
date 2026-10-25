"""DeepLOB: interactive web app.

    streamlit run app/streamlit_app.py

Reads the slices written by app/precompute.py (test period of the Binance recording and
every model's saved test predictions) and reruns the fee-aware backtest live with
src/deeplob/backtest.py. No model runs here and torch is not needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(ROOT / "src"))

from deeplob.backtest import backtest  # noqa: E402

REPO = "https://github.com/Subham1709/DeepLOB"
HORIZONS = {"10 s": 100, "30 s": 300, "60 s": 600}
FAMILIES = {"deeplob": "DeepLOB", "lightgbm": "LightGBM, book features",
            "logreg_book": "Logistic regression, book features",
            "logreg_past": "Logistic regression, past returns only"}
COLORS = {"deeplob": "#d62728", "lightgbm": "#1f77b4", "logreg_book": "#2ca02c",
          "logreg_past": "#999999"}

st.set_page_config(page_title="DeepLOB", page_icon="📊", layout="wide")


@st.cache_data
def load(sym):
    return dict(np.load(APP / "assets" / f"binance_{sym}.npz"))


@st.cache_data
def summary(sym, h, fam, fee, min_prob):
    z = load(sym)
    return backtest(z["mid"], z["bid1"], z["ask1"], z[f"ends_{fam}_h{h}"],
                    z[f"prob_{fam}_h{h}"], h, fee, min_prob)


def equity(z, fam, h, fee, min_prob):
    """Cumulative net bps per decision, same rules as deeplob.backtest (rebuilt to keep the path)."""
    ends, prob = z[f"ends_{fam}_h{h}"], z[f"prob_{fam}_h{h}"]
    take, last = [], -np.inf
    for i, t in enumerate(ends):
        if t >= last + h:
            take.append(i)
            last = t
    t, p = ends[take], prob[take]
    sig = np.where((p[:, 2] >= p.max(1)) & (p[:, 2] > min_prob), 1,
                   np.where((p[:, 0] >= p.max(1)) & (p[:, 0] > min_prob), -1, 0))
    cont = np.r_[False, np.diff(t) == h]
    prev = np.where(cont, np.r_[0, sig[:-1]], 0)
    mid, bid, ask = z["mid"], z["bid1"], z["ask1"]
    gross = sig * (mid[t + h] / mid[t] - 1) * 1e4
    hs = (ask[t] - bid[t]) / 2 / mid[t] * 1e4
    cost = np.abs(sig - prev) * (hs + fee)
    run_end = np.r_[~cont[1:], True]
    hs_c = (ask[t + h] - bid[t + h]) / 2 / mid[t + h] * 1e4
    cost = cost + run_end * np.abs(sig) * (hs_c + fee)
    return z["t"][t], np.cumsum(gross), np.cumsum(gross - cost)


st.title("DeepLOB: predicting price moves from the order book")
st.markdown(
    "A from-scratch PyTorch **DeepLOB** (Zhang, Zohren & Roberts, 2019) trained on **BTC and "
    "ETH order books recorded from Binance every 100 ms** for this project, against logistic "
    "regression and LightGBM baselines. Labels use future prices only. The question at the "
    f"end: can any of it pay the exchange fee? Code: [GitHub]({REPO}).")

tab_book, tab_bt, tab_res, tab_fi = st.tabs(
    ["The order book", "Trade the predictions", "Results", "FI-2010 finding"])

# ------------------------------------------------------------------------------ book
with tab_book:
    sym = st.radio("Symbol", ["BTCUSDT", "ETHUSDT"], horizontal=True, key="sym_book")
    z = load(sym)
    bt_ = z["book_t"]
    st.markdown(
        "What the model sees: the best 10 bid and ask prices with their quantities, a new "
        "snapshot every 100 ms. DeepLOB takes the last **100 snapshots (10 seconds)** as a "
        "100 x 40 image and outputs P(down), P(flat), P(up) for the mid price over the next "
        "10, 30 or 60 seconds.")
    i = st.slider("Snapshot (first 10 minutes of the test period)", 0, len(bt_) - 1, 300,
                  format="%d")
    ts = pd.to_datetime(int(bt_[i]), unit="ms")
    ap, aq, bp, bq = (z[f"book_{k}"][i] for k in ("ask_p", "ask_q", "bid_p", "bid_q"))
    mid = (ap[0] + bp[0]) / 2
    c1, c2 = st.columns([1, 1])
    fig = go.Figure()
    fig.add_trace(go.Bar(y=[f"{x:,.2f}" for x in bp], x=bq, orientation="h", name="bids",
                         marker_color="#2ca02c"))
    fig.add_trace(go.Bar(y=[f"{x:,.2f}" for x in ap], x=-aq, orientation="h", name="asks",
                         marker_color="#d62728"))
    fig.update_layout(title=f"{sym} book at {ts:%Y-%m-%d %H:%M:%S.%f}"[:-3] + " UTC",
                      xaxis_title="quantity (asks drawn to the left)", yaxis_title="price",
                      yaxis=dict(type="category", categoryorder="array",
                                 categoryarray=[f"{x:,.2f}" for x in np.r_[bp[::-1], ap]]),
                      height=520, barmode="overlay")
    c1.plotly_chart(fig, width="stretch")
    lo, hi = max(0, i - 600), min(len(bt_), i + 600)
    mids = (z["book_ask_p"][lo:hi, 0] + z["book_bid_p"][lo:hi, 0]) / 2
    x = pd.to_datetime(bt_[lo:hi], unit="ms")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=z["book_ask_p"][lo:hi, 0], name="best ask",
                             line=dict(color="#d62728", width=1, shape="hv")))
    fig.add_trace(go.Scatter(x=x, y=z["book_bid_p"][lo:hi, 0], name="best bid",
                             line=dict(color="#2ca02c", width=1, shape="hv")))
    fig.add_vline(x=ts, line_dash="dot")
    fig.update_layout(title="Best bid and ask, two minutes around the snapshot", height=520,
                      yaxis_title="price")
    c2.plotly_chart(fig, width="stretch")
    st.caption(f"Spread {ap[0] - bp[0]:.2f} = {(ap[0] - bp[0]) / mid * 1e4:.3f} bp of the mid. "
               "BTC's spread is usually one tick, far below the 10 bp taker fee.")

# ------------------------------------------------------------------------- backtest
with tab_bt:
    st.markdown(
        "Every h seconds the strategy looks at a model's probabilities on the **test period** "
        "(the last 20% of the recording, never used for training or model choice) and holds "
        "+1, -1 or 0 units until the next decision. Trades cross the spread and pay the fee "
        "on notional. Change the fee and watch the edge disappear.")
    c1, c2, c3, c4 = st.columns(4)
    sym = c1.radio("Symbol", ["BTCUSDT", "ETHUSDT"], horizontal=True, key="sym_bt")
    hl = c2.radio("Horizon", list(HORIZONS), horizontal=True, index=2)
    fee = c3.slider("Fee per side (bp)", 0.0, 10.0, 1.0, 0.25,
                    help="Binance spot: 10 bp base, 7.5 with BNB; ~0-1 bp only for top tiers")
    min_prob = c4.slider("Only trade if top probability above", 0.33, 0.8, 0.33, 0.01)
    h = HORIZONS[hl]
    z = load(sym)
    rows = {}
    fig = make_subplots(rows=1, cols=1)
    for fam, label in FAMILIES.items():
        r = summary(sym, h, fam, fee, min_prob)
        rows[label] = {"positions taken": r["active"], "hit rate": r["hit_rate"],
                       "gross bp / position": r["gross_bps_per_active"],
                       "cost bp / position": r["cost_bps_per_active"],
                       "break-even fee (bp)": r["breakeven_fee_bps"],
                       "net bp, total": r["net_bps_total"]}
        tt, g, n = equity(z, fam, h, fee, min_prob)
        fig.add_trace(go.Scatter(x=pd.to_datetime(tt, unit="ms"), y=n, name=label,
                                 line=dict(color=COLORS[fam], width=3 if fam == "deeplob" else 1.5)))
    fig.add_hline(y=0, line_color="gray")
    fig.update_layout(title=f"Cumulative net P&L, {sym}, {hl} horizon, fee {fee:g} bp per side",
                      yaxis_title="bp of notional", height=460, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    df = pd.DataFrame(rows).T
    st.dataframe(df.style.format({"positions taken": "{:,.0f}", "hit rate": "{:.3f}", "gross bp / position": "{:+.2f}",
                                  "cost bp / position": "{:.2f}", "break-even fee (bp)": "{:+.2f}",
                                  "net bp, total": "{:+,.0f}"}), width="stretch")
    best = df["break-even fee (bp)"].max()
    st.info(f"The best model here breaks even at a fee of **{best:.2f} bp** per side. Binance "
            "charges 10 bp (7.5 with BNB). Predicting the direction better than chance is not "
            "the same as making money after costs.")

# --------------------------------------------------------------------------- results
with tab_res:
    st.subheader("Classification on the test period (macro F1, chance = 0.333)")
    st.markdown("""
| model | 10 s | 30 s | 60 s |
|---|---|---|---|
| Logistic regression, past returns only | 0.308 | 0.311 | 0.306 |
| Logistic regression, book features | 0.524 | 0.450 | 0.407 |
| LightGBM, book features | 0.531 | 0.384 | 0.337 |
| **DeepLOB** | **0.541** | **0.466** | **0.435** |
""")
    chosen = json.loads((APP / "assets" / "chosen.json").read_text())
    curves = go.Figure()
    for p in sorted((ROOT / "results" / "binance").glob("deeplob_h*_*.json")):
        d = json.loads(p.read_text())
        hist = d["history"]
        curves.add_trace(go.Scatter(x=[e["epoch"] for e in hist],
                                    y=[e["val_macro_f1"] for e in hist],
                                    name=p.stem.replace("deeplob_", "")))
    curves.update_layout(title="DeepLOB training: validation macro F1 per epoch "
                               "(h100 = 10 s; B, C, E = optimiser settings)",
                         xaxis_title="epoch", yaxis_title="val macro F1", height=380)
    st.plotly_chart(curves, width="stretch")
    st.markdown(
        "Data: about 1.1 million snapshots per symbol, 31 hours of recording (23-24 Oct 2026). "
        "Chronological 60/20/20 split with an embargo; classes balanced on the training "
        f"period; configs chosen on validation (models used here: `{chosen['600']['deeplob']}`, "
        f"`{chosen['600']['lightgbm']}`). Full tables: "
        f"[results/binance_summary.md]({REPO}/blob/main/results/binance_summary.md).")
    st.subheader("Inference latency")
    st.markdown("Batch 1, one CPU thread: **ONNX Runtime 0.81 ms median, 0.93 ms p99**; "
                "PyTorch eager 2.7 ms. Same logits to 4e-7.")

# --------------------------------------------------------------------------- fi2010
with tab_fi:
    st.markdown(
        "FI-2010 is the standard benchmark (Nordic stocks, 10 days). Its labels compare the "
        "*mean of past and future* mid prices, so a good part of the answer is already in "
        "the model's input window. A logistic regression on **8 past returns only** nearly "
        "matches the published DeepLOB scores:")
    st.markdown("""
| model | k = 10 | k = 20 | k = 50 |
|---|---|---|---|
| Logistic regression, 8 past returns only | 0.800 | 0.690 | 0.786 |
| LightGBM, 53 book features | 0.806 | 0.716 | 0.832 |
| DeepLOB, this repo | 0.663 | 0.609 | 0.737 |
| DeepLOB, paper | 0.834 | 0.728 | 0.803 |
""")
    st.caption("So FI-2010 served as a reproduction check only, and the Binance labels are "
               "built from future prices only. Test macro F1; configs chosen on validation.")
