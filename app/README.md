# Web app

Live: https://subham-deeplob.streamlit.app (Streamlit Community Cloud, main file
`app/streamlit_app.py`, requirements in `app/requirements.txt`).

```
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

| Tab | What happens |
|---|---|
| The order book | Steps through the recorded top-10 Binance book, 100 ms at a time, for the first 10 minutes of the test period. |
| Trade the predictions | Reruns `src/deeplob/backtest.py` live on every model's saved test predictions, at the fee and probability threshold you choose; cumulative net P&L and break-even fee. |
| Results | Test macro F1 table, DeepLOB validation curves, ONNX latency. |
| FI-2010 finding | Why FI-2010's labels leak, and the reproduction table. |

`precompute.py` cuts what the app needs out of the full recording (`data/binance/`, too big
for a free host's memory): the test-period mid, bid and ask, a 10-minute slice of the full
book, and the predictions of the validation-chosen model in each family, into
`assets/`. Rerun it after retraining. The app needs no torch: no model runs live.
Checked: the app's backtest reproduces every net P&L figure in `results/binance_summary.md`.
