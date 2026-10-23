# DeepLOB: predicting mid-price moves from limit order books

A from-scratch PyTorch implementation of DeepLOB (Zhang, Zohren & Roberts, 2019) on the
FI-2010 benchmark and on Binance BTCUSDT / ETHUSDT Level-2 data recorded for this project,
with logistic regression and LightGBM baselines, a fee-aware trading backtest and ONNX
latency measurements.

**Status: in progress.** Data, checks and the model are done; training comes next.
The step-by-step write-up is in [docs/LEARNING_LOG.md](docs/LEARNING_LOG.md).

## Finding so far: FI-2010's labels are mostly known in advance

A logistic regression on just 8 *past* mid-price returns, all inside the model's input
window, gets close to the published DeepLOB scores on FI-2010
([results/fi2010_checks.md](results/fi2010_checks.md)):

| horizon k | macro F1, past returns only | DeepLOB paper |
|---|---|---|
| 10 | 0.799 | 0.834 |
| 20 | 0.689 | 0.728 |
| 50 | 0.789 | 0.804 |

So FI-2010 serves here as a reproduction check, and the Binance labels are built from future
prices only.

## Layout

```
src/deeplob/fi2010.py     FI-2010 download, stock segments, windows, per-stock time split
src/deeplob/recorder.py   Binance top-10 book recorder (100 ms), exact delta-encoded storage
src/deeplob/model.py      DeepLOB network (returns logits)
scripts/01_check_fi2010.py   data checks -> results/fi2010_checks.md
scripts/record_binance.py    run by the record-binance workflow
data/binance/             recorded order book chunks (committed by the workflow)
```

## Running

```
pip install -r requirements.txt
python scripts/01_check_fi2010.py     # downloads FI-2010 (56 MB) into data/fi2010/
pytest -q
```
