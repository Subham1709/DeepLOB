# DeepLOB: predicting mid-price moves from limit order books

A from-scratch PyTorch implementation of DeepLOB (Zhang, Zohren & Roberts, 2019) on the
FI-2010 benchmark and on Binance BTCUSDT / ETHUSDT Level-2 data recorded for this project,
with logistic regression and LightGBM baselines, a fee-aware trading backtest and ONNX
latency measurements.

**Status:** FI-2010 done. Binance results below are from the first 31 hours of recording; they will be rerun on a full week of data.
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

## FI-2010 results (test macro F1; hyperparameters chosen on validation only)

| model | k = 10 | k = 20 | k = 50 |
|---|---|---|---|
| Logistic regression, 8 past returns | 0.800 | 0.690 | 0.786 |
| LightGBM, 53 book features | 0.806 | 0.716 | 0.832 |
| DeepLOB, this repo | 0.663 | 0.609 | 0.737 |
| DeepLOB, paper | 0.834 | 0.728 | 0.803 |

Four DeepLOB settings were tried ([results/fi2010_summary.md](results/fi2010_summary.md)).
The network overfits the 7 training days and doesn't reach the published scores; a tree
model on simple features beats it. Details in step 5 of the learning log.

## Binance BTC / ETH results (first 31 hours; test macro F1, chance = 0.333)

| model | 10 s | 30 s | 60 s |
|---|---|---|---|
| Logistic regression, past returns only | 0.308 | 0.311 | 0.306 |
| Logistic regression, book features | 0.524 | 0.450 | 0.407 |
| LightGBM, book features | 0.531 | 0.384 | 0.337 |
| DeepLOB, this repo | **0.541** | **0.466** | **0.435** |

With labels built from future prices only, DeepLOB is the best model. But the
[backtest](results/binance_summary.md) shows none of them can pay Binance's fees: trading the
predictions makes at most +0.9 bp per trade before costs, the break-even fee is at most 0.7 bp
per side, and the base taker fee is 10 bp.

ONNX inference: 0.8 ms median, 0.9 ms p99 per prediction on one CPU core
([results/latency.md](results/latency.md)).

## Layout

```
src/deeplob/fi2010.py     FI-2010 download, stock segments, windows, per-stock time split
src/deeplob/recorder.py   Binance top-10 book recorder (100 ms), exact delta-encoded storage
src/deeplob/model.py      DeepLOB network (returns logits)
src/deeplob/train.py      training loop, validation-based model selection
src/deeplob/features.py   hand-made book features for the baselines
src/deeplob/binance.py    Binance windows, future-only labels
src/deeplob/backtest.py   taker backtest with spread and fees
scripts/01_check_fi2010.py   data checks -> results/fi2010_checks.md
scripts/02_train_fi2010.py   one (k, config) training run; train-fi2010 workflow runs them in parallel
scripts/03_baselines_fi2010.py  logistic regression + LightGBM
scripts/04_summarize_fi2010.py  -> results/fi2010_summary.md
scripts/05_baselines_binance.py / 05_train_binance.py   Binance baselines / DeepLOB (train-binance workflow)
scripts/06_latency.py        ONNX export + latency
scripts/07_backtest_binance.py  -> results/binance_summary.md
scripts/record_binance.py    run by the record-binance workflow
data/binance/             recorded order book chunks (committed by the workflow)
```

## Running

```
pip install -r requirements.txt
python scripts/01_check_fi2010.py     # downloads FI-2010 (56 MB) into data/fi2010/
pytest -q
```
