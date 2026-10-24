# Binance results

BTCUSDT and ETHUSDT spot, top 10 levels every 100 ms, recorded from Binance's public WebSocket. Chronological split 60 / 20 / 20 with an embargo; labels from future mids only, classes balanced on the training period. Configs chosen on validation macro F1.

Data: 1,105,240 BTC snapshots, recording time 30.9 h.

## Classification (test macro F1, both symbols)

| model | 10 s | 30 s | 60 s |
|---|---|---|---|
| Logistic regression, past returns only | 0.308 | 0.311 | 0.306 |
| Logistic regression, book features | 0.524 | 0.450 | 0.407 |
| LightGBM, book features | 0.531 | 0.384 | 0.337 |
| Chance (balanced classes) | 0.333 | 0.333 | 0.333 |

## Backtest (test period, taker orders, one unit, decisions every horizon)

Net P&L summed over the test period in bps of notional. Break-even fee: the fee per side at which net P&L would be zero, after paying the spread.

| horizon | model | symbol | trades | hit rate | gross bps / trade | break-even fee | net @ 0 bp | net @ 1 bp | net @ 7.5 bp | net @ 10 bp |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 s | Logistic regression, book features | BTC | 985 | 0.550 | 0.48 | 0.39 | 475 | -751 | -8,720 | -11,785 |
| 10 s | Logistic regression, book features | ETH | 1792 | 0.533 | 0.62 | 0.58 | 1,079 | -789 | -12,931 | -17,601 |
| 10 s | LightGBM, book features | BTC | 1342 | 0.499 | 0.32 | 0.29 | 434 | -1,044 | -10,651 | -14,346 |
| 10 s | LightGBM, book features | ETH | 2141 | 0.507 | 0.48 | 0.55 | 998 | -812 | -12,577 | -17,102 |
| 30 s | Logistic regression, book features | BTC | 616 | 0.531 | 0.33 | 0.31 | 203 | -447 | -4,672 | -6,297 |
| 30 s | Logistic regression, book features | ETH | 614 | 0.531 | 0.17 | 0.13 | 90 | -586 | -4,980 | -6,670 |
| 30 s | LightGBM, book features | BTC | 707 | 0.539 | 0.42 | 0.44 | 293 | -369 | -4,672 | -6,327 |
| 30 s | LightGBM, book features | ETH | 714 | 0.510 | 0.48 | 0.48 | 328 | -358 | -4,817 | -6,532 |
| 60 s | Logistic regression, book features | BTC | 310 | 0.532 | 0.69 | 0.66 | 214 | -112 | -2,231 | -3,046 |
| 60 s | Logistic regression, book features | ETH | 308 | 0.536 | -0.24 | -0.23 | -82 | -438 | -2,752 | -3,642 |
| 60 s | LightGBM, book features | BTC | 363 | 0.493 | 0.45 | 0.46 | 161 | -193 | -2,494 | -3,379 |
| 60 s | LightGBM, book features | ETH | 357 | 0.543 | 0.15 | 0.14 | 45 | -281 | -2,400 | -3,215 |
