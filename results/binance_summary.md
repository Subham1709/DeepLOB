# Binance results

BTCUSDT and ETHUSDT spot, top 10 levels every 100 ms, recorded from Binance's public WebSocket. Chronological split 60 / 20 / 20 with an embargo; labels from future mids only, classes balanced on the training period. Configs chosen on validation macro F1.

Data: 6,177,481 BTC snapshots, 172.6 h of recording (2026-10-23 09:17 to 2026-10-30 19:07 UTC).

## Classification (test macro F1, both symbols)

| model | 10 s | 30 s | 60 s |
|---|---|---|---|
| Logistic regression, past returns only | 0.259 | 0.337 | 0.315 |
| Logistic regression, book features | 0.485 | 0.426 | 0.409 |
| LightGBM, book features | 0.466 | 0.372 | 0.345 |
| DeepLOB | 0.499 (E) | 0.421 (B) | 0.395 (B) |
| Chance (balanced classes) | 0.333 | 0.333 | 0.333 |

## DeepLOB configs tried

B: lr 1e-3, batch 128. C: lr 1e-2, Adam eps 1, batch 32 (paper). E: lr 3e-4, batch 128, AdamW weight decay 1e-4.

| horizon | config | epochs (best) | val macro F1 | test macro F1 |
|---|---|---|---|---|
| 10 s | B | 10 (5) | 0.534 | 0.502 |
| 10 s | C | 16 (11) | 0.526 | 0.496 |
| 10 s | E | 10 (5) | 0.538 | 0.499 |
| 30 s | B | 11 (6) | 0.472 | 0.421 |
| 30 s | C | 14 (9) | 0.465 | 0.427 |
| 30 s | E | 11 (6) | 0.468 | 0.425 |
| 60 s | B | 17 (12) | 0.438 | 0.395 |
| 60 s | C | 7 (2) | 0.422 | 0.401 |
| 60 s | E | 10 (5) | 0.430 | 0.392 |

## Backtest (test period, taker orders, one unit, decisions every horizon)

Net P&L summed over the test period in bps of notional. Break-even fee: the fee per side at which net P&L would be zero, after paying the spread.

| horizon | model | symbol | trades | hit rate | gross bps / trade | break-even fee | net @ 0 bp | net @ 1 bp | net @ 7.5 bp | net @ 10 bp |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 s | Logistic regression, book features | BTC | 9158 | 0.574 | 0.59 | 0.56 | 5,387 | -4,163 | -66,238 | -90,113 |
| 10 s | Logistic regression, book features | ETH | 10713 | 0.550 | 0.49 | 0.45 | 4,969 | -6,175 | -78,611 | -106,471 |
| 10 s | LightGBM, book features | BTC | 11003 | 0.548 | 0.52 | 0.58 | 5,743 | -4,127 | -68,282 | -92,957 |
| 10 s | LightGBM, book features | ETH | 11635 | 0.534 | 0.41 | 0.41 | 4,485 | -6,381 | -77,010 | -104,175 |
| 10 s | DeepLOB | BTC | 10633 | 0.558 | 0.53 | 0.53 | 5,617 | -4,949 | -73,628 | -100,043 |
| 10 s | DeepLOB | ETH | 11048 | 0.545 | 0.47 | 0.43 | 4,887 | -6,391 | -79,698 | -107,893 |
| 30 s | Logistic regression, book features | BTC | 3693 | 0.555 | 0.51 | 0.51 | 1,867 | -1,795 | -25,598 | -34,753 |
| 30 s | Logistic regression, book features | ETH | 3543 | 0.528 | 0.38 | 0.34 | 1,269 | -2,475 | -26,811 | -36,171 |
| 30 s | LightGBM, book features | BTC | 3953 | 0.545 | 0.60 | 0.69 | 2,372 | -1,084 | -23,548 | -32,188 |
| 30 s | LightGBM, book features | ETH | 3885 | 0.521 | 0.34 | 0.43 | 1,243 | -1,679 | -20,672 | -27,977 |
| 30 s | DeepLOB | BTC | 3777 | 0.546 | 0.54 | 0.55 | 2,054 | -1,648 | -25,711 | -34,966 |
| 30 s | DeepLOB | ETH | 3711 | 0.531 | 0.53 | 0.51 | 1,896 | -1,828 | -26,034 | -35,344 |
| 60 s | Logistic regression, book features | BTC | 1847 | 0.533 | 0.68 | 0.68 | 1,258 | -588 | -12,587 | -17,202 |
| 60 s | Logistic regression, book features | ETH | 1587 | 0.542 | 0.69 | 0.58 | 1,058 | -752 | -12,517 | -17,042 |
| 60 s | LightGBM, book features | BTC | 1980 | 0.529 | 0.55 | 0.68 | 1,096 | -524 | -11,054 | -15,104 |
| 60 s | LightGBM, book features | ETH | 1947 | 0.525 | 0.29 | 0.47 | 531 | -591 | -7,884 | -10,689 |
| 60 s | DeepLOB | BTC | 1915 | 0.515 | 0.43 | 0.43 | 815 | -1,097 | -13,525 | -18,305 |
| 60 s | DeepLOB | ETH | 1840 | 0.527 | 0.62 | 0.59 | 1,096 | -752 | -12,764 | -17,384 |
