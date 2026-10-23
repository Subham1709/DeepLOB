# FI-2010 data checks

## Files

| file | events | stock segments | down / flat / up share at k=10 |
|---|---|---|---|
| train | 254,750 | [23336, 44874, 38549, 54675, 93316] | 0.20 / 0.61 / 0.20 |
| test8 | 55,478 | [2647, 8662, 8591, 13229, 22349] | 0.13 / 0.73 / 0.14 |
| test9 | 52,172 | [1873, 9271, 10036, 12880, 18112] | 0.15 / 0.70 / 0.15 |
| test10 | 31,937 | [1888, 5128, 5722, 5821, 13378] | 0.16 / 0.67 / 0.17 |

## Class balance (all test days)

| k | down | flat | up |
|---|---|---|---|
| 10 | 0.142 | 0.707 | 0.152 |
| 20 | 0.183 | 0.621 | 0.197 |
| 30 | 0.210 | 0.561 | 0.229 |
| 50 | 0.252 | 0.473 | 0.276 |
| 100 | 0.312 | 0.344 | 0.344 |

## Can the labels be recomputed from the file's mid-prices?

Agreement between the stored label and a recomputed one (test day 8).

| k | next-k mean vs current | next-k mean vs last-k mean |
|---|---|---|
| 10 | 0.639 | 0.691 |
| 50 | 0.489 | 0.553 |
| 100 | 0.429 | 0.505 |

## How much of the label is known at prediction time?

Logistic regression on 8 past mid-price returns (lags (1, 2, 3, 5, 10, 20, 50, 99) events), all of which are inside the 100-event input window. Train days 1-7, test days 8-10.

| k | accuracy | macro F1 | DeepLOB paper F1 | always-majority accuracy |
|---|---|---|---|---|
| 10 | 0.877 | 0.799 | 83.40 | 0.708 |
| 20 | 0.786 | 0.689 | 72.82 | 0.622 |
| 50 | 0.805 | 0.789 | 80.35 | 0.274 |
| 100 | 0.605 | 0.528 | - | 0.342 |
