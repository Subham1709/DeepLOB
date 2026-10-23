# FI-2010 results

Train: days 1-7 (last 20% of each stock held out for validation). Test: days 8-10.
Metric: macro F1 (mean of the down / flat / up F1 scores). Config chosen on validation.

## DeepLOB: the configs tried

| k | config | epochs run (best) | val macro F1 | test macro F1 | test accuracy |
|---|---|---|---|---|---|
| 10 | A: lr 1e-4, batch 64 (authors' notebook) | 35 (25) | 0.574 | 0.599 | 0.775 |
| 10 | B: lr 1e-3, batch 128 **(chosen)** | 45 (35) | 0.644 | 0.667 | 0.804 |
| 10 | C: lr 1e-2, Adam eps 1, batch 32 (paper) | 33 (23) | 0.618 | 0.644 | 0.797 |
| 20 | A: lr 1e-4, batch 64 (authors' notebook) | 30 (20) | 0.440 | 0.459 | 0.632 |
| 20 | B: lr 1e-3, batch 128 **(chosen)** | 20 (10) | 0.581 | 0.609 | 0.728 |
| 20 | C: lr 1e-2, Adam eps 1, batch 32 (paper) | 36 (26) | 0.548 | 0.579 | 0.710 |
| 50 | A: lr 1e-4, batch 64 (authors' notebook) | 23 (13) | 0.653 | 0.679 | 0.708 |
| 50 | B: lr 1e-3, batch 128 **(chosen)** | 22 (12) | 0.697 | 0.724 | 0.751 |
| 50 | C: lr 1e-2, Adam eps 1, batch 32 (paper) | 35 (25) | 0.691 | 0.712 | 0.735 |

## Comparison (test macro F1, each with its validation-chosen config)

| model | k = 10 | k = 20 | k = 50 |
|---|---|---|---|
| Logistic regression, 8 past returns only | 0.800 | 0.690 | 0.786 |
| Logistic regression, 53 book features | 0.799 | 0.686 | 0.795 |
| LightGBM, 53 book features | 0.806 | 0.716 | 0.832 |
| DeepLOB, ours | 0.667 (B) | 0.609 (B) | 0.724 (B) |
| DeepLOB, paper (Zhang et al. 2019) | 0.834 | 0.728 | 0.803 |
