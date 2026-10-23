"""Download FI-2010 and run sanity checks on it. Writes results/fi2010_checks.md.

1. Shapes, the five stock segments, class balance per horizon.
2. Can the labels be recomputed from the mid-prices in the file? (Two textbook
   definitions, threshold chosen to match the class shares as closely as possible.)
3. How much of the label is already known at prediction time? Logistic regression on
   nothing but the past mid-price returns inside the input window, train -> test.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deeplob import fi2010 as F  # noqa: E402

DATA = ROOT / "data" / "fi2010"
LAGS = (1, 2, 3, 5, 10, 20, 50, 99)
out = ["# FI-2010 data checks", ""]

raws = {n: F.load_raw(DATA, n) for n in F.FILES}
out += ["## Files", "", "| file | events | stock segments | down / flat / up share at k=10 |", "|---|---|---|---|"]
for n, r in raws.items():
    y = F.to_class(r[144])
    share = " / ".join(f"{np.mean(y == c):.2f}" for c in (0, 1, 2))
    out.append(f"| {n} | {r.shape[1]:,} | {[e - s for s, e in F.stock_segments(r)]} | {share} |")

out += ["", "## Class balance (all test days)", "", "| k | down | flat | up |", "|---|---|---|---|"]
test = np.concatenate([raws[n] for n in ("test8", "test9", "test10")], axis=1)
for h, k in enumerate(F.HORIZONS):
    y = F.to_class(test[144 + h])
    out.append(f"| {k} | " + " | ".join(f"{np.mean(y == c):.3f}" for c in (0, 1, 2)) + " |")


def relabel(raw, k, kind):
    """Recompute labels; threshold set from the file's own class shares."""
    lab_all, l_all = [], []
    m_all = F.mid_price(raw)
    y_all = F.to_class(raw[144 + F.HORIZONS.index(k)])
    for s, e in F.stock_segments(raw):
        m = m_all[s:e]
        c = np.cumsum(np.r_[0, m])
        t = np.arange(k, len(m) - k)
        m_plus = (c[t + 1 + k] - c[t + 1]) / k          # mean of the next k mids
        m_minus = (c[t + 1] - c[t + 1 - k]) / k         # mean of the last k mids (incl. t)
        base = m[t] if kind == "next-k mean vs current" else m_minus
        l_all.append((m_plus - base) / base)
        lab_all.append(y_all[s:e][t])
    l, y = np.concatenate(l_all), np.concatenate(lab_all)
    hi, lo = np.quantile(l, 1 - np.mean(y == 2)), np.quantile(l, np.mean(y == 0))
    pred = np.where(l > hi, 2, np.where(l < lo, 0, 1))
    return np.mean(pred == y)


out += ["", "## Can the labels be recomputed from the file's mid-prices?", "",
        "Agreement between the stored label and a recomputed one (test day 8).", "",
        "| k | next-k mean vs current | next-k mean vs last-k mean |", "|---|---|---|"]
for k in (10, 50, 100):
    a = relabel(raws["test8"], k, "next-k mean vs current")
    b = relabel(raws["test8"], k, "next-k mean vs last-k mean")
    out.append(f"| {k} | {a:.3f} | {b:.3f} |")


def past_returns(raw, k):
    s = F.make_split(raw, k)
    lm = np.log(F.mid_price(raw))
    t = s.ends
    return np.stack([(lm[t] - lm[t - g]) * 1e4 for g in LAGS], 1), s.y[t]


out += ["", "## How much of the label is known at prediction time?", "",
        f"Logistic regression on {len(LAGS)} past mid-price returns (lags {LAGS} events), all of"
        " which are inside the 100-event input window. Train days 1-7, test days 8-10.", "",
        "| k | accuracy | macro F1 | DeepLOB paper F1 | always-majority accuracy |", "|---|---|---|---|---|"]
paper = {10: "83.40", 20: "72.82", 50: "80.35", 100: "-"}
for k in (10, 20, 50, 100):
    X, y = past_returns(raws["train"], k)
    Xt, yt = zip(*[past_returns(raws[n], k) for n in ("test8", "test9", "test10")])
    Xt, yt = np.vstack(Xt), np.concatenate(yt)
    p = LogisticRegression(max_iter=2000).fit(X, y).predict(Xt)
    out.append(f"| {k} | {accuracy_score(yt, p):.3f} | {f1_score(yt, p, average='macro'):.3f} | "
               f"{paper[k]} | {np.mean(yt == np.bincount(y).argmax()):.3f} |")

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "fi2010_checks.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
