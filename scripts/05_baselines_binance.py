"""Logistic regression and LightGBM on Binance BTCUSDT / ETHUSDT, three configs each.

Chronological split (60 / 20 / 20 by recording time, with an embargo), labels from future
mids only, classes balanced on the training period. One model per horizon, trained on both
symbols together; scored per symbol. Config chosen on validation macro F1. Saves test
probabilities for the backtest. Writes results/binance/baselines.json and
results/binance/pred_<model>_h<h>.npz.
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deeplob import binance as B  # noqa: E402
from deeplob.features import BINANCE_LAGS, binance_features  # noqa: E402
from deeplob.train import scores  # noqa: E402

HORIZONS = (100, 300, 600)      # 10 s, 30 s, 60 s
STRIDE = 10                     # one prediction per second
OUT = ROOT / "results" / "binance"
OUT.mkdir(parents=True, exist_ok=True)

results = {}
for h in HORIZONS:
    preps = [B.prepare(ROOT / "data" / "binance", s, h) for s in B.SYMBOLS]
    data = {}
    for split in ("train", "val", "test"):
        Xs, ys, ids = [], [], []
        for i, p in enumerate(preps):
            e = p.ends[split][::STRIDE]
            X, names = binance_features(p.book, e)
            Xs.append(np.column_stack([X, np.full(len(e), i)]))   # symbol id as a feature
            ys.append(p.y[e])
            ids.append(np.full(len(e), i))
        data[split] = (np.vstack(Xs), np.concatenate(ys), np.concatenate(ids))
    Xtr, ytr, _ = data["train"]
    Xva, yva, _ = data["val"]
    Xte, yte, idte = data["test"]
    past = list(range(len(BINANCE_LAGS)))
    models = {}
    for C in (0.01, 0.1, 1.0):
        models[f"logreg_past_C{C}"] = (make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)), past)
        models[f"logreg_all_C{C}"] = (make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)), None)
    for leaves in (15, 31, 63):
        models[f"lightgbm_L{leaves}"] = (lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=leaves, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, verbose=-1, random_state=0), None)
    res = {}
    for name, (m, cols) in models.items():
        c = slice(None) if cols is None else cols
        if name.startswith("lightgbm"):
            m.fit(Xtr[:, c], ytr, eval_X=(Xva[:, c],), eval_y=(yva,),
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        else:
            m.fit(Xtr[:, c], ytr)
        pva, pte = m.predict_proba(Xva[:, c]), m.predict_proba(Xte[:, c])
        res[name] = dict(val=scores(yva, pva), test=scores(yte, pte),
                         test_by_symbol={s: scores(yte[idte == i], pte[idte == i]) for i, s in enumerate(B.SYMBOLS)})
        print(h, name, "val", round(res[name]["val"]["macro_f1"], 4), "test", round(res[name]["test"]["macro_f1"], 4), flush=True)
        for i, s in enumerate(B.SYMBOLS):
            np.savez_compressed(OUT / f"pred_{name}_h{h}_{s}.npz", ends=preps[i].ends["test"][::STRIDE],
                                prob=pte[idte == i].astype(np.float32))
    results[h] = dict(res=res, alpha_bps={p.symbol: p.alpha * 1e4 for p in preps},
                      n={k: int(len(v[1])) for k, v in data.items()},
                      class_share_test=np.bincount(yte, minlength=3).tolist())
(OUT / "baselines.json").write_text(json.dumps(results, indent=1))
