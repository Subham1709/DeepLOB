"""Logistic regression and LightGBM baselines on FI-2010, three configs each.

Config chosen on validation macro F1 (same per-stock time split as DeepLOB), test scored
with that config. Writes results/fi2010/baselines.json.
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

from deeplob import fi2010 as F  # noqa: E402
from deeplob.features import LAGS, book_features  # noqa: E402
from deeplob.train import scores  # noqa: E402

DATA = ROOT / "data" / "fi2010"
LR_C = (0.01, 0.1, 1.0)
LGB = {"L15": dict(num_leaves=15), "L31": dict(num_leaves=31), "L63": dict(num_leaves=63)}


def onehot_prob(p):
    return np.eye(3)[p]


raw = F.load_raw(DATA, "train")
tr_seg, va_seg = F.train_val_segments(raw)
tests = {d: F.load_raw(DATA, d) for d in ("test8", "test9", "test10")}
out = {}
for k in (10, 20, 50):
    def xy(r, seg=None):
        s = F.make_split(r, k, segments=seg)
        X, names = book_features(s.lob, s.ends)
        return X, s.y[s.ends], names
    Xtr, ytr, names = xy(raw, tr_seg)
    Xva, yva, _ = xy(raw, va_seg)
    te = [xy(r) for r in tests.values()]
    Xte, yte = np.vstack([t[0] for t in te]), np.concatenate([t[1] for t in te])
    res = {}
    past = slice(0, len(LAGS))
    for feat_name, cols in (("past_returns", past), ("all", slice(None))):
        for C in LR_C:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
            m.fit(Xtr[:, cols], ytr)
            res[f"logreg_{feat_name}_C{C}"] = dict(
                val=scores(yva, m.predict_proba(Xva[:, cols])), test=scores(yte, m.predict_proba(Xte[:, cols])))
            print(k, f"logreg_{feat_name}_C{C}", round(res[f"logreg_{feat_name}_C{C}"]["val"]["macro_f1"], 4), flush=True)
    for name, extra in LGB.items():
        m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, verbose=-1, random_state=0, **extra)
        m.fit(Xtr, ytr, eval_X=(Xva,), eval_y=(yva,), callbacks=[lgb.early_stopping(50, verbose=False)])
        res[f"lightgbm_{name}"] = dict(val=scores(yva, m.predict_proba(Xva)), test=scores(yte, m.predict_proba(Xte)),
                                       best_iteration=int(m.best_iteration_ or 500))
        print(k, f"lightgbm_{name}", round(res[f"lightgbm_{name}"]["val"]["macro_f1"], 4), flush=True)
    out[k] = res

(ROOT / "results" / "fi2010").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "fi2010" / "baselines.json").write_text(json.dumps(out, indent=1))
