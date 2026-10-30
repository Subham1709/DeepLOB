"""Train DeepLOB on Binance BTCUSDT + ETHUSDT for one horizon and one config.

    python scripts/05_train_binance.py --h 300 --config B

Same chronological split, labels and class balance as scripts/05_baselines_binance.py. One
model per horizon on both symbols (inputs are relative to each window's own mid, so the two
look alike). Config chosen later on validation macro F1. Writes
results/binance/deeplob_h{h}_{config}.json, test probabilities
results/binance/pred_deeplob_{config}_h{h}_{symbol}.npz and models/binance/*.pt.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deeplob import binance as B  # noqa: E402
from deeplob.train import TrainConfig, fit, predict, scores  # noqa: E402

# Snapshots, 100 ms each. With a week of data, every 6th second of training windows is
# ~130k windows a pass (~10 min on a runner); one test prediction per second.
TRAIN_STRIDE, VAL_STRIDE, TEST_STRIDE = 60, 60, 10
CONFIGS = {
    "B": TrainConfig(name="B", lr=1e-3, batch=128),                          # best on FI-2010
    "C": TrainConfig(name="C", lr=1e-2, adam_eps=1.0, batch=32),             # paper
    "E": TrainConfig(name="E", lr=3e-4, batch=128, weight_decay=1e-4),       # slower + weight decay (FI-2010 overfit)
}
for c in CONFIGS.values():
    c.train_stride, c.val_stride, c.max_epochs, c.patience = 1, 1, 30, 5

ap = argparse.ArgumentParser()
ap.add_argument("--h", type=int, required=True)
ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
ap.add_argument("--max-epochs", type=int)
a = ap.parse_args()
cfg = CONFIGS[a.config]
if a.max_epochs:
    cfg.max_epochs = a.max_epochs

preps = [B.prepare(ROOT / "data" / "binance", s, a.h) for s in B.SYMBOLS]
train = torch.utils.data.ConcatDataset([p.windows("train", TRAIN_STRIDE) for p in preps])
val = torch.utils.data.ConcatDataset([p.windows("val", VAL_STRIDE) for p in preps])
y_val = np.concatenate([p.y[p.ends["val"][::VAL_STRIDE]] for p in preps])
model, report = fit(cfg, train, val, y_val=y_val)

report["val"] = scores(y_val, predict(model, val))
ys, ps = [], []
out = ROOT / "results" / "binance"
out.mkdir(parents=True, exist_ok=True)
report["test_by_symbol"] = {}
for p in preps:
    e = p.ends["test"][::TEST_STRIDE]
    prob = predict(model, p.windows("test", TEST_STRIDE))
    report["test_by_symbol"][p.symbol] = scores(p.y[e], prob)
    np.savez_compressed(out / f"pred_deeplob_{a.config}_h{a.h}_{p.symbol}.npz", ends=e, prob=prob.astype(np.float32))
    ys.append(p.y[e])
    ps.append(prob)
report["test"] = scores(np.concatenate(ys), np.concatenate(ps))
report.update(h=a.h, alpha_bps={p.symbol: p.alpha * 1e4 for p in preps},
              n_train=len(train), n_val=len(val), data_end_ms=int(preps[0].book.t[-1]))
print(json.dumps({k: report[k] for k in ("best_epoch", "best_val_macro_f1", "test")}, indent=1))
(out / f"deeplob_h{a.h}_{a.config}.json").write_text(json.dumps(report, indent=1))
(ROOT / "models" / "binance").mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), ROOT / "models" / "binance" / f"deeplob_h{a.h}_{a.config}.pt")
