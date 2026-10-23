"""Train DeepLOB on FI-2010 for one horizon and one hyperparameter config.

    python scripts/02_train_fi2010.py --k 10 --config B

Writes results/fi2010/deeplob_k{k}_{config}.json (history, validation and test scores)
and models/fi2010/deeplob_k{k}_{config}.pt. Run by the train-fi2010 workflow, one job
per (k, config). The config is chosen later on validation macro F1 only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deeplob import fi2010 as F  # noqa: E402
from deeplob.train import TrainConfig, Windows, fit, predict, scores  # noqa: E402

# The three configurations tried (Subh: "try 2-3, keep the best").
CONFIGS = {
    "A": TrainConfig(name="A", lr=1e-4, batch=64),               # authors' notebook
    "B": TrainConfig(name="B", lr=1e-3, batch=128),              # 10x lr, bigger batch
    "C": TrainConfig(name="C", lr=1e-2, adam_eps=1.0, batch=32),  # paper: lr 0.01, eps 1
}

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, required=True, choices=F.HORIZONS)
ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
ap.add_argument("--max-epochs", type=int)
ap.add_argument("--time-budget-min", type=float)
a = ap.parse_args()
cfg = CONFIGS[a.config]
if a.max_epochs:
    cfg.max_epochs = a.max_epochs
if a.time_budget_min:
    cfg.time_budget_min = a.time_budget_min
torch.set_num_threads(max(1, torch.get_num_threads()))

data = ROOT / "data" / "fi2010"
raw = F.load_raw(data, "train")
tr_seg, va_seg = F.train_val_segments(raw)
tr, va = F.make_split(raw, a.k, segments=tr_seg), F.make_split(raw, a.k, segments=va_seg)
model, report = fit(cfg, Windows(tr.lob, tr.y, tr.ends), Windows(va.lob, va.y, va.ends))

# Final scores with the best-validation weights: full validation set and each test day.
report["val"] = scores(va.y[va.ends], predict(model, Windows(va.lob, va.y, va.ends)))
ys, ps = [], []
for day in ("test8", "test9", "test10"):
    r = F.load_raw(data, day)
    s = F.make_split(r, a.k)
    prob = predict(model, Windows(s.lob, s.y, s.ends))
    report[day] = scores(s.y[s.ends], prob)
    ys.append(s.y[s.ends])
    ps.append(prob)
report["test"] = scores(np.concatenate(ys), np.concatenate(ps))
report["k"] = a.k
print(json.dumps({k: report[k] for k in ("best_epoch", "best_val_macro_f1", "val", "test")}, indent=1))

(ROOT / "results" / "fi2010").mkdir(parents=True, exist_ok=True)
(ROOT / "models" / "fi2010").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "fi2010" / f"deeplob_k{a.k}_{a.config}.json").write_text(json.dumps(report, indent=1))
torch.save(model.state_dict(), ROOT / "models" / "fi2010" / f"deeplob_k{a.k}_{a.config}.pt")
