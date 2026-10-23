"""Training loop for DeepLOB on any (book array, labels, window-end indices) data.

Windows are sliced from the stored book on the fly (see fi2010.windows), so memory stays at
one copy of the book. Model selection uses validation macro F1, never the test set.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .model import DeepLOB


@dataclass
class TrainConfig:
    name: str = "A"
    lr: float = 1e-4
    adam_eps: float = 1e-8
    batch: int = 64
    max_epochs: int = 50
    patience: int = 10
    train_stride: int = 4      # use every n-th training window (neighbours overlap 99%)
    val_stride: int = 2
    time_budget_min: float = 320  # stop before the 6 h Actions job limit
    seed: int = 0


class Windows(torch.utils.data.Dataset):
    def __init__(self, lob: np.ndarray, y: np.ndarray, ends: np.ndarray, T: int = 100):
        self.lob = torch.from_numpy(np.ascontiguousarray(lob, dtype=np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.ends = ends
        self.T = T

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, i):
        t = int(self.ends[i])
        return self.lob[t - self.T + 1:t + 1], self.y[t]


@torch.no_grad()
def predict(model: torch.nn.Module, ds: Windows, batch: int = 1024) -> np.ndarray:
    """Class probabilities, (n, 3)."""
    model.eval()
    out = []
    for x, _ in torch.utils.data.DataLoader(ds, batch_size=batch):
        out.append(torch.softmax(model(x), 1).numpy())
    return np.concatenate(out)


def scores(y: np.ndarray, prob: np.ndarray) -> dict:
    p = prob.argmax(1)
    return dict(accuracy=float(accuracy_score(y, p)), macro_f1=float(f1_score(y, p, average="macro")),
                f1_per_class=[float(v) for v in f1_score(y, p, average=None, labels=[0, 1, 2])])


def fit(cfg: TrainConfig, train: Windows, val: Windows, log=print):
    """Adam + cross-entropy; keep the weights with the best validation macro F1."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = DeepLOB()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
    sub_tr = torch.utils.data.Subset(train, np.arange(0, len(train), cfg.train_stride))
    sub_va = torch.utils.data.Subset(val, np.arange(0, len(val), cfg.val_stride))
    y_va = val.y[val.ends[::cfg.val_stride]].numpy()
    loader = torch.utils.data.DataLoader(sub_tr, batch_size=cfg.batch, shuffle=True,
                                         generator=torch.Generator().manual_seed(cfg.seed))
    best, best_state, history, bad = -1.0, None, [], 0
    t0 = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        losses = []
        for x, y in loader:
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        prob = predict(model, sub_va)
        val_loss = float(-np.mean(np.log(prob[np.arange(len(y_va)), y_va] + 1e-12)))
        s = scores(y_va, prob)
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), val_loss=val_loss,
                            val_macro_f1=s["macro_f1"], val_accuracy=s["accuracy"],
                            minutes=(time.time() - t0) / 60))
        log(json.dumps(history[-1]))
        if s["macro_f1"] > best:
            best, bad = s["macro_f1"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= cfg.patience or (time.time() - t0) / 60 > cfg.time_budget_min:
            break
    model.load_state_dict(best_state)
    return model, dict(config=asdict(cfg), history=history, best_val_macro_f1=best,
                       best_epoch=int(np.argmax([h["val_macro_f1"] for h in history]) + 1))
