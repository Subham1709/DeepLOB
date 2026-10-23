"""Collect FI-2010 results into results/fi2010_summary.md.

For each model family the config is picked on validation macro F1 only; the test score of
that config is the reported number. Test scores of the other configs are shown for
transparency but never used for the choice.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "fi2010"
PAPER = {10: 0.8340, 20: 0.7282, 50: 0.8035}
DESC = {"A": "lr 1e-4, batch 64 (authors' notebook)", "B": "lr 1e-3, batch 128",
        "C": "lr 1e-2, Adam eps 1, batch 32 (paper)", "D": "as B, inputs z-scored per stock"}

base = json.loads((R / "baselines.json").read_text())
out = ["# FI-2010 results", "",
       "Train: days 1-7 (last 20% of each stock held out for validation). Test: days 8-10.",
       "Metric: macro F1 (mean of the down / flat / up F1 scores). Config chosen on validation.", ""]

out += ["## DeepLOB: the configs tried", "",
        "| k | config | epochs run (best) | val macro F1 | test macro F1 | test accuracy |",
        "|---|---|---|---|---|---|"]
chosen = {}
for k in (10, 20, 50):
    runs = {c: json.loads(p.read_text()) for c in "ABCD" if (p := R / f"deeplob_k{k}_{c}.json").exists()}
    best = max(runs, key=lambda c: runs[c]["val"]["macro_f1"])
    chosen[k] = (best, runs[best])
    for c, r in runs.items():
        mark = " **(chosen)**" if c == best else ""
        out.append(f"| {k} | {c}: {DESC[c]}{mark} | {len(r['history'])} ({r['best_epoch']}) | "
                   f"{r['val']['macro_f1']:.3f} | {r['test']['macro_f1']:.3f} | {r['test']['accuracy']:.3f} |")


def pick(res, prefix):
    c = {n: v for n, v in res.items() if n.startswith(prefix)}
    n = max(c, key=lambda n: c[n]["val"]["macro_f1"])
    return n, c[n]


out += ["", "## Comparison (test macro F1, each with its validation-chosen config)", "",
        "| model | k = 10 | k = 20 | k = 50 |", "|---|---|---|---|"]
rows = {"Logistic regression, 8 past returns only": "logreg_past_returns",
        "Logistic regression, 53 book features": "logreg_all",
        "LightGBM, 53 book features": "lightgbm"}
for label, prefix in rows.items():
    cells = []
    for k in (10, 20, 50):
        n, v = pick(base[str(k)], prefix)
        cells.append(f"{v['test']['macro_f1']:.3f}")
    out.append(f"| {label} | " + " | ".join(cells) + " |")
out.append("| DeepLOB, ours | " + " | ".join(f"{chosen[k][1]['test']['macro_f1']:.3f} ({chosen[k][0]})"
                                          for k in (10, 20, 50)) + " |")
out.append("| DeepLOB, paper (Zhang et al. 2019) | " + " | ".join(f"{PAPER[k]:.3f}" for k in (10, 20, 50)) + " |")

(ROOT / "results" / "fi2010_summary.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
