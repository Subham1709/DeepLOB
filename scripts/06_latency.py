"""Export DeepLOB to ONNX and measure single-prediction latency.

    python scripts/06_latency.py [--weights models/fi2010/deeplob_k10_A.pt]

Latency doesn't depend on the weights, so random weights work too. Measures batch-1
inference (one new prediction per book update) on one CPU thread, PyTorch eager vs ONNX
Runtime, and checks the two give the same outputs. Writes results/latency.md.
"""
import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deeplob.model import DeepLOB  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--weights", type=Path)
ap.add_argument("--n", type=int, default=2000)
a = ap.parse_args()

torch.set_num_threads(1)
model = DeepLOB().eval()
if a.weights:
    model.load_state_dict(torch.load(a.weights, map_location="cpu"))
x = torch.randn(1, 100, 40)
path = ROOT / "models" / "deeplob.onnx"
path.parent.mkdir(exist_ok=True)
torch.onnx.export(model, (x,), path, input_names=["book"], output_names=["logits"],
                  dynamic_axes={"book": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)

import onnxruntime as ort  # noqa: E402

so = ort.SessionOptions()
so.intra_op_num_threads = so.inter_op_num_threads = 1
sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def timeit(f, n):
    for _ in range(50):
        f()
    ts = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter_ns()
        f()
        ts[i] = (time.perf_counter_ns() - t0) / 1e3
    return ts


xs = x.numpy()
with torch.inference_mode():
    t_torch = timeit(lambda: model(x), a.n)
    ref = model(x).numpy()
t_onnx = timeit(lambda: sess.run(None, {"book": xs}), a.n)
err = float(np.abs(sess.run(None, {"book": xs})[0] - ref).max())

rows = ["# Inference latency (batch 1, one CPU thread)", "",
        f"Machine: {platform.processor() or platform.machine()}, {platform.system()}; "
        f"torch {torch.__version__}, onnxruntime {ort.__version__}; {a.n} calls each.", "",
        "| engine | median (us) | p99 (us) |", "|---|---|---|",
        f"| PyTorch eager | {np.median(t_torch):.0f} | {np.percentile(t_torch, 99):.0f} |",
        f"| ONNX Runtime | {np.median(t_onnx):.0f} | {np.percentile(t_onnx, 99):.0f} |", "",
        f"Max |difference| in logits between the two: {err:.1e}.",
        f"ONNX file size: {path.stat().st_size / 1024:.0f} KB."]
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "latency.md").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
