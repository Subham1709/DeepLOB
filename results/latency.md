# Inference latency (batch 1, one CPU thread)

Machine: x86_64, Linux; torch 2.14.1+cu130, onnxruntime 1.29.0; 2000 calls each.

| engine | median (us) | p99 (us) |
|---|---|---|
| PyTorch eager | 2714 | 4491 |
| ONNX Runtime | 807 | 930 |

Max |difference| in logits between the two: 3.6e-07.
ONNX file size: 580 KB.
