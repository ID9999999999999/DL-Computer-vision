# Task 5D — ONNX export and CPU benchmark
**9 November 2026. Status: ONNX export and forward-only CPU benchmark EXECUTED; original held-out Test accuracy parity pending.**

## Real execution and integrity
- Original YOLOv8n-cls checkpoint SHA-256: c820e22debf8d3fe3c3ca41f7c10f55d9e7db0b2246fb90c92a8548a70a514f7.
- Exported ONNX SHA-256: cf8f353758ab9b6562758537482b6ae4f898e3450cd41071a56b9133298dbf45.
- [Successful GitHub Actions run and logs](https://github.com/ID9999999999999/DL-Computer-vision/actions/runs/42515991510).
- [ONNX model and raw benchmark JSON in a ZIP on Google Drive](https://drive.google.com/file/d/1xdHClyFQhvANfbYA2fs6r365yLawbvgR/view).
- [Automated source script](../scripts/task5d_ci_export_smoke.py).

## Measured paired CPU results
| Measurement | Original PyTorch | ONNX |
|---|---:|---:|
| Bytes | 2,963,154 | 5,767,752 |
| MiB | 2.826 | 5.501 |
| Mean forward-only ms | 8.651 | 2.965 |
| Warm-up calls | 20 | 20 |
| Timed calls | 100 | 100 |
| Threads | 4 | 4 |

ONNX was approximately **2.92x faster** for forward-only CPU inference on the same cloud machine, but is approximately **1.95x bigger** on disk. The export passed ONNX checker. Twelve identical synthetic 1x3x224x224 tensors had zero predicted-class disagreements and maximum absolute probability difference 3.5763e-7. These synthetic-tensor results are **not a document-quality accuracy metric**.

## What is still pending
- The original 270-image held-out Test is private and was intentionally not uploaded or shared with the cloud CI runner.
- **Before/after Test accuracy, Macro-F1, confusion matrices, and measured accuracy drop have not been produced. Do not report a zero drop.**
- Historical source accuracy was 267/270 = 98.89%; this is from earlier tests, not a paired ONNX validation.
- Existing [full-test Colab notebook](../Task5D_ONNX_Export_Benchmark.ipynb) and [paired Test benchmark](../scripts/task5d_export_onnx_benchmark.py) use the original private Test with a fixed 0.5 threshold. They have not been executed on the 270 images as part of Task 5D.
- Benchmark is forward-only excluding disk IO and preprocessing and cannot be directly compared with the previous 26.97 ms pipeline timing. GitHub Actions CPU is not a real edge device.
- No INT8/FP16 quantization. Such quantization can change confidence and decisions; it would require training/validation calibration and a fresh accuracy check.

## Deployment decision
ONNX Runtime is a plausible CPU-server candidate because forward-only CPU latency was lower in this paired experiment, but the larger FP32 ONNX storage and unverified held-out accuracy preclude claiming full deployment readiness or edge suitability. **Task 5D remains partially completed pending original Test accuracy parity.**
