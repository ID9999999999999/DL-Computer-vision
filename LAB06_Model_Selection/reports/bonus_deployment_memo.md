# Task 5D — ONNX Export, Paired Test Accuracy and CPU Benchmark
**Executed 9 November 2026 on the original private 270-image Test; Task 5D technically completed, instructor approval not confirmed.**

## Real reproducible measurements
- Model: original YOLOv8n-cls pretrained/fine-tuned checkpoint (original SHA-256 `c820e22debf8d3fe3c3ca41f7c10f55d9e7db0b2246fb90c92a8548a70a514f7`).
- ONNX exported model SHA-256: `cf8f353758ab9b6562758537482b6ae4f898e3450cd41071a56b9133298dbf45`; structural ONNX check passed.
- Test: 270 source-hash-verified original images, 135 good / 135 bad, 45 underlying document sources. No image sharing to public CI, no retraining, no threshold tuning; fixed P(bad) >= 0.50.
- Native and exported runtimes processed **the same normalized and resized 224x224 test tensors**.

| Measure | Native PyTorch | Exported ONNX |
|---|---:|---:|
| Accuracy | **267/270 = 98.89%** | **267/270 = 98.89%** |
| Macro-F1 | **0.988888** | **0.988888** |
| Bad precision | 1.0000 | 1.0000 |
| Bad recall | 0.97778 | 0.97778 |
| Confusion matrix (true rows good/bad, pred columns good/bad) | [[135,0],[3,132]] | [[135,0],[3,132]] |
| Checkpoint size | 2.826 MiB | 5.501 MiB |
| CPU mean forward latency | **6.104 ms** | **2.451 ms** |

- **Measured Accuracy drop = 0.0000; Macro-F1 drop = 0.0000; 0/270 differing class predictions.**
- Maximum absolute P(bad) numerical difference: **3.5492×10⁻⁶** across the 270 Test images.
- CPU benchmark on same private local host, 4 CPU threads, batch 1, **20 warm-up + 100 timed calls each**, 10 cached originally preprocessed Test images. Native-to-ONNX mean-forward ratio: **2.49×**.
- Forward-only timing **excludes** preprocessing, file I/O, model loading and GPU/edge deployment. The prior 26.97ms historic CPU benchmark covered a different scope and must not be compared as if identical.
- Earlier cloud GitHub Actions synthetic-input smoke run (8.651ms/2.965ms) used a different CPU runner; it is archived separately and is not the paired original-Test run.

## Deployment memo
ONNX Runtime is a **candidate for a CPU server**: all held-out test classifications agreed with the PyTorch original, and mean forward-only inference was lower on the measured host. The ONNX FP32 export is approximately 1.95x larger, not smaller. Production/edge suitability remains conditional on an end-to-end pipeline measurement, real mobile images and target-device profiling.
INT8/FP16 quantization was not applied (optional). Quantization could change confidences and class decisions; calibrate using train/validation and recheck independent held-out accuracy.

## Artifacts and reproducibility
- [Full verified package: actual ONNX, all 270 paired predictions, raw JSON with 200 timing samples, summary CSV and exact executed source script](https://drive.google.com/file/d/1zIEXnesFzi4JPrzxuNRbYaemqEUEbQ0w/view).
- [Original successfully exported ONNX and cloud CPU-run evidence](https://drive.google.com/file/d/1xdHClyFQhvANfbYA2fs6r365yLawbvgR/view).
- [Task 5D per-requirement audit](../docs/TASK5D_ONNX_2026-11-09_AUDIT.md).
- Private original dataset and checkpoint are preserved, not moved, altered or published.

**Scope of success: complete Task5D experimental deliverables, not an instructor's acceptance record.**
