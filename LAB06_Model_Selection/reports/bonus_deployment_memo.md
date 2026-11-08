# Task 5D — Export & CPU benchmark: execution status
**Prepared 9 November 2026.** **NOT YET EXECUTED** in the current environment.

## Scope and actual progress
- Model to export: our **original YOLOv8n-cls** (not the classmate checkpoint, and not the Task 5C modified weights).
- Original saved artifact: [Google Drive original best.pt](https://drive.google.com/file/d/1htkgf8oiox9QT-s-JWnvxJxRKJTpPNiL/view), documented size **2.83 MiB**.
- Earlier native benchmark: **26.97 ms/image**, test accuracy **267/270 (98.89%)**. **These are historical values, not Task 5D measurements.**
- New ONNX export artifact: **not yet generated**.
- New CPU 20-warmup/100-run before/after measurements: **not yet performed**.
- ONNX accuracy, size, latency and accuracy drop: **unknown** until execution.
- This entry is a reproducible experiment implementation and truthful tracking record, **not a completed benchmark**.

## Exact run
Open [Task5D_ONNX_Export_Benchmark.ipynb](../Task5D_ONNX_Export_Benchmark.ipynb) in Colab, run all cells with the existing Google Drive mount, and retain output directory
\`MyDrive/Lab06_results/Task5D_2026-11-09/\`.

Alternatively install \`ultralytics onnx onnxruntime\` and run:

\`\`\`bash
python scripts/task5d_export_onnx_benchmark.py \
  --checkpoint /path/to/Lab06_results/yolov8n/weights/best.pt \
  --test /path/to/Lab06_results/yolo_data/test \
  --manifest /path/to/Lab06_results/manifest.json \
  --out /path/to/NEW_Task5D_results
\`\`\`

## Quality gates implemented
1. Export from a staged copy (original checkpoint unaffected).
2. ONNX checker validates structure; source/export SHA-256 and size recorded.
3. Verify 270 original held-out test images and 135 good / 135 bad; verify SHA-256 if manifest supplied.
4. Evaluate identical 224x224 preprocessed tensors before/after, fixed P(bad)>=0.50.
5. Compare accuracy, macro-F1, per-class F1 and prediction disagreements; record probability differences.
6. Measure **20 warm-up + 100 timed** CPU **forward-only** calls on each backend using ten preprocessed cached images, four threads.
7. Auto-generate JSON, CSV prediction table, speed/size table and English memo.
8. Check whether native result reproduces original 267/270. Flag mismatches without hiding them.

**Limitations:** The benchmark excludes preprocessing/file IO; thus its latencies **cannot be directly compared** with archived 26.97ms end-to-end-preprocessed CPU latency. It is not a real edge-device benchmark. No quantization has been conducted, and quantization risk is discussed, not measured.

## Instructor compliance
Still pending: **actual ONNX bytes**, numeric export-vs-source benchmark, accuracy drop, and measured deployment conclusion. Do not mark Task 5D complete until these are produced and independently verified.
