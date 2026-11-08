"""Lab06 Task 5D: honest paired PyTorch -> ONNX CPU export benchmark.

Does not train or modify the original checkpoint or dataset.
Requires the original held-out Test as test/{good,bad}/*.jpg.
Run on the same CPU with the same preprocessed images for both backends.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import time

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prob_vector(raw: object) -> np.ndarray:
    """Return two normalized class probabilities, not arbitrary logits."""
    import torch
    if isinstance(raw, (tuple, list)):
        candidates = [x for x in raw if (isinstance(x, (np.ndarray, torch.Tensor))
                      and getattr(x, "ndim", 0) == 2 and x.shape[-1] == 2)]
        if not candidates:
            raise ValueError(f"Cannot identify two-class output from {[getattr(x, 'shape', None) for x in raw]}")
        raw = candidates[0]
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().numpy()
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError(f"Expected two finite output scores, got {values}")
    if (values >= 0).all() and abs(float(values.sum()) - 1.0) < 1e-4:
        return values / values.sum()
    values = values - values.max()
    e = np.exp(values)
    return e / e.sum()


def metric_summary(actual: list[int], predicted: list[int]) -> dict:
    # Labels are binary: good=0, bad=1. Matrix rows true, columns predicted.
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Invalid prediction lengths")
    cm = [[0, 0], [0, 0]]
    for a, p in zip(actual, predicted):
        cm[a][p] += 1
    f1 = {}
    precision = {}
    recall = {}
    for cls, name in ((0, "good"), (1, "bad")):
        tp = cm[cls][cls]
        fp = cm[1-cls][cls]
        fn = cm[cls][1-cls]
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        precision[name], recall[name] = pr, rc
        f1[name] = 2*pr*rc/(pr+rc) if pr+rc else 0.0
    return {
        "n": len(actual), "correct": cm[0][0] + cm[1][1],
        "accuracy": (cm[0][0]+cm[1][1])/len(actual),
        "macro_f1": (f1["good"]+f1["bad"])/2,
        "precision": precision, "recall": recall, "f1": f1,
        "confusion_matrix_good_bad": cm
    }


def ensure_test(test: Path, manifest: Path | None) -> list[tuple[Path, int]]:
    records = []
    for cls_name, positive in (("good", 0), ("bad", 1)):
        root = test/cls_name
        if not root.is_dir():
            raise FileNotFoundError(f"Missing folder {root}")
        found = sorted(p for p in root.iterdir() if p.is_file()
                       and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
        if len(found) != 135:
            raise ValueError(f"{cls_name}: expected 135 ORIGINAL held-out images, got {len(found)}")
        records.extend((path, positive) for path in found)
    if manifest is not None:
        rows = json.loads(manifest.read_text(encoding="utf-8"))
        expected = {(r["quality_label"], Path(r["image_path"]).name): r
                    for r in rows if r["split"] == "test"}
        if len(expected) != 270:
            raise ValueError(f"Manifest test count invalid: {len(expected)}")
        for path, y in records:
            label = "bad" if y else "good"
            meta = expected.get((label, path.name))
            if not meta:
                raise ValueError(f"Not an original test image: {label}/{path.name}")
            if sha256(path) != meta["sha256"]:
                raise ValueError(f"Original image SHA-256 mismatch: {path}")
    return records


def benchmark(fun, inputs: list, warmups: int = 20, runs: int = 100) -> dict:
    """Forward pass only. Inputs are preprocessed/cached; file IO excluded."""
    for i in range(warmups):
        fun(inputs[i % len(inputs)])
    times = []
    for i in range(runs):
        t = time.perf_counter()
        fun(inputs[i % len(inputs)])
        times.append((time.perf_counter() - t)*1000)
    return {
        "mean_ms": float(statistics.mean(times)),
        "median_ms": float(statistics.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "warmup_runs": warmups, "timed_runs": runs,
        "latencies_ms": times,
        "scope": "forward-only, one 224x224 image, CPU, preprocessed tensor; excludes preprocess, disk IO, session initialization"
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True, help="Original YOLOv8n best.pt")
    ap.add_argument("--test", type=Path, required=True, help="Original held-out yolo_data/test")
    ap.add_argument("--manifest", type=Path, help="Original source-grouped results/manifest.json")
    ap.add_argument("--out", type=Path, required=True, help="NEW output directory")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    if args.threads < 1:
        ap.error("--threads must be positive")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    import onnx
    import onnxruntime as ort
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.data.augment import classify_transforms

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {args.out}; use a fresh folder")
    records = ensure_test(args.test, args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)

    # Export a staged copy so Ultralytics never writes into the source folder.
    staged_pt = args.out/"source_best.pt"
    shutil.copy2(checkpoint, staged_pt)
    source_sha = sha256(checkpoint)
    original = YOLO(str(staged_pt))
    names = {int(k): str(v).lower() for k, v in original.names.items()}
    if sorted(names.values()) != ["bad", "good"] or len(names) != 2:
        raise ValueError(f"Expected original binary bad/good checkpoint, got {names}")
    bad_idx = next(i for i, name in names.items() if name == "bad")

    original.model.cpu().eval()
    exported = Path(original.export(format="onnx", imgsz=224, batch=1,
                                    device="cpu", dynamic=False, simplify=False))
    if not exported.is_file():
        raise FileNotFoundError(f"ONNX export did not create a file: {exported}")
    onnx.checker.check_model(str(exported))
    exported_sha = sha256(exported)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(exported), sess_options=opts,
                                   providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(f"Unexpected provider(s): {session.get_providers()}")
    inp = session.get_inputs()[0]
    if len(session.get_inputs()) != 1:
        raise ValueError("Expected one ONNX input")
    transforms = classify_transforms(size=224)

    true_labels: list[int] = []
    torch_predictions: list[int] = []
    onnx_predictions: list[int] = []
    probabilities_delta = []
    csv_rows = []
    cached = []

    with torch.inference_mode():
        for path, y in records:
            with Image.open(path) as im:
                x = transforms(im.convert("RGB")).unsqueeze(0).to(dtype=torch.float32)
            xnp = x.numpy()
            # Evaluate the identical tensor with both inference engines.
            native_out = original.model(x)
            ort_out = session.run(None, {inp.name: xnp})[0]
            native_prob = prob_vector(native_out)
            onnx_prob = prob_vector(ort_out)
            p1 = float(native_prob[bad_idx])
            p2 = float(onnx_prob[bad_idx])
            native_y = int(p1 >= 0.5)
            onnx_y = int(p2 >= 0.5)
            true_labels.append(y)
            torch_predictions.append(native_y)
            onnx_predictions.append(onnx_y)
            probabilities_delta.append(abs(p1-p2))
            csv_rows.append([path.parent.name, path.name, y, p1, native_y,
                             p2, onnx_y, abs(p1-p2)])
            if len(cached) < 10:
                cached.append((x, xnp))

        native_bench = benchmark(lambda item: original.model(item[0]), cached)
    ort_bench = benchmark(lambda item: session.run(None, {inp.name: item[1]}), cached)

    native_metrics = metric_summary(true_labels, torch_predictions)
    onnx_metrics = metric_summary(true_labels, onnx_predictions)
    disagree = sum(a != b for a, b in zip(torch_predictions, onnx_predictions))

    result = {
        "task": "5D", "date": "2026-11-09",
        "evaluation_split": "original held-out Test, no tuning",
        "model": "YOLOv8n-cls", "classes": names, "bad_class_index": bad_idx,
        "threshold": 0.5, "image_size": 224,
        "preprocessing": "ultralytics.data.augment.classify_transforms(size=224), identical input tensor",
        "input_count": len(records), "manifest_sha256": sha256(args.manifest) if args.manifest else None,
        "original_pt": {
            "path": str(checkpoint), "sha256": source_sha,
            "size_bytes": checkpoint.stat().st_size,
            "metrics": native_metrics, "benchmark": native_bench
        },
        "exported_onnx": {
            "path": str(exported), "sha256": exported_sha,
            "size_bytes": exported.stat().st_size,
            "metrics": onnx_metrics, "benchmark": ort_bench
        },
        "parity": {
            "prediction_disagreements": disagree,
            "maximum_abs_delta_p_bad": max(probabilities_delta),
            "mean_abs_delta_p_bad": statistics.mean(probabilities_delta),
            "accuracy_drop_original_minus_onnx": native_metrics["accuracy"]-onnx_metrics["accuracy"],
            "macro_f1_drop_original_minus_onnx": native_metrics["macro_f1"]-onnx_metrics["macro_f1"]
        },
        "baseline_archive_check": {
            "expected_correct_267_of_270": native_metrics["correct"] == 267,
            "note": "If false, investigate preprocessing/package differences before comparison with historical results."
        },
        "environment": {
            "platform": platform.platform(), "cpu": platform.processor(),
            "threads": args.threads, "torch": torch.__version__,
            "onnx": onnx.__version__, "onnxruntime": ort.__version__,
            "ultralytics": ultralytics.__version__
        },
        "limitations": [
            "CPU-only forward path; not an edge-device measurement.",
            "Historical 26.97 ms included preprocessing, so not directly comparable to forward-only results.",
            "No quantization was performed.",
            "Held-out test contains 270 images derived from 45 source documents."
        ]
    }

    (args.out/"results_5d.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.out/"predictions_before_after.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class","filename","y_true_bad","p_bad_pt","pred_bad_pt",
                         "p_bad_onnx","pred_bad_onnx","absolute_delta_p_bad"])
        writer.writerows(csv_rows)
    with (args.out/"benchmark_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["format","bytes","accuracy","macro_f1","mean_ms","median_ms","p95_ms",
                         "warmups","timed_runs","threads"])
        for name, row in (("PyTorch", result["original_pt"]), ("ONNX", result["exported_onnx"])):
            b = row["benchmark"]
            writer.writerow([name,row["size_bytes"],row["metrics"]["accuracy"],
                             row["metrics"]["macro_f1"],b["mean_ms"],b["median_ms"],
                             b["p95_ms"],b["warmup_runs"],b["timed_runs"],args.threads])

    note = f"""# Lab 06 Bonus Task 5D — ONNX Export and CPU Benchmark
**Status:** measured, generated directly by the benchmark script. Source: original saved YOLOv8n-cls checkpoint.

## Same-Test accuracy (n={len(records)}; threshold P(bad) >= 0.50)
| Engine | Accuracy | Macro-F1 | Size MiB | CPU forward mean ms | CPU forward p95 ms |
|---|---:|---:|---:|---:|---:|
| Original PyTorch | {native_metrics['accuracy']:.4f} | {native_metrics['macro_f1']:.4f} | {checkpoint.stat().st_size/2**20:.3f} | {native_bench['mean_ms']:.3f} | {native_bench['p95_ms']:.3f} |
| Exported ONNX | {onnx_metrics['accuracy']:.4f} | {onnx_metrics['macro_f1']:.4f} | {exported.stat().st_size/2**20:.3f} | {ort_bench['mean_ms']:.3f} | {ort_bench['p95_ms']:.3f} |

- Accuracy drop (PyTorch minus ONNX): {native_metrics['accuracy'] - onnx_metrics['accuracy']:+.4f}.
- Prediction disagreements: {disagree}/{len(records)}.
- Maximum absolute P(bad) difference: {max(probabilities_delta):.6g}.
- Historical native model accuracy consistency (expected 267/270): **{native_metrics['correct']==267}**.
- Export validated using ONNX checker; SHA256 of both files recorded.
- Benchmark: **20 warm-ups + 100 timed runs**, same CPU, four threads (or configured count), batch size 1, 10 cached preprocessed inputs, forward inference only.

## Deployment decision
ONNX is a **potential CPU-server deployment option**, subject to the parity and measured latency above. Edge-device suitability is NOT established by a CPU-only benchmark: an actual target device must be profiled. The originally reported **26.97 ms** benchmark used a different execution scope and is **not** an apples-to-apples speedup denominator. Real camera data must be checked before deployment.

## Quantization
No INT8 or FP16 quantization has been performed. Quantization can change rounding, confidence scores and sometimes class labels, especially near decision thresholds. Any quantized model needs independent calibration on training/validation data and fresh accuracy checks; never use held-out Test to tune quantization parameters.
"""
    (args.out/"bonus_deployment_memo.md").write_text(note, encoding="utf-8")
    print(json.dumps({
        "source": str(checkpoint), "exported": str(exported),
        "original_accuracy": native_metrics["accuracy"],
        "onnx_accuracy": onnx_metrics["accuracy"],
        "prediction_disagreements": disagree,
        "native_mean_forward_ms": native_bench["mean_ms"],
        "onnx_mean_forward_ms": ort_bench["mean_ms"],
        "original_size_mib": checkpoint.stat().st_size/2**20,
        "onnx_size_mib": exported.stat().st_size/2**20,
        "output_directory": str(args.out)
    }, indent=2))


if __name__ == "__main__":
    main()
