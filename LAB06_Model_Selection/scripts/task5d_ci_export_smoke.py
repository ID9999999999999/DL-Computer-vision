"""CPU-only Task 5D ONNX export smoke benchmark (synthetic inputs, not test accuracy).
Real 270-image evaluation remains in task5d_export_onnx_benchmark.py.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def as_prob(x):
    import torch
    if isinstance(x, (tuple, list)):
        x = next(y for y in x if isinstance(y, torch.Tensor) and y.ndim == 2 and y.shape[-1] == 2)
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x, dtype="float64").reshape(-1)
    if a.shape != (2,) or not np.isfinite(a).all():
        raise RuntimeError(f"Unexpected classification output: {a.shape}")
    if a.min() >= 0 and abs(a.sum()-1) < 1e-4:
        return a/a.sum()
    e = np.exp(a-a.max())
    return e/e.sum()

def timing(fn, inputs):
    for i in range(20):
        fn(inputs[i % len(inputs)])
    samples = []
    for i in range(100):
        t0 = time.perf_counter()
        fn(inputs[i % len(inputs)])
        samples.append((time.perf_counter()-t0)*1000.0)
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": float(np.percentile(samples, 95)),
        "warmups": 20, "measured": 100,
        "timed_ms": samples
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import torch
    import onnx
    import onnxruntime as ort
    import ultralytics
    from ultralytics import YOLO
    os.environ["OMP_NUM_THREADS"] = "4"
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    args.out.mkdir(parents=True, exist_ok=True)
    expected_sha = "c820e22debf8d3fe3c3ca41f7c10f55d9e7db0b2246fb90c92a8548a70a514f7"
    source_sha = digest(args.weights)
    if source_sha != expected_sha:
        raise RuntimeError("Source checkpoint SHA-256 differs from original Lab06 YOLOv8n")
    m = YOLO(str(args.weights))
    names = {int(k):str(v).lower() for k,v in m.names.items()}
    if set(names.values()) != {"bad","good"} or len(names)!=2:
        raise RuntimeError(f"Unexpected classes: {names}")
    bad_idx = next(i for i,v in names.items() if v=="bad")
    m.model.cpu().eval()
    exported = Path(m.export(format="onnx",imgsz=224,batch=1,device="cpu",dynamic=False,simplify=False))
    onnx.checker.check_model(str(exported))
    dest = args.out / "yolov8n_quality_gate.onnx"
    dest.write_bytes(exported.read_bytes())

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    session=ort.InferenceSession(str(dest),sess_options=opts,providers=["CPUExecutionProvider"])
    if session.get_providers()!=["CPUExecutionProvider"]:
        raise RuntimeError("ORT did not use only CPUExecutionProvider")
    in_name=session.get_inputs()[0].name
    generator=torch.Generator().manual_seed(2473986)
    tensors=[torch.rand((1,3,224,224),generator=generator) for _ in range(12)]
    np_inputs=[x.numpy() for x in tensors]
    deviations=[]
    mismatches=0
    with torch.inference_mode():
        for a,b in zip(tensors,np_inputs):
            p=as_prob(m.model(a))
            q=as_prob(session.run(None,{in_name:b})[0])
            deviations.append(float(np.max(np.abs(p-q))))
            mismatches+=int(int(np.argmax(p))!=int(np.argmax(q)))
        native=timing(lambda i:m.model(tensors[i]),list(range(10)))
    onnx_perf=timing(lambda i:session.run(None,{in_name:np_inputs[i]}),list(range(10)))
    result={
      "task":"5D","status":"ONNX export and synthetic CPU forward-only benchmark EXECUTED; held-out Test accuracy NOT RUN",
      "source_sha256":source_sha,"onnx_sha256":digest(dest),
      "source_bytes":args.weights.stat().st_size,"onnx_bytes":dest.stat().st_size,
      "input_shape":[1,3,224,224],"cpu_threads":4,
      "warmup_runs":20,"timed_runs":100,
      "synthetic_input_count":len(tensors),
      "test_set_accuracy_before":None,"test_set_accuracy_after":None,
      "test_accuracy_drop":None,
      "parity_synthetic_max_prob_abs_difference":max(deviations),
      "parity_synthetic_label_disagreements":mismatches,
      "native_cpu_forward_only":native,"onnx_cpu_forward_only":onnx_perf,
      "environment":{"platform":platform.platform(),"torch":torch.__version__,"ultralytics":ultralytics.__version__,"onnx":onnx.__version__,"onnxruntime":ort.__version__},
      "limitations":[
        "No original test images available to GitHub Actions; no accuracy evidence.",
        "Synthetic inputs are not representative of real document photos.",
        "Forward-only timings exclude preprocessing and IO; not comparable to old 26.97 ms pipeline result.",
        "CI runner is not an edge device; no quantization performed."]
    }
    (args.out/"results_5d_smoke.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    memo=f"""# Task 5D ONNX export — automated CPU smoke run
- Model: original checkpoint, SHA-256 {source_sha}
- ONNX checker: passed
- Size: PyTorch {args.weights.stat().st_size/1048576:.3f} MiB; ONNX {dest.stat().st_size/1048576:.3f} MiB
- CPU forward-only: PyTorch {native["mean_ms"]:.3f} ms, ONNX {onnx_perf["mean_ms"]:.3f} ms (20 warm-up + 100 measured, four threads, same runner)
- Synthetic tensor output max absolute probability difference: {max(deviations):.8f}
- Synthetic tensor class disagreements: {mismatches}/{len(tensors)}
- Actual held-out Test before/after accuracy: **NOT MEASURED**, because private Dataset A was intentionally not uploaded to CI.
- **Task 5D not complete until the 270 original test images are evaluated before/after export**. Do not report a zero accuracy drop.
- No edge device measurement or quantization.
"""
    (args.out/"CI_README.md").write_text(memo,encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("native_cpu_forward_only","onnx_cpu_forward_only")},indent=2))
    print("PyTorch mean",native["mean_ms"],"ONNX mean",onnx_perf["mean_ms"])

if __name__=="__main__":
    main()
