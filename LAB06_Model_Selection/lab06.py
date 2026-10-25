"""Lab 06: reproducible document-quality classification. No fabricated results.

Training needs the user's local image dataset and pretrained-weight downloads.
Use `python lab06.py run --dataset ".../demo_quality_dataset"`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ("mobilenetv2", "efficientnetb0", "yolov8n", "yolov8s")
LABELS = {"good": 0, "bad": 1}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def source_split(rows, seed):
    """Exact 70/15/15 per document class; never separate variants of a source.

    Refuse unequal group sizes or nonintegral quotas rather than silently
    producing a nonconforming split. The supplied metadata satisfies both.
    """
    groups = defaultdict(list)
    for r in rows:
        groups[r["source_id"]].append(r)
    strata = defaultdict(list)
    for source, members in groups.items():
        names = {r["class_name"] for r in members}
        if len(names) != 1:
            raise ValueError(f"Source {source} has inconsistent document classes")
        if Counter(r["quality_label"] for r in members) != {"good": 3, "bad": 3}:
            raise ValueError(f"Expected 3 good + 3 bad variants for source {source}")
        strata[next(iter(names))].append(source)
    allocation = {}
    rng = random.Random(seed)
    for name, sources in sorted(strata.items()):
        sources = sorted(sources)
        rng.shuffle(sources)
        if len(sources) % 20:
            raise ValueError(f"Cannot make exact 70/15/15 source quotas for {name}")
        a, b = len(sources) * 70 // 100, len(sources) * 85 // 100
        for i, source in enumerate(sources):
            allocation[source] = "train" if i < a else "val" if i < b else "test"
    return [dict(r, original_split=r["split"], split=allocation[r["source_id"]]) for r in rows]


def safe_image(dataset, relative):
    path = (dataset / relative).resolve()
    if not path.is_relative_to(dataset) or not path.is_file():
        raise FileNotFoundError(f"Missing or invalid dataset image: {relative}")
    return path


def prepare(dataset, out, cfg):
    from PIL import Image
    if (out / "protocol.json").exists():
        raise FileExistsError("This output already has a frozen protocol. Use a NEW --out directory for a new experiment.")
    dataset = dataset.expanduser().resolve()
    metadata = dataset / "metadata.jsonl"
    if not metadata.is_file():
        raise FileNotFoundError(f"Expected {metadata}; point --dataset to the folder containing good/ and bad/.")
    rows = load_rows(metadata)
    if not rows:
        raise ValueError("Empty metadata")
    required = {"source_id", "class_name", "quality_label", "image_id", "image_path", "split"}
    for row in rows:
        if not required.issubset(row) or row["quality_label"] not in LABELS:
            raise ValueError("Invalid metadata record")
        if "seed" in row and row["seed"] != cfg["seed"]:
            raise ValueError("Dataset seed differs from config.json")
    for key in ("image_id", "image_path"):
        if len({r[key] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate {key}")
    rows = source_split(rows, cfg["seed"])
    hashes = defaultdict(set)
    for row in rows:
        path = safe_image(dataset, row["image_path"])
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.size != (640, 640):
                raise ValueError(f"Expected 640x640 image: {path.name}, got {image.size}")
            image.convert("RGB").load()
        row["sha256"] = digest(path)
        hashes[row["sha256"]].add(row["split"])
    if any(len(splits) > 1 for splits in hashes.values()):
        raise ValueError("Identical image bytes cross the new splits; resolve duplicate source identities first.")
    out.mkdir(parents=True, exist_ok=True)
    # Ultralytics receives precisely the same manifest as Keras. Hard links save
    # disk space when possible; copying is the portable cross-volume fallback.
    for row in rows:
        destination = out / "yolo_data" / row["split"] / row["quality_label"] / (row["image_id"] + ".jpg")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Partial previous preparation: {destination}. Choose a fresh --out.")
        source = safe_image(dataset, row["image_path"])
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    write_json(out / "manifest.json", rows)
    shutil.copy2(metadata, out / "metadata_original.jsonl")
    protocol = {"config": cfg, "config_sha256": digest(ROOT / "config.json"),
                "manifest_sha256": digest(out / "manifest.json"), "dataset": str(dataset),
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "old_split": dict(Counter(r["original_split"] for r in rows)),
                "new_split": dict(Counter(r["split"] for r in rows)),
                "positive_class": "bad", "label_mapping": LABELS,
                "selection": "validation only; fastest passing model, otherwise highest validation F1 (provisional)",
                "test_policy": "fixed 0.5 threshold, no test-driven tuning or selection",
                "augmentation": "none; preserve synthetic quality defects and whole document",
                "latency": "CPU, batch=1, full RGB PIL image already in RAM through resize/preprocess + forward + output; excludes disk decode, includes tensor conversion and CPU host output"}
    write_json(out / "protocol.json", protocol)
    versions = {p: importlib.metadata.version(p) for p in ("tensorflow", "ultralytics", "torch", "torchvision", "numpy", "Pillow", "scikit-learn", "matplotlib")}
    write_json(out / "environment.json", {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(), "versions": versions})
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (out / "environment.txt").write_text(freeze.stdout, encoding="utf-8")
    print("Data prepared:", protocol["new_split"], flush=True)


def protocol(out):
    p = read_json(out / "protocol.json")
    if digest(ROOT / "config.json") != p["config_sha256"]:
        raise ValueError("config.json changed after preparation; use a new output directory.")
    if digest(out / "manifest.json") != p["manifest_sha256"]:
        raise ValueError("Frozen manifest changed")
    return p


def check_images(out, p):
    dataset = Path(p["dataset"])
    for r in read_json(out / "manifest.json"):
        if digest(safe_image(dataset, r["image_path"])) != r["sha256"]:
            raise ValueError(f"Image changed after preparation: {r['image_id']}")


def setup(cfg, tensorflow=False):
    os.environ["PYTHONHASHSEED"] = str(cfg["seed"])
    os.environ["OMP_NUM_THREADS"] = str(cfg["cpu_threads"])
    random.seed(cfg["seed"])
    import numpy as np
    np.random.seed(cfg["seed"])
    if tensorflow:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(cfg["cpu_threads"])
        tf.keras.utils.set_random_seed(cfg["seed"])
        tf.config.experimental.enable_op_determinism()
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
        return tf


def keras_dataset(rows, dataset, cfg, training=False):
    import tensorflow as tf
    import numpy as np
    from PIL import Image
    paths = [str(safe_image(dataset, r["image_path"])) for r in rows]
    labels = [LABELS[r["quality_label"]] for r in rows]

    def load(path, label):
        # PIL bilinear matches the explicit YOLO transform and benchmark input.
        def pixels(value):
            filename = value.numpy().decode("utf-8")
            with Image.open(filename) as im:
                return np.asarray(im.convert("RGB").resize((cfg["image_size"], cfg["image_size"]), Image.Resampling.BILINEAR), dtype=np.float32)
        image = tf.py_function(pixels, [path], Tout=tf.float32)
        image.set_shape((cfg["image_size"], cfg["image_size"], 3))
        return image, tf.cast(label, tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=cfg["seed"], reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=1, deterministic=True)
    return ds.batch(cfg["batch_size"]).prefetch(1)


def train_keras(out, name):
    p = protocol(out)
    cfg = p["config"]
    tf = setup(cfg, tensorflow=True)
    check_images(out, p)
    rows = read_json(out / "manifest.json")
    folder = out / name
    folder.mkdir(exist_ok=True)
    if (folder / "training.json").exists():
        raise FileExistsError(f"{name} already trained; use existing outputs or a fresh experiment.")
    x = tf.keras.Input((224, 224, 3), name="rgb_0_255")
    if name == "mobilenetv2":
        # Exact MobileNetV2 preprocessing, inside the exported model.
        v = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(x)
        base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    else:
        # Keras EfficientNet already includes its own Rescaling layer.
        v = x
        base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base.trainable = False
    v = base(v, training=False)
    v = tf.keras.layers.GlobalAveragePooling2D()(v)
    v = tf.keras.layers.Dense(cfg["dense_units"], activation="relu")(v)
    v = tf.keras.layers.Dropout(cfg["dropout"])(v)
    y = tf.keras.layers.Dense(1, activation="sigmoid", name="p_bad")(v)
    model = tf.keras.Model(x, y, name=name)
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["keras_learning_rate"]), loss="binary_crossentropy", metrics=["accuracy"])
    train = keras_dataset([r for r in rows if r["split"] == "train"], Path(p["dataset"]), cfg, True)
    val = keras_dataset([r for r in rows if r["split"] == "val"], Path(p["dataset"]), cfg)
    callbacks = [tf.keras.callbacks.ModelCheckpoint(str(folder / "best.keras"), monitor="val_loss", save_best_only=True),
                 tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["patience"], restore_best_weights=True),
                 tf.keras.callbacks.CSVLogger(str(folder / "history.csv"))]
    start = time.perf_counter()
    history = model.fit(train, validation_data=val, epochs=cfg["epochs"], callbacks=callbacks)
    elapsed = time.perf_counter() - start
    write_json(folder / "history.json", history.history)
    write_json(folder / "training.json", {"model": name, "regime": "frozen ImageNet base, train head", "training_seconds": elapsed,
               "parameters": int(model.count_params()), "trainable_parameters": int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)),
               "epochs_run": len(history.history["loss"]), "selection": "lowest validation loss", "checkpoint": str(folder / "best.keras")})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, metric in enumerate(("loss", "accuracy")):
        ax[i].plot(range(1, len(history.history[metric]) + 1), history.history[metric], label="train")
        ax[i].plot(range(1, len(history.history[metric]) + 1), history.history["val_" + metric], label="validation")
        ax[i].set(xlabel="Epoch", ylabel=metric.title())
        ax[i].legend()
    fig.tight_layout()
    fig.savefig(folder / "curves.png", dpi=160)
    plt.close(fig)


def yolo_transform():
    from torchvision import transforms as T
    # Ultralytics classification normalization is currently identity (0/1).
    # Explicit transform avoids random crops erasing a defect in training.
    return T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor()])


def train_yolo(out, name):
    p = protocol(out)
    cfg = p["config"]
    setup(cfg)
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.classify.train import ClassificationTrainer
    torch.set_num_threads(cfg["cpu_threads"])
    torch.manual_seed(cfg["seed"])
    check_images(out, p)
    folder = out / name
    if (folder / "training.json").exists():
        raise FileExistsError(f"{name} already trained")

    class WholeDocumentTrainer(ClassificationTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            ds = super().build_dataset(img_path, mode, batch)
            ds.torch_transforms = yolo_transform()
            return ds

    model = YOLO(name + "-cls.pt")
    device = 0 if torch.cuda.is_available() else "cpu"
    start = time.perf_counter()
    model.train(trainer=WholeDocumentTrainer, data=str(out / "yolo_data"), epochs=cfg["epochs"],
                imgsz=224, batch=cfg["batch_size"], workers=0, device=device,
                optimizer="Adam", lr0=cfg["yolo_learning_rate"], lrf=1.0,
                seed=cfg["seed"], deterministic=True, patience=cfg["patience"],
                pretrained=True, dropout=cfg["dropout"], amp=False,
                project=str(out), name=name, exist_ok=True,
                plots=True, save=True, cache=False, val=True,
                auto_augment=None, erasing=0.0, fliplr=0.0, flipud=0.0)
    elapsed = time.perf_counter() - start
    best = folder / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Training did not produce {best}")
    trained = YOLO(str(best)).model
    write_json(folder / "training.json", {"model": name, "regime": "full ImageNet fine-tuning; nano/small variants", "training_seconds": elapsed,
               "parameters": sum(v.numel() for v in trained.parameters()), "checkpoint": str(best),
               "selection": "Ultralytics validation fitness (not test)", "device": str(device),
               "note": "Frameworks and optimization regimes differ. This is a practical pipeline comparison, not a controlled backbone-only ablation."})


def metrics(y, probabilities, threshold):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    pred = [int(v >= threshold) for v in probabilities]
    return {"accuracy": float(accuracy_score(y, pred)),
            "precision_bad": float(precision_score(y, pred, zero_division=0)),
            "recall_bad": float(recall_score(y, pred, zero_division=0)),
            "f1_bad": float(f1_score(y, pred, zero_division=0)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(), "n": len(y)}


def evaluate(out, name):
    # The common deployment benchmark runs ALL models on CPU, in separate
    # processes, at batch=1. Native training may use the available GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    p = protocol(out)
    cfg = p["config"]
    setup(cfg)
    import numpy as np
    from PIL import Image
    folder = out / name
    if (folder / "evaluation.json").exists():
        raise FileExistsError("Evaluation already exists. Do not tune or repeatedly inspect the test set.")
    check_images(out, p)
    training = read_json(folder / "training.json")
    checkpoint = Path(training["checkpoint"])
    if name in ("mobilenetv2", "efficientnetb0"):
        tf = setup(cfg, tensorflow=True)
        tf.config.set_visible_devices([], "GPU")
        model = tf.keras.models.load_model(checkpoint, compile=False)

        @tf.function(reduce_retracing=True)
        def forward(x):
            return model(x, training=False)

        def predict(image):
            pixels = np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32)[None]
            return float(forward(tf.convert_to_tensor(pixels)).numpy().reshape(-1)[0])
    else:
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(cfg["cpu_threads"])
        torch.manual_seed(cfg["seed"])
        model = YOLO(str(checkpoint)).model.to("cpu").float().eval()
        names = model.names
        entries = names.items() if isinstance(names, dict) else enumerate(names)
        mapping = {str(v): int(k) for k, v in entries}
        if set(mapping) != {"good", "bad"}:
            raise ValueError(f"Unexpected YOLO class map: {names}")
        bad_index = mapping["bad"]
        transform = yolo_transform()

        def predict(image):
            tensor = transform(image).unsqueeze(0)
            with torch.inference_mode():
                value = model(tensor)
                probabilities = value[0] if isinstance(value, tuple) else value
                return float(probabilities[0, bad_index].cpu().item())
    rows = read_json(out / "manifest.json")
    val_rows = [r for r in rows if r["split"] == "val"]
    # Cycling real validation inputs (already decoded) makes benchmark input
    # preparation consistent without timing a single lucky image repeatedly.
    images = []
    for r in val_rows[: min(32, len(val_rows))]:
        with Image.open(safe_image(Path(p["dataset"]), r["image_path"])) as im:
            images.append(im.convert("RGB").copy())
    for i in range(cfg["latency_warmup"]):
        predict(images[i % len(images)])
    timings = []
    for i in range(cfg["latency_runs"]):
        start = time.perf_counter_ns()
        predict(images[i % len(images)])
        timings.append((time.perf_counter_ns() - start) / 1e6)
    evaluation = {"model": name, "threshold": cfg["threshold"], "positive_class": "bad",
                  "latency_ms": statistics.mean(timings), "latency_median_ms": statistics.median(timings),
                  "latency_p95_ms": float(np.percentile(timings, 95)), "latency_samples_ms": timings,
                  "latency_warmup": cfg["latency_warmup"], "latency_runs": cfg["latency_runs"],
                  "benchmark_device": "CPU", "cpu_threads": cfg["cpu_threads"],
                  "artifact_mib": checkpoint.stat().st_size / 1024**2,
                  "checkpoint_sha256": digest(checkpoint), **training}
    predictions = []
    for split in ("val", "test"):
        selected = [r for r in rows if r["split"] == split]
        probs = []
        for r in selected:
            with Image.open(safe_image(Path(p["dataset"]), r["image_path"])) as im:
                prob = predict(im.convert("RGB"))
            if not np.isfinite(prob) or not 0 <= prob <= 1:
                raise ValueError(f"Invalid probability: {prob}")
            probs.append(prob)
            predictions.append({"image_id": r["image_id"], "source_id": r["source_id"], "image_path": r["image_path"],
                                "split": split, "defect_type": r.get("defect_type", "unknown"),
                                "class_name": r["class_name"], "y_true": LABELS[r["quality_label"]],
                                "p_bad": prob, "y_pred": int(prob >= cfg["threshold"])})
        evaluation[split] = metrics([LABELS[r["quality_label"]] for r in selected], probs, cfg["threshold"])
    write_json(folder / "predictions.json", predictions)
    write_json(folder / "evaluation.json", evaluation)
    print(name, "test:", evaluation["test"], flush=True)


def qualifies(e, thresholds, split="val"):
    m = e[split]
    return (m["accuracy"] >= thresholds["accuracy"] and m["f1_bad"] >= thresholds["f1_bad"]
            and m["recall_bad"] >= thresholds["recall_bad"] and e["latency_ms"] <= thresholds["latency_ms"]
            and e["artifact_mib"] <= thresholds["artifact_mib"])


def choose(evals, thresholds):
    passing = [e for e in evals if qualifies(e, thresholds)]
    if passing:
        return min(passing, key=lambda e: (e["latency_ms"], -e["val"]["f1_bad"])), True
    return max(evals, key=lambda e: (e["val"]["f1_bad"], e["val"]["recall_bad"], -e["latency_ms"])), False


def report(out):
    p = protocol(out)
    cfg = p["config"]
    evals = [read_json(out / n / "evaluation.json") for n in MODELS]
    selected, passes = choose(evals, cfg["acceptance"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    rows = []
    for e in evals:
        rows.append({"model": e["model"], "val_accuracy": e["val"]["accuracy"], "val_f1_bad": e["val"]["f1_bad"],
                     **{k: e["test"][k] for k in ("accuracy", "precision_bad", "recall_bad", "f1_bad", "macro_f1")},
                     **{k: e[k] for k in ("latency_ms", "latency_p95_ms", "artifact_mib", "parameters", "training_seconds")}})
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for e in evals:
        ax.scatter(e["latency_ms"], e["test"]["accuracy"], s=70)
        ax.annotate(e["model"], (e["latency_ms"], e["test"]["accuracy"]), xytext=(4, 5), textcoords="offset points")
    ax.set(xlabel="CPU latency (ms/image; preprocessing + forward)", ylabel="Test accuracy", title="Final test characterization — selection uses validation only")
    ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(out / "accuracy_vs_speed.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, e in zip(axes, evals):
        cm = np.array(e["test"]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(max(row) for v in evals for row in v["test"]["confusion_matrix"]))
        for (i, j), count in np.ndenumerate(cm): ax.text(j, i, str(count), ha="center", va="center")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["good", "bad"], yticklabels=["good", "bad"], xlabel="Predicted", ylabel="True", title=e["model"])
    fig.tight_layout(); fig.savefig(out / "confusion_matrices.png", dpi=170); plt.close(fig)
    defect_rows = []
    for e in evals:
        preds = [r for r in read_json(out / e["model"] / "predictions.json") if r["split"] == "test"]
        for defect in sorted({r["defect_type"] for r in preds if r["y_true"] == 1}):
            cases = [r for r in preds if r["y_true"] == 1 and r["defect_type"] == defect]
            missed = sum(r["y_pred"] == 0 for r in cases)
            defect_rows.append({"model": e["model"], "defect": defect, "n": len(cases), "missed_bad": missed, "recall": 1 - missed / len(cases)})
        failures = sorted([r for r in preds if r["y_true"] != r["y_pred"]], key=lambda r: abs(r["p_bad"] - cfg["threshold"]), reverse=True)
        fig, axes = plt.subplots(2, 3, figsize=(11, 8))
        for ax in axes.flat: ax.axis("off")
        if not failures:
            axes.flat[0].text(0.1, 0.5, "No test errors at the fixed threshold", wrap=True)
        for ax, r in zip(axes.flat, failures[:6]):
            with Image.open(safe_image(Path(p["dataset"]), r["image_path"])) as image: ax.imshow(image.convert("RGB"))
            ax.set_title(f"{r['defect_type']} | true={r['y_true']}, pred={r['y_pred']}\np(bad)={r['p_bad']:.3f}", fontsize=10)
        fig.suptitle(e["model"] + ": most confident test errors; local analysis only")
        fig.tight_layout(); fig.savefig(out / e["model"] / "failure_gallery.png", dpi=150); plt.close(fig)
    with (out / "defect_analysis.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(defect_rows[0])); writer.writeheader(); writer.writerows(defect_rows)
    hard = min((r for r in defect_rows if r["model"] == selected["model"]), key=lambda r: r["recall"])
    lines = ["# Final comparison and decision memo", "", "Generated from actual local runs; no manually invented metrics.", "",
             "## Protocol", "", f"Seed {cfg['seed']}. Source-grouped split {p['new_split']}; original Lab05 split {p['old_split']}.",
             "All variants of each source remain together. RGB, 224×224, positive class bad=1, fixed threshold 0.5. No test labels were used for checkpoint choice, thresholds, or recommendation.", "",
             "## Final test comparison", "", "| Model | Accuracy | Precision bad | Recall bad | F1 bad | CPU ms | MiB | Parameters | Train s |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for e in evals:
        m = e["test"]
        lines.append(f"| {e['model']} | {m['accuracy']:.4f} | {m['precision_bad']:.4f} | {m['recall_bad']:.4f} | {m['f1_bad']:.4f} | {e['latency_ms']:.2f} | {e['artifact_mib']:.2f} | {e['parameters']:,} | {e['training_seconds']:.1f} |")
    lines += ["", "## Recommendation based on validation", "", f"Selected candidate: **{selected['model']}**.",
              "The candidate passed all provisional validation and cost limits and was the fastest passing candidate." if passes else "No candidate passed every provisional acceptance limit. The highest validation F1 candidate is retained for investigation; deployment is NOT accepted.",
              f"Validation F1={selected['val']['f1_bad']:.4f}; validation bad recall={selected['val']['recall_bad']:.4f}.",
              f"The selected candidate {'also passes' if qualifies(selected, cfg['acceptance'], 'test') else 'does not pass'} the same limits on final test characterization. This observation does not trigger reselection or tuning.", "",
              "Alternatives: the other measured candidates remain alternatives only if validated under a new preregistered experiment or on fresh independent data. A faster model can still be unsuitable if it accepts defective documents.", "",
              "## Defect analysis", "", f"Lowest measured defect recall for the selected model: {hard['defect']} ({hard['recall']:.3f}, n={hard['n']}). Ties are possible; inspect defect_analysis.csv. Glare was an expectation, not an imposed conclusion.",
              "The failure gallery shows the most confidently wrong examples. Bad→good mistakes allow unusable documents through the first gate; good→bad mistakes cause avoidable retries.", "",
              "## Training dynamics", ""]
    for name in ("mobilenetv2", "efficientnetb0"):
        h = read_json(out / name / "history.json")
        best = int(np.argmin(h["val_loss"]))
        gap = h["accuracy"][best] - h["val_accuracy"][best]
        trend = "Possible overfitting: validation loss finishes above its earlier minimum." if h["val_loss"][-1] > h["val_loss"][best] + 0.03 else "No strong late validation-loss rise by the declared 0.03 descriptive margin."
        lines.append(f"- {name}: best validation-loss epoch {best+1}; train/validation accuracy gap {gap:.3f}. {trend} Inspect curves.png before claiming overfitting or underfitting.")
    lines += ["", "YOLO learning curves and epoch metrics are in each YOLO run's results.png and results.csv. Compare training and validation loss trends there; do not infer underfitting from one metric alone.", "",
              "## Limits and risks", "", "This small synthetic dataset contains only 300 independent source documents. Image counts overstate independent sample size because variants are correlated. Real camera captures and unfamiliar documents require separate validation. Results compare complete training pipelines: Keras frozen bases versus YOLO full fine-tuning, different optimizers/checkpoint criteria, and different frameworks. No causal architecture-only claim is justified.",
              "CPU batch-one timings use the same decoded RGB images, thread count and timing scope; framework and host load still affect results. Artifact size is the actual checkpoint (including framework serialization and any optimizer state), not raw weight bytes. Training times include fit/trainer overhead and are not equivalent optimization budgets.",
              "A 0.5 sigmoid/softmax threshold does not establish calibration. Weak performance can reflect missing quality cues in frozen ImageNet features; a fresh experiment may test low-rate fine-tuning while holding out fresh final test data.", "",
              "## Completion status", "", "Models, metrics and plots are complete only if this report generated successfully. Repository commit, student review/defense, and the in-class presentation are separate actions still required."]
    (out / "Final_Report.md").write_text("\n".join(lines), encoding="utf-8")
    talk = f"""# Five-minute presentation — speaking outline

1. 0:00–0:40 — Problem: reject poor document images at a fast first gate. Explain good=0, bad=1 and the cost of a missed bad image.
2. 0:40–1:20 — Dataset: 1,800 variants, 300 source groups, inn/passport/snils; show protocol.json and the leakage-safe 70/15/15 allocation. Explain why related variants cannot cross splits.
3. 1:20–2:10 — Models: MobileNetV2 and EfficientNetB0 with frozen ImageNet bases; YOLOv8 nano and small classification with full fine-tuning. Explain preprocessing and the fixed threshold.
4. 2:10–3:10 — Show comparison.csv, accuracy_vs_speed.png, and confusion_matrices.png. Explain precision, recall, F1, batch-one CPU timing and checkpoint size.
5. 3:10–4:10 — Show training curves and failure galleries. Lowest measured defect recall for {selected['model']}: {hard['defect']} ({hard['recall']:.3f}); do not assume glare.
6. 4:10–5:00 — Validation-selected candidate: {selected['model']}. Acceptance status: {'passed provisional validation limits' if passes else 'no fully passing candidate'}. Explain the synthetic-to-real limitation and the next independent validation study.

Before class: read Final_Report.md, inspect every plotted result and be ready to explain source grouping, model freezing, normalization, bad-class mapping, checkpoint selection and the difference between validation and test data. This is an outline; the presentation has not been delivered.
"""
    (out / "Presentation_5min.md").write_text(talk, encoding="utf-8")
    write_json(out / "decision.json", {"candidate": selected["model"], "validation_acceptance_passed": passes, "test_acceptance_passed": qualifies(selected, cfg["acceptance"], "test"), "selection_data": "validation only"})
    print("Report ready:", out / "Final_Report.md", flush=True)


def run_step(command, out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.log").open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND " + repr(command) + "\n"); log.flush()
        environment = dict(os.environ, PYTHONHASHSEED=str(read_json(ROOT / "config.json")["seed"]), PYTHONIOENCODING="utf-8")
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=environment)
        for line in child.stdout:
            print(line, end="", flush=True); log.write(line); log.flush()
        code = child.wait()
        if code: raise subprocess.CalledProcessError(code, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "prepare", "train", "evaluate", "report"])
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    cfg = read_json(ROOT / "config.json")
    if cfg["image_size"] != 224 or cfg["threshold"] != 0.5:
        raise ValueError("This protocol implements 224x224 inputs and a fixed threshold of 0.5.")
    if args.action in ("run", "prepare") and not args.dataset:
        parser.error("--dataset is required (folder containing metadata.jsonl, good/, bad/)")
    if args.action == "run":
        prefix = [sys.executable, str(Path(__file__).resolve())]
        if not (out / "protocol.json").exists():
            run_step(prefix + ["prepare", "--dataset", str(args.dataset.resolve()), "--out", str(out)], out)
        elif str(args.dataset.expanduser().resolve()) != protocol(out)["dataset"]:
            raise ValueError("Resume dataset differs from frozen protocol")
        # Train every model BEFORE opening final test outputs.
        for name in MODELS:
            if not (out / name / "training.json").exists():
                if any(out.glob("*/evaluation.json")):
                    raise ValueError("Test evaluation already started; training new candidates in this experiment is prohibited.")
                run_step(prefix + ["train", "--model", name, "--out", str(out)], out)
        for name in MODELS:
            if not (out / name / "evaluation.json").exists():
                run_step(prefix + ["evaluate", "--model", name, "--out", str(out)], out)
        run_step(prefix + ["report", "--out", str(out)], out)
    elif args.action == "prepare":
        prepare(args.dataset, out, cfg)
    elif args.action == "train":
        if not args.model: parser.error("--model required")
        if any(out.glob("*/evaluation.json")):
            raise ValueError("Final evaluation already started; use a new independent experiment.")
        (train_keras if args.model in MODELS[:2] else train_yolo)(out, args.model)
    elif args.action == "evaluate":
        if not args.model: parser.error("--model required")
        if not all((out / name / "training.json").exists() for name in MODELS):
            raise ValueError("Train all four preregistered candidates before final evaluation")
        evaluate(out, args.model)
    else:
        report(out)


if __name__ == "__main__":
    main()
