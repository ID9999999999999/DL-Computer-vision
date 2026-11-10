"""Separate, resume-safe full-epoch pilot. NEVER changes the original 216-step study.

Data: 420 balanced KSDD2 official TRAIN images, plus a disjoint 180-image
validation subset selected only from remaining official TRAIN images.
No official TEST image or label is read by this experiment.

Run:
  python COURSEWORK_NAZIH/scripts/full_epoch_pilot.py --archive "C:/Users/YASSER/Downloads/KolektorSDD2.zip" --check-only
  python COURSEWORK_NAZIH/scripts/full_epoch_pilot.py --archive "C:/Users/YASSER/Downloads/KolektorSDD2.zip" --epochs 30 --size 420 --seed 20261028 --loss ce --device auto

Outputs live under COURSEWORK_NAZIH/data/full_epoch_pilot/ (Git-ignored).
The old results/, protocol.json, and checkpoints are never modified.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import io
import json
import os
import random
import sys
import time
import zipfile
import re
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import ROOT, read_json, sha_file, poly1, verify_archive

IMAGE_SIDE = 448
MEAN = torch.tensor([.485, .456, .406])[:, None, None]
STD = torch.tensor([.229, .224, .225])[:, None, None]
PILOT_VERSION = "full_epochs_v1_validation_from_train_20261110"
SEED_VALIDATION = 20261110
VAL_COUNTS = {0: 150, 1: 30}
FROZEN_SIZE_MANIFEST_SHA256 = "a5c51cca306a8d66bec0dbd5e4ecbe24df71fb331aa2afec023fdc4518ae44cf"


def verified_size_manifests(archive: Path):
    """Reuse the old manifest, or reproduce it EXACTLY from official train/ only.

    The original manifest may be untracked on a fresh Colab clone. Rebuilding
    it is permitted only when the SHA256 matches the historically frozen one.
    No existing original results/ file is modified.
    """
    old = ROOT / "results" / "size_manifests.json"
    generated = ROOT / "data" / "full_epoch_pilot" / "reference_size_manifests.json"
    if old.is_file():
        assert sha_file(old) == FROZEN_SIZE_MANIFEST_SHA256, (
            "Existing training manifest no longer matches the frozen study"
        )
        return read_json(old), FROZEN_SIZE_MANIFEST_SHA256
    if generated.is_file():
        assert sha_file(generated) == FROZEN_SIZE_MANIFEST_SHA256, (
            "Locally reconstructed manifest is corrupted"
        )
        return read_json(generated), FROZEN_SIZE_MANIFEST_SHA256

    protocol = read_json(ROOT / "protocol.json")
    original70 = read_json(ROOT / "input" / "target_manifest.json")
    assert len(original70) == 70
    by_path = {r["zip_path"]: r for r in original70}
    assert len(by_path) == 70
    extra = {0: [], 1: []}
    counts = collections.Counter()
    found_original = 0
    with zipfile.ZipFile(archive) as z:
        names = sorted(n for n in z.namelist()
                       if re.fullmatch(r"train/[0-9]+\.png", n))
        assert len(names) == 2331
        for name in names:
            raw = z.read(name)
            file_hash = hashlib.sha256(raw).hexdigest()
            with Image.open(io.BytesIO(z.read(name[:-4] + "_GT.png"))) as gt:
                label = int(np.asarray(gt).max() > 0)
            counts[label] += 1
            if name in by_path:
                ref = by_path[name]
                assert ref["sha256"] == file_hash and ref["label"] == label
                found_original += 1
            else:
                extra[label].append({
                    "zip_path": name, "sha256": file_hash, "label": label,
                    "from_original70": False
                })
    assert found_original == 70 and counts == {0: 2085, 1: 246}, counts
    baseline = [{
        "zip_path": r["zip_path"], "sha256": r["sha256"],
        "label": r["label"], "from_original70": True
    } for r in original70]
    rng = random.Random(protocol["extra_selection_seed"])
    for label in (0, 1):
        rng.shuffle(extra[label])
    pools = {"70": baseline}
    for size in (140, 280, 420):
        take = size // 2 - 35
        rows = baseline + extra[0][:take] + extra[1][:take]
        assert len(rows) == size
        assert len({r["sha256"] for r in rows}) == size
        pools[str(size)] = rows
    reconstructed = (json.dumps(pools, indent=2) + "\n").encode("utf-8")
    assert hashlib.sha256(reconstructed).hexdigest() == FROZEN_SIZE_MANIFEST_SHA256, (
        "Reconstruction failed to match the frozen experimental population"
    )
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_bytes(reconstructed)
    print("✅ FROZEN TRAINING POOLS EXACTLY RECONSTRUCTED")
    return pools, FROZEN_SIZE_MANIFEST_SHA256


def atomic_torch_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        # On a previous Colab disconnection, an incomplete temporary file
        # might remain. Preserve it for inspection; never treat it as a
        # valid checkpoint and never overwrite a completed result.
        abandoned = temporary.with_name(
            temporary.name + f".interrupted_{int(time.time())}_{os.getpid()}"
        )
        if abandoned.exists():
            raise FileExistsError(f"Cannot preserve interrupted file: {abandoned}")
        temporary.replace(abandoned)
        print(f"PRESERVED INTERRUPTED TEMP FILE: {abandoned}", flush=True)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        # On a previous Colab disconnection, an incomplete temporary file
        # might remain. Preserve it for inspection; never treat it as a
        # valid checkpoint and never overwrite a completed result.
        abandoned = temporary.with_name(
            temporary.name + f".interrupted_{int(time.time())}_{os.getpid()}"
        )
        if abandoned.exists():
            raise FileExistsError(f"Cannot preserve interrupted file: {abandoned}")
        temporary.replace(abandoned)
        print(f"PRESERVED INTERRUPTED TEMP FILE: {abandoned}", flush=True)
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def fixed_splits(archive: Path, size: int):
    """Only train/ files; labels from corresponding train/*_GT.png; never test/."""
    protocol = read_json(ROOT / "protocol.json")
    assert size in (140, 280, 420), "Only existing experimental pools may be used"
    pools, manifest_hash = verified_size_manifests(archive)
    train_rows = pools[str(size)]
    assert len(train_rows) == size
    assert sum(r["label"] == 0 for r in train_rows) == size // 2
    assert sum(r["label"] == 1 for r in train_rows) == size // 2
    assert len({r["sha256"] for r in train_rows}) == size

    # Official test is consulted ONLY as a local hash manifest to exclude overlap.
    # Actual official test images and labels are never accessed from the archive.
    test_hashes = {r["sha256"] for r in read_json(ROOT / "input" / "test_manifest.json")}
    assert not {r["sha256"] for r in train_rows} & test_hashes
    pool_paths = {r["zip_path"] for r in train_rows}
    all_train_names = sorted(n for n in zipfile.ZipFile(archive).namelist()
                             if n.startswith("train/") and n.endswith(".png")
                             and not n.endswith("_GT.png")
                             and n[6:-4].isdigit())
    assert len(all_train_names) == protocol["official_training_images"]
    remaining = {0: [], 1: []}

    with zipfile.ZipFile(archive) as z:
        for name in all_train_names:
            if name in pool_paths:
                continue
            with Image.open(io.BytesIO(z.read(name[:-4] + "_GT.png"))) as gt:
                label = int(np.asarray(gt).max() > 0)
            raw = z.read(name)
            entry = {"zip_path": name,
                     "sha256": hashlib.sha256(raw).hexdigest(),
                     "label": label}
            remaining[label].append(entry)

    # Deliberately fixed before any model results are seen; classes are scarce.
    rng = random.Random(SEED_VALIDATION)
    validation = []
    for label in (0, 1):
        remaining[label].sort(key=lambda r: r["zip_path"])
        rng.shuffle(remaining[label])
        assert len(remaining[label]) >= VAL_COUNTS[label]
        validation.extend(remaining[label][:VAL_COUNTS[label]])

    train_hashes = {r["sha256"] for r in train_rows}
    val_hashes = {r["sha256"] for r in validation}
    assert len(val_hashes) == sum(VAL_COUNTS.values())
    assert not train_hashes & val_hashes and not val_hashes & test_hashes
    train_names = {r["zip_path"] for r in train_rows}
    val_names = {r["zip_path"] for r in validation}
    assert not train_names & val_names
    # Dataset coverage across train and validation remains from official train only.
    return list(train_rows), validation, manifest_hash


class ZipImageReader:
    def __init__(self, archive, rows):
        self.zip = zipfile.ZipFile(archive)
        self.by_name = {r["zip_path"]: r for r in rows}
        self.cache = {}
        self.queue = []

    def image(self, name):
        if name in self.cache:
            return self.cache[name]
        row = self.by_name[name]
        raw = self.zip.read(name)
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"Image SHA256 mismatch: {name}")
        with Image.open(io.BytesIO(raw)) as im:
            arr = np.asarray(im.convert("RGB").resize(
                (IMAGE_SIDE, IMAGE_SIDE), Image.Resampling.BICUBIC
            ), dtype=np.uint8).copy()
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255)
        tensor = (tensor - MEAN) / STD
        if len(self.queue) >= 48:
            del self.cache[self.queue.pop(0)]
        self.cache[name] = tensor
        self.queue.append(name)
        return tensor

    def close(self):
        self.zip.close()


def batches(rows, reader, batch_size, order):
    for start in range(0, len(order), batch_size):
        subset = [rows[i] for i in order[start:start + batch_size]]
        x = torch.stack([reader.image(r["zip_path"]) for r in subset])
        y = torch.tensor([r["label"] for r in subset], dtype=torch.long)
        yield x, y


def scores(y, pred):
    values = {}
    f1s = []
    for label in (0, 1):
        tp = sum(a == label and b == label for a, b in zip(y, pred))
        fp = sum(a != label and b == label for a, b in zip(y, pred))
        fn = sum(a == label and b != label for a, b in zip(y, pred))
        f1s.append(2 * tp / max(1, (2 * tp + fp + fn)))
    values["macro_f1"] = float(sum(f1s) / 2)
    values["accuracy"] = sum(a == b for a, b in zip(y, pred)) / len(y)
    values["defect_recall"] = sum(a == 1 and b == 1 for a, b in zip(y, pred)) / sum(a == 1 for a in y)
    values["normal_fpr"] = sum(a == 0 and b == 1 for a, b in zip(y, pred)) / sum(a == 0 for a in y)
    return values


def train(args):
    archive = Path(args.archive).expanduser()
    with verify_archive(archive):
        pass
    train_rows, val_rows, manifest_hash = fixed_splits(archive, args.size)
    selection = {
        "version": PILOT_VERSION, "dataset_sha256": sha_file(archive),
        "manifest_sha256": manifest_hash,
        "training_rows": [{k: r[k] for k in ("zip_path", "sha256", "label")} for r in train_rows],
        "validation_rows": val_rows,
        "train_count": len(train_rows), "validation_count": len(val_rows),
        "validation_origin": "official training split ONLY",
        "test_policy": "no test image opened, no test-based epoch selection",
        "physical_part_independence": "not yet externally verified",
        "validation_seed": SEED_VALIDATION,
    }
    out = ROOT / "data" / "full_epoch_pilot" / f"n{args.size}_seed{args.seed}_{args.loss}"
    plan_file = out / "split.json"
    if plan_file.exists():
        assert read_json(plan_file) == selection, "Existing dataset split differs. Stop."
    elif not args.check_only:
        atomic_json_save(selection, plan_file)
    print("CHECK: official archive verified; train", len(train_rows),
          "validation", len(val_rows), "train/validation disjoint; no test images read")

    if args.check_only:
        print("✅ PILOT DATA PREFLIGHT PASSED")
        return

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device == "cpu" and not args.allow_cpu:
        raise RuntimeError("Full 448px ResNet18 training on CPU is expensive. "
                           "Run on Colab CUDA or pass --allow-cpu explicitly.")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.batch_size != 4:
        raise ValueError("Pilot comparison freezes batch size at 4")
    if args.seed not in (20261028, 20261029, 20261030):
        raise ValueError("Seed must match an archived reference seed")

    complete = out / "COMPLETE.json"
    state_file = out / "resume_state.pt"
    progress_file = out / "progress.json"
    best_file = out / "best_model.pt"
    if complete.exists():
        raise FileExistsError(f"Completed run preserved; refusing overwrite: {complete}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    official_weights = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
    assert official_weights.is_file(), "The original ImageNet1K V1 pretrained weights were not downloaded"
    assert sha_file(official_weights) == read_json(ROOT / "protocol.json")["pretrained_sha256"], (
        "Pretrained ImageNet weights differ from the frozen original protocol"
    )
    model.fc = nn.Linear(512, 2)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    steps_each = (len(train_rows) + args.batch_size - 1) // args.batch_size
    identity = {
        "version": PILOT_VERSION, "size": args.size, "seed": args.seed,
        "loss": args.loss, "epochs": args.epochs, "batch_size": 4,
        "updates_per_full_epoch": steps_each,
        "total_planned_optimizer_steps": steps_each * args.epochs,
        "preprocessing": "448 RGB BICUBIC, ImageNet mean/std",
        "model": "ResNet18 ImageNet1K_V1 full finetune",
        "optimizer": "AdamW lr=1e-4 weight_decay=1e-4",
        "validation_selection": "best macro-F1, first epoch on ties",
        "split_manifest_sha256": manifest_hash,
        "environment": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "pillow": Image.__version__,
            "cuda_device": (torch.cuda.get_device_name(0) if device == "cuda" else None),
        },
    }
    history = []
    best_metric = -1.0
    best_epoch = None
    best_state = None
    first_epoch = 1
    if state_file.exists():
        state = torch.load(state_file, map_location="cpu", weights_only=False)
        assert state["identity"] == identity, "Resume configuration mismatch"
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        best_metric = state["best_metric"]
        best_epoch = state["best_epoch"]
        best_state = state["best_state"]
        first_epoch = state["finished_epoch"] + 1
        assert len(history) == first_epoch - 1
        if best_epoch is not None:
            assert best_state is not None, "Resumable best model missing from state"
            # Restore exactly the BEST corresponding to last COMMITTED epoch.
            # If interrupted after writing best_model.pt but before resume state,
            # the newer uncommitted checkpoint cannot silently become the winner.
            atomic_torch_save(best_state, best_file)
        print(f"RESUMING at full epoch {first_epoch}")

    reader = ZipImageReader(archive, train_rows + val_rows)
    try:
        for epoch in range(first_epoch, args.epochs + 1):
            # Paired CE / Poly-1 runs see the exact same sample order.
            order = list(range(len(train_rows)))
            random.Random(args.seed * 100000 + epoch).shuffle(order)
            model.train()
            total_objective = 0.0
            total_examples = 0
            t0 = time.monotonic()
            for x, y in batches(train_rows, reader, args.batch_size, order):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                objective = (F.cross_entropy(logits, y) if args.loss == "ce"
                             else poly1(logits, y, epsilon=1.0))
                if not torch.isfinite(objective):
                    raise FloatingPointError(f"Nonfinite objective at epoch {epoch}")
                objective.backward()
                optimizer.step()
                total_objective += float(objective.detach()) * len(y)
                total_examples += len(y)
            assert total_examples == args.size

            model.eval()
            valid_labels, valid_preds = [], []
            validation_loss_sum = 0.0
            with torch.inference_mode():
                for x, y in batches(val_rows, reader, args.batch_size, range(len(val_rows))):
                    x, y_gpu = x.to(device), y.to(device)
                    logits = model(x)
                    validation_objective = (F.cross_entropy(logits, y_gpu) if args.loss == "ce"
                                            else poly1(logits, y_gpu, epsilon=1.0))
                    validation_loss_sum += float(validation_objective) * len(y)
                    valid_labels.extend(y.tolist())
                    valid_preds.extend(logits.argmax(1).cpu().tolist())
            metric = scores(valid_labels, valid_preds)
            record = {
                "full_epoch": epoch,
                "training_samples_seen": len(train_rows),
                "training_optimizer_steps": steps_each,
                "train_objective": total_objective / total_examples,
                "validation_objective": validation_loss_sum / len(val_rows),
                "validation": metric,
                "epoch_seconds": round(time.monotonic() - t0, 2),
            }
            history.append(record)
            if metric["macro_f1"] > best_metric:
                best_metric = metric["macro_f1"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                atomic_torch_save(best_state, best_file)

            # Crash-safe, epoch-boundary recovery; never touches legacy results.
            atomic_torch_save({
                "identity": identity, "finished_epoch": epoch,
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(), "history": history,
                "best_metric": best_metric, "best_epoch": best_epoch,
                "best_state": best_state,
            }, state_file)
            atomic_json_save({"identity": identity, "history": history,
                              "best_epoch": best_epoch, "best_validation_macro_f1": best_metric,
                              "status": "in_progress"}, progress_file)
            print(f"FULL EPOCH {epoch}/{args.epochs} train_obj={record['train_objective']:.5f}"
                  f" val_macro_f1={metric['macro_f1']:.4f} best_epoch={best_epoch}", flush=True)

        result = {
            "identity": identity,
            "status": "trained_and_selected_on_internal_training_holdout_only",
            "history": history, "best_epoch": best_epoch,
            "best_validation_macro_f1": best_metric,
            "best_model_sha256": sha_file(best_file),
            "split_sha256": sha_file(plan_file),
            "exploratory": True,
            "official_test_evaluated": False,
        }
        atomic_json_save(result, complete)
        print("✅ FULL-EPOCH PILOT COMPLETE:", complete)
        print("✅ BEST MODEL:", best_file)
    finally:
        reader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--size", type=int, default=420)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--loss", choices=("ce", "poly1"), default="ce")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    train(parser.parse_args())
