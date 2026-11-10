"""Additive 30-FULL-epoch pilot. Never reads official test images or mutates old experiments."""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import random
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from core import poly1, digest_state, sha_file

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_json(obj):
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    if temp.exists():
        raise FileExistsError("Inspect previous unfinished write: " + str(temp))
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def validated_archive(archive):
    archive = Path(archive)
    assert archive.is_file(), f"Dataset ZIP missing: {archive}"
    actual = sha_file(archive)
    assert actual == CFG["dataset_sha256"], f"Wrong dataset SHA256: {actual}"
    return archive


def create_validation_manifest(archive, study):
    """Use only unused images from official TRAIN; test predictions/labels never opened."""
    pools = json.loads((ROOT / "results" / "size_manifests.json").read_text(encoding="utf-8"))
    train = pools[str(CFG["pilot_size"])]
    assert len(train) == 420
    assert collections.Counter(row["label"] for row in train) == {0: 210, 1: 210}
    train_paths = {r["zip_path"] for r in train}
    train_hashes = {r["sha256"] for r in train}
    assert len(train_paths) == len(train_hashes) == len(train)

    prior_test = json.loads((ROOT / "input" / "test_manifest.json").read_text(encoding="utf-8"))
    test_hashes = {r["sha256"] for r in prior_test}
    assert len(prior_test) == 1004
    assert not train_hashes.intersection(test_hashes)

    grouped = {0: [], 1: []}
    with zipfile.ZipFile(archive) as z:
        official = sorted(n for n in z.namelist() if re.fullmatch(r"train/[0-9]+\.png", n))
        assert len(official) == 2331
        assert train_paths.issubset(set(official))
        # Exact verify all 420 old pool images, without opening the official test.
        for row in train:
            assert hashlib.sha256(z.read(row["zip_path"])).hexdigest() == row["sha256"]
        for name in official:
            if name in train_paths:
                continue
            with Image.open(io.BytesIO(z.read(name[:-4] + "_GT.png"))) as im:
                label = int(np.asarray(im).max() > 0)
            grouped[label].append(name)
        assert len(grouped[1]) == 36 and len(grouped[0]) == 1875, (
            len(grouped[0]), len(grouped[1])
        )
        rng = random.Random(CFG["validation_selection_seed"])
        for label in (0, 1):
            rng.shuffle(grouped[label])
        chosen = grouped[0][:CFG["val_normal"]] + grouped[1][:CFG["val_defective"]]
        rows = []
        for name in chosen:
            data = z.read(name)
            sha = hashlib.sha256(data).hexdigest()
            assert sha not in train_hashes and sha not in test_hashes
            label = 0 if name in grouped[0][:CFG["val_normal"]] else 1
            rows.append({"zip_path": name, "sha256": sha, "label": label})
    assert len(rows) == CFG["val_normal"] + CFG["val_defective"]
    assert len({r["sha256"] for r in rows}) == len(rows)
    assert set(x["zip_path"] for x in rows).isdisjoint(train_paths)
    assert collections.Counter(r["label"] for r in rows) == {0: 270, 1: 30}
    manifest = {
        "protocol_version": CFG["version"],
        "source_sha256": CFG["dataset_sha256"],
        "selection_seed": CFG["validation_selection_seed"],
        "selection_rule": "unused official train only; natural approximately 10-percent defect prevalence",
        "physical_part_independence": "unverified",
        "rows": rows
    }
    target = study / "validation_manifest.json"
    if target.exists():
        saved = json.loads(target.read_text(encoding="utf-8"))
        assert saved == manifest, "Existing validation manifest differs; STOP."
    else:
        atomic_json(target, manifest)
    return train, rows, digest_json(manifest)


class Images(Dataset):
    def __init__(self, archive, rows):
        self.archive = str(archive)
        self.rows = rows
        self.handle = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        if self.handle is None:
            self.handle = zipfile.ZipFile(self.archive)
        data = self.handle.read(row["zip_path"])
        # Raw source hashes were all verified in create_validation_manifest().
        with Image.open(io.BytesIO(data)) as im:
            pixels = np.asarray(
                im.convert("RGB").resize((448, 448), Image.Resampling.BICUBIC),
                dtype=np.uint8,
            ).copy()
        x = torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255.0)
        return (x - MEAN) / STD, int(row["label"])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def compute_scores(labels, predictions, scores, ce_total):
    from sklearn.metrics import roc_auc_score
    cm = [
        [sum(y == a and p == b for y, p in zip(labels, predictions)) for b in (0, 1)]
        for a in (0, 1)
    ]
    f1 = []
    for label in (0, 1):
        tp = cm[label][label]
        denom = sum(cm[label]) + sum(row[label] for row in cm)
        f1.append(2.0 * tp / denom if denom else 0.0)
    return {
        "val_ce": ce_total / len(labels),
        "val_macro_f1": sum(f1) / 2,
        "val_auroc": float(roc_auc_score(labels, scores)),
        "val_defective_recall": cm[1][1] / sum(cm[1]),
        "val_normal_fpr": cm[0][1] / sum(cm[0]),
        "val_confusion_matrix": cm,
    }


@torch.inference_mode()
def validate(model, data_loader, device):
    model.eval()
    ys, ps, scores = [], [], []
    ce_total = 0.0
    for xs, labels in data_loader:
        logits = model(xs.to(device, non_blocking=True))
        targets = labels.to(device, non_blocking=True)
        ce_total += F.cross_entropy(logits, targets, reduction="sum").item()
        ys.extend(labels.tolist())
        ps.extend(logits.argmax(dim=1).cpu().tolist())
        scores.extend((logits[:, 1] - logits[:, 0]).cpu().tolist())
    assert len(ys) == CFG["val_normal"] + CFG["val_defective"]
    return compute_scores(ys, ps, scores, ce_total)


def ckpt_path(run_dir, epoch):
    return run_dir / f"epoch_{epoch:03d}.pt"


def save_checkpoint(run_dir, epoch, payload):
    destination = ckpt_path(run_dir, epoch)
    if destination.exists() or destination.with_suffix(".sha256").exists():
        raise FileExistsError("Refusing to overwrite a checkpoint: " + str(destination))
    temp = destination.with_name(destination.name + ".partial")
    if temp.exists():
        raise FileExistsError("Inspect unfinished checkpoint: " + str(temp))
    torch.save(payload, temp)
    digest = sha_file(temp)
    os.replace(temp, destination)
    destination.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")


def load_latest_checkpoint(run_dir, device, run_identity):
    candidates = []
    for file in run_dir.glob("epoch_*.pt"):
        match = re.fullmatch(r"epoch_(\d{3})\.pt", file.name)
        if not match:
            raise ValueError("Unexpected checkpoint name: " + str(file))
        epoch = int(match.group(1))
        signature = file.with_suffix(".sha256")
        if not signature.exists() or sha_file(file) != signature.read_text().strip():
            raise ValueError("Checkpoint missing/corrupt checksum: " + str(file))
        candidates.append((epoch, file))
    if not candidates:
        return None
    epoch, file = max(candidates)
    # Load ONLY snapshots made by this script in the user's trusted study folder.
    data = torch.load(file, map_location=device, weights_only=False)
    assert data["epoch"] == epoch and data["identity"] == run_identity
    assert len(data["history"]) == epoch
    return data


def run(args):
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "No CUDA GPU. In Colab select Runtime > Change runtime type > T4 GPU. "
            "Use --allow-cpu only if deliberately accepting much slower training."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    archive = validated_archive(args.archive)
    study = Path(args.study_dir).resolve()
    study.mkdir(parents=True, exist_ok=True)
    run_dir = study / "runs" / f"size420_seed{args.seed}_{args.loss}"
    run_dir.mkdir(parents=True, exist_ok=True)
    completion = run_dir / "completed.json"
    if completion.exists():
        print("ALREADY COMPLETE; no overwrite:", completion)
        return

    train_rows, val_rows, split_hash = create_validation_manifest(archive, study)
    seed_all(args.seed)
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    checkpoint_location = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
    assert checkpoint_location.exists()
    assert sha_file(checkpoint_location) == CFG["pretrained_sha256"]
    model.fc = nn.Linear(512, 2)
    initial_hash = digest_state(model.state_dict())
    identity = {
        "study": CFG["version"],
        "size": CFG["pilot_size"],
        "seed": args.seed,
        "loss": args.loss,
        "epochs": CFG["full_epochs"],
        "dataset_sha256": CFG["dataset_sha256"],
        "validation_sha256": split_hash,
        "initial_state_sha256": initial_hash,
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "device_kind": device.type,
    }
    atomic_identity = run_dir / "identity.json"
    if atomic_identity.exists():
        assert json.loads(atomic_identity.read_text()) == identity, "Run identity drift"
    else:
        atomic_json(atomic_identity, identity)

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"]
    )
    previous = load_latest_checkpoint(run_dir, device, identity)
    start = 0
    history = []
    if previous is not None:
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        history = previous["history"]
        start = previous["epoch"]
        print(f"RESUME verified checkpoint epoch {start}/{CFG['full_epochs']}")

    train_ds, val_ds = Images(archive, train_rows), Images(archive, val_rows)
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=0)
    if args.dry_run:
        xs, ys = next(iter(DataLoader(train_ds, batch_size=CFG["batch_size"])))
        assert xs.shape == (CFG["batch_size"], 3, 448, 448)
        print("DRY RUN PASS", identity, "first_labels=", ys.tolist())
        return

    for epoch in range(start + 1, CFG["full_epochs"] + 1):
        clock = time.monotonic()
        # Fixed per-epoch shuffle independently of any resumed worker/RNG state.
        rng = torch.Generator().manual_seed(args.seed + 104729 * epoch)
        train_loader = DataLoader(
            train_ds, batch_size=CFG["batch_size"], shuffle=True, generator=rng, num_workers=0
        )
        model.train()
        objective_sum = 0.0
        seen = 0
        steps = 0
        for xs, labels in train_loader:
            xs = xs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xs)
            loss = F.cross_entropy(logits, labels) if args.loss == "ce" else poly1(logits, labels, epsilon=1)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            objective_sum += float(loss.detach()) * len(labels)
            seen += len(labels)
            steps += 1
        assert seen == 420 and steps == 105, (seen, steps)
        metrics = validate(model, val_loader, device)
        entry = {
            "epoch": epoch,
            "train_objective": objective_sum / seen,
            "train_images_seen": seen,
            "optimizer_steps": steps,
            "elapsed_seconds": time.monotonic() - clock,
            **metrics,
        }
        history.append(entry)
        # Mutable lightweight progress file; immutable optimizer snapshots every 5 full epochs.
        atomic_json(run_dir / "history.json", {"identity": identity, "epochs": history})
        print(
            f"epoch={epoch:02d}/30 train={entry['train_objective']:.4f} "
            f"val_CE={entry['val_ce']:.4f} val_F1={entry['val_macro_f1']:.4f} "
            f"val_AUROC={entry['val_auroc']:.4f}",
            flush=True
        )
        if epoch % CFG["checkpoint_interval"] == 0:
            save_checkpoint(
                run_dir, epoch,
                {"epoch": epoch, "identity": identity, "history": list(history),
                 "model": model.state_dict(), "optimizer": optimizer.state_dict()}
            )
            print("SAVED verified resume snapshot:", epoch, flush=True)

    assert len(history) == CFG["full_epochs"]
    # Prespecified candidate set: checkpoints every 5 epochs; no test-based selection.
    candidates = [r for r in history if r["epoch"] % CFG["checkpoint_interval"] == 0]
    best = max(candidates, key=lambda r: (r["val_macro_f1"], -r["val_ce"], -r["epoch"]))
    best_path = ckpt_path(run_dir, best["epoch"])
    assert best_path.is_file()
    # Preserve all optimizer snapshots; save isolated lightweight selected weights.
    selected = torch.load(best_path, map_location="cpu", weights_only=False)
    chosen = run_dir / "best_validation_model.pt"
    if chosen.exists():
        raise FileExistsError("Selected model already exists; inspect instead of overwrite")
    torch.save(selected["model"], chosen)
    chosen_sha = sha_file(chosen)
    result = {
        "status": "COMPLETE; VALIDATION ONLY, OFFICIAL TEST NOT READ",
        "identity": identity,
        "total_full_dataset_epochs": CFG["full_epochs"],
        "total_optimizer_steps": 105 * CFG["full_epochs"],
        "best_epoch_from_prespecified_candidates": best["epoch"],
        "selection": "max validation macro-F1 among epochs 5,10,15,20,25,30; val CE tie-break; earlier epoch final tie-break",
        "best_validation_metrics": best,
        "best_model_sha256": chosen_sha,
        "best_model_filename": chosen.name,
        "snapshots": [f"epoch_{e:03d}.pt" for e in range(5, 31, 5)],
        "warning": "small exploratory validation (30 defective); physical-part isolation unverified; prior official test already examined; do not claim independent test improvement"
    }
    atomic_json(completion, result)
    print("PILOT COMPLETE; best val epoch:", best["epoch"], "weights SHA256:", chosen_sha)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True)
    p.add_argument("--study-dir", required=True)
    p.add_argument("--seed", type=int, default=20261028, choices=CFG["seeds"])
    p.add_argument("--loss", choices=["ce", "poly1"], default="ce")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
