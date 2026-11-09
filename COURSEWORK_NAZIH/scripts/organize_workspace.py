"""Non-destructive, read-only scientific audit plus safe Drive upload staging.

Run in VS Code: python COURSEWORK_NAZIH/scripts/organize_workspace.py

No training, inference, Drive API, Git commits, file deletion, or overwrite of
different content occurs. Drive folders are already created separately.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
HOME = Path.home()
BACKUP = HOME / "Desktop" / "Training_Size_Results"
DRIVE_STAGE = ROOT / "_SYNC_TO_DRIVE"
DATASET = HOME / "Downloads" / "KolektorSDD2.zip"
DATASET_HASH = "edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8"
SIZES = (140, 280, 420)
SEEDS = (20261028, 20261029, 20261030)
LOSSES = ("ce", "poly1")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def safe_copy(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or hash_file(source) != hash_file(target):
            raise RuntimeError(f"CONFLICT; refusing overwrite: {target}")
    else:
        shutil.copy2(source, target)
    if hash_file(source) != hash_file(target):
        raise RuntimeError(f"COPY VERIFY FAILED: {target}")


def main():
    expected = []
    errors = []
    summary = {"checkpoint_pairs": 0, "new_evaluations": 0,
               "historical_pairs": 0, "official_zip_sha256_ok": False}
    # Checkpoint is a large file: never stage it for Drive a second time.
    for size in SIZES:
        for seed in SEEDS:
            for loss in LOSSES:
                stem = f"size{size}_seed{seed}_{loss}"
                model = RESULTS / f"{stem}.pt"
                info = RESULTS / f"{stem}.json"
                evaluation = RESULTS / f"eval_{stem}.json"
                if not model.exists() and not info.exists():
                    if (BACKUP / model.name).is_file() and (BACKUP / info.name).is_file():
                        safe_copy(BACKUP / model.name, model)
                        safe_copy(BACKUP / info.name, info)
                for path in (model, info, evaluation):
                    if not path.is_file():
                        errors.append(f"MISSING: {path.relative_to(ROOT)}")
                if not all(p.is_file() for p in (model, info, evaluation)):
                    continue
                report = load_json(info)
                outcome = load_json(evaluation)
                if (report.get("size"), report.get("seed"), report.get("loss")) != (size, seed, loss):
                    errors.append(f"Training identifier mismatch: {stem}")
                    continue
                if (outcome.get("size"), outcome.get("seed"), outcome.get("loss")) != (size, seed, loss):
                    errors.append(f"Evaluation identifier mismatch: {stem}")
                    continue
                digest = hash_file(model)
                if digest != report.get("checkpoint_sha256"):
                    errors.append(f"Training SHA256 mismatch: {stem}")
                    continue
                if digest != outcome.get("checkpoint_sha256"):
                    errors.append(f"Evaluation/checkpoint SHA256 mismatch: {stem}")
                    continue
                if len(report.get("train_history", [])) != 12 or report.get("total_optimizer_steps") != 216:
                    errors.append(f"Training protocol fields mismatch: {stem}")
                    continue
                if len(outcome.get("predictions", [])) != 1004 or not isinstance(outcome.get("metrics"), dict):
                    errors.append(f"Missing/invalid 1004 predictions or metrics: {stem}")
                    continue
                summary["checkpoint_pairs"] += 1
                summary["new_evaluations"] += 1
                expected.extend((model, info, evaluation))

    for seed in SEEDS:
        for loss in LOSSES:
            train = ROOT / "input" / "baseline_train" / f"target_train_{seed}_{loss}.json"
            evaluation = ROOT / "input" / "baseline_eval" / f"eval_{seed}_{loss}_epoch12.json"
            if not train.is_file() or not evaluation.is_file():
                errors.append(f"Missing historical pair: {seed}_{loss}")
                continue
            old_train, old_eval = load_json(train), load_json(evaluation)
            if (old_train.get("checkpoint_sha256") != old_eval.get("checkpoint_sha256")
                    or len(old_eval.get("predictions", [])) != 1004):
                errors.append(f"Historical provenance conflict: {seed}_{loss}")
                continue
            summary["historical_pairs"] += 1
            expected.extend((train, evaluation))

    if DATASET.is_file():
        summary["official_zip_sha256_ok"] = (hash_file(DATASET) == DATASET_HASH)
        if not summary["official_zip_sha256_ok"]:
            errors.append("Official dataset ZIP SHA256 mismatch.")
    else:
        errors.append("Official dataset ZIP missing from Downloads.")

    aggregate = RESULTS / "observed_results.csv"
    if aggregate.is_file():
        with aggregate.open("r", encoding="utf-8-sig", newline="") as file:
            count = len(list(csv.DictReader(file)))
        if count != 24:
            errors.append(f"Aggregate CSV rows: {count} (expected 24).")
        else:
            expected.append(aggregate)
    else:
        errors.append("Missing observed_results.csv.")

    extra_csv = ("NEW_18_EVALUATIONS.csv", "PAIRED_SEED_COMPARISON.csv")
    for name in extra_csv:
        file = RESULTS / name
        if file.is_file():
            expected.append(file)

    deliverables = ("REPORT_EN.md", "FINAL_ACADEMIC_REPORT_EN.md",
                    "NAZIH_FINAL_PRESENTATION.pptx")
    missing_deliverables = []
    for name in deliverables:
        file = RESULTS / name
        if file.is_file():
            expected.append(file)
        else:
            missing_deliverables.append(name)
    figures = sorted((RESULTS / "figures").glob("*.png")) if (RESULTS / "figures").is_dir() else []
    expected.extend(figures)

    if errors:
        print("\n".join("ERROR: " + e for e in errors))
        raise SystemExit("Audit stopped. No Drive staging attempted.")

    manifest_rows = []
    for file in sorted(set(expected)):
        manifest_rows.append({
            "relative_path": file.relative_to(ROOT).as_posix(),
            "size_bytes": file.stat().st_size,
            "sha256": hash_file(file),
        })
    manifest = RESULTS / "LOCAL_INTEGRITY_MANIFEST.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    status = RESULTS / "LOCAL_WORKSPACE_STATUS.md"
    status.write_text(
        "# Local project verification\n\n"
        f"- Verified new model/training pairs: {summary['checkpoint_pairs']}/18\n"
        f"- Verified new evaluation records: {summary['new_evaluations']}/18\n"
        f"- Verified archived historical pairs: {summary['historical_pairs']}/6\n"
        f"- Official dataset SHA256: {'MATCH' if summary['official_zip_sha256_ok'] else 'NOT VERIFIED'}\n"
        f"- Final report / PPTX missing: {', '.join(missing_deliverables) or 'none'}\n"
        f"- PNG figure count: {len(figures)}\n"
        "- 1,004-image reused test set: exploratory, not independent confirmation.\n"
        "- This is a LOCAL audit, not proof of GitHub or Google Drive parity.\n",
        encoding="utf-8",
    )

    # Safe staging copies, preserving the existing 36-file model backup on Drive.
    for file in expected:
        rel = file.relative_to(ROOT)
        if rel.parts[0] == "input":
            drive_folder = "01_Historical_70"
            target = DRIVE_STAGE / drive_folder / rel.parts[1] / rel.name
        elif file.name.startswith("eval_size") or file.suffix.lower() == ".csv":
            target = DRIVE_STAGE / "02_Evaluation_Results" / file.name
        elif file.name in deliverables or rel.parts[:2] == ("results", "figures"):
            target = DRIVE_STAGE / "03_Final_Deliverables" / (
                Path("figures") / file.name if rel.parts[:2] == ("results", "figures") else Path(file.name))
        else:
            continue  # large .pt + training .json already backed up on Drive
        safe_copy(file, target)

    for file in (manifest, status):
        safe_copy(file, DRIVE_STAGE / "04_Integrity_And_Audits" / file.name)

    print("VERIFIED:", summary)
    print("Optional missing deliverables:", missing_deliverables)
    print("PNG figures:", len(figures))
    print("DRIVE STAGING:", DRIVE_STAGE)
    print("NO training, deletion, Git push or Drive upload performed.")
    print("Review these staged folders before uploading to corresponding Drive folders.")


if __name__ == "__main__":
    main()
