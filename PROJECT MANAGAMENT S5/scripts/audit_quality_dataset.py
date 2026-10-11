import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


REQUIRED_FIELDS = [
    "image_id",
    "class_name",
    "quality_label",
    "image_path",
    "split",
    "source_id",
]

VALID_QUALITY_LABELS = {"good", "bad"}
VALID_SPLITS = {"train", "validation", "test"}


def audit_dataset(dataset_dir, target_size=(640, 640), check_pixels=True):
    """Check metadata consistency and image files for the quality dataset."""
    dataset_dir = Path(dataset_dir)
    metadata_path = dataset_dir / "metadata.jsonl"

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.jsonl not found in {dataset_dir}")

    valid_records = []
    errors = []
    total_metadata_records = 0

    image_id_counts = Counter()
    image_path_counts = Counter()
    source_split_map = defaultdict(set)

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        for line_number, line in enumerate(metadata_file, start=1):
            line = line.strip()
            if not line:
                continue

            total_metadata_records += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({
                    "line": line_number,
                    "error": f"Invalid JSON: {exc}",
                })
                continue

            missing_fields = [
                field
                for field in REQUIRED_FIELDS
                if field not in record or record[field] in (None, "")
            ]
            if missing_fields:
                errors.append({
                    "line": line_number,
                    "image_id": record.get("image_id", "unknown"),
                    "error": f"Missing fields: {missing_fields}",
                })
                continue

            image_id = record["image_id"]
            image_path = record["image_path"]
            quality_label = record["quality_label"]
            split = record["split"]
            source_id = record["source_id"]

            image_id_counts[image_id] += 1
            image_path_counts[image_path] += 1
            source_split_map[source_id].add(split)

            if quality_label not in VALID_QUALITY_LABELS:
                errors.append({
                    "line": line_number,
                    "image_id": image_id,
                    "error": f"Invalid quality_label: {quality_label}",
                })
                continue

            if split not in VALID_SPLITS:
                errors.append({
                    "line": line_number,
                    "image_id": image_id,
                    "error": f"Invalid split: {split}",
                })
                continue

            full_image_path = dataset_dir / image_path
            if not full_image_path.is_file():
                errors.append({
                    "line": line_number,
                    "image_id": image_id,
                    "error": f"Image file not found: {image_path}",
                })
                continue

            if check_pixels:
                try:
                    with Image.open(full_image_path) as image:
                        image.verify()

                    with Image.open(full_image_path) as image:
                        if image.size != tuple(target_size):
                            errors.append({
                                "line": line_number,
                                "image_id": image_id,
                                "error": (
                                    f"Wrong image size: {image.size}. "
                                    f"Expected {target_size}."
                                ),
                            })
                            continue
                except Exception as exc:
                    errors.append({
                        "line": line_number,
                        "image_id": image_id,
                        "error": f"Cannot open image: {exc}",
                    })
                    continue

            valid_records.append(record)

    duplicate_image_ids = sorted(
        image_id for image_id, count in image_id_counts.items() if count > 1
    )
    duplicate_image_paths = sorted(
        image_path for image_path, count in image_path_counts.items() if count > 1
    )
    leakage_sources = {
        source_id: sorted(splits)
        for source_id, splits in sorted(source_split_map.items())
        if len(splits) > 1
    }

    quality_counts = Counter(r["quality_label"] for r in valid_records)
    class_counts = Counter(r["class_name"] for r in valid_records)
    split_counts = Counter(r["split"] for r in valid_records)

    warnings = []

    good_count = quality_counts.get("good", 0)
    bad_count = quality_counts.get("bad", 0)
    if good_count == 0 or bad_count == 0:
        warnings.append("Dataset has only one quality class.")
    else:
        imbalance = abs(good_count - bad_count) / max(good_count, bad_count)
        if imbalance > 0.10:
            warnings.append(
                f"Good/bad imbalance detected: good={good_count}, bad={bad_count}."
            )

    if class_counts:
        max_class = max(class_counts.values())
        min_class = min(class_counts.values())
        if min_class > 0 and max_class / min_class > 1.10:
            warnings.append(f"Document class imbalance detected: {dict(class_counts)}.")

    for split_name in ("train", "validation", "test"):
        if split_counts.get(split_name, 0) == 0:
            warnings.append(f"Split '{split_name}' is empty.")

    if duplicate_image_ids:
        warnings.append("Duplicate image_id values found.")
    if duplicate_image_paths:
        warnings.append("Duplicate image_path values found.")
    if leakage_sources:
        warnings.append(
            "Data leakage detected: the same source_id appears in multiple splits."
        )

    ready_for_training = (
        len(errors) == 0
        and not duplicate_image_ids
        and not duplicate_image_paths
        and not leakage_sources
        and good_count > 0
        and bad_count > 0
        and all(split_counts.get(name, 0) > 0 for name in VALID_SPLITS)
    )

    summary = {
        "dataset_dir": str(dataset_dir),
        "total_metadata_records": total_metadata_records,
        "total_valid_records": len(valid_records),
        "total_errors": len(errors),
        "quality_label": dict(sorted(quality_counts.items())),
        "class_name": dict(sorted(class_counts.items())),
        "split": {
            name: split_counts.get(name, 0)
            for name in ("train", "validation", "test")
        },
        "duplicate_image_ids": duplicate_image_ids,
        "duplicate_image_paths": duplicate_image_paths,
        "source_id_leakage": leakage_sources,
        "warnings": warnings,
        "ready_for_training": ready_for_training,
    }

    return summary, errors


def main():
    parser = argparse.ArgumentParser(description="Audit a good/bad quality dataset.")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("data/demo_quality_dataset"),
        help="Folder containing good/, bad/, metadata.jsonl and summary.json",
    )
    parser.add_argument(
        "--output_report",
        type=Path,
        default=Path("reports/dataset_audit_report.json"),
        help="Path for the JSON audit report",
    )
    parser.add_argument(
        "--no_pixel_check",
        action="store_true",
        help="Skip opening images and checking 640x640 size",
    )
    args = parser.parse_args()

    summary, errors = audit_dataset(
        args.dataset_dir,
        check_pixels=not args.no_pixel_check,
    )

    report = {"summary": summary, "errors": errors}
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Dataset audit finished.")
    print(f"Report saved to: {args.output_report}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
