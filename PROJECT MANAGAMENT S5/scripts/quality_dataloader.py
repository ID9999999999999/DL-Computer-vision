import argparse
import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


CLASS_TO_LABEL = {"good": 0, "bad": 1}
LABEL_TO_CLASS = {0: "good", 1: "bad"}
SPLITS = ("train", "validation", "test")


def default_transform():
    """Resize images and convert them to normalized tensors for CNN training."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def find_dataset_dir(dataset_dir=None):
    """
    Resolve the dataset directory safely.

    Priority:
    1) --dataset_dir provided by the user
    2) Common locations relative to this script/project
    3) Search nearby project folders for metadata.jsonl
    """
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    if dataset_dir is not None:
        candidate = Path(dataset_dir).expanduser().resolve()
        if (candidate / "metadata.jsonl").is_file():
            return candidate

        raise FileNotFoundError(
            "\nDataset directory is not valid.\n"
            f"Expected file:\n  {candidate / 'metadata.jsonl'}\n\n"
            "Pass the folder that directly contains metadata.jsonl, good/, and bad/."
        )

    candidates = [
        project_dir / "data" / "demo_quality_dataset",
        project_dir / "data" / "quality_dataset",
        project_dir / "demo_quality_dataset",
        project_dir / "quality_dataset",
        Path.cwd() / "data" / "demo_quality_dataset",
        Path.cwd() / "data" / "quality_dataset",
    ]

    for candidate in candidates:
        if (candidate / "metadata.jsonl").is_file():
            return candidate.resolve()

    search_roots = []
    for root in [project_dir, project_dir.parent, Path.cwd()]:
        root = root.resolve()
        if root not in search_roots and root.exists():
            search_roots.append(root)

    found = []
    for root in search_roots:
        try:
            for metadata_path in root.rglob("metadata.jsonl"):
                lowered_parts = {part.lower() for part in metadata_path.parts}
                if any(
                    blocked in lowered_parts
                    for blocked in {
                        ".git",
                        ".venv",
                        "venv",
                        "__pycache__",
                        "site-packages",
                        "node_modules",
                    }
                ):
                    continue

                dataset_candidate = metadata_path.parent
                found.append(dataset_candidate)

                if (
                    (dataset_candidate / "good").is_dir()
                    and (dataset_candidate / "bad").is_dir()
                ):
                    return dataset_candidate.resolve()
        except (PermissionError, OSError):
            continue

    if found:
        return found[0].resolve()

    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "\nCould not find the quality dataset automatically.\n\n"
        "The program needs a folder containing:\n"
        "  metadata.jsonl\n"
        "  good/\n"
        "  bad/\n\n"
        "Checked common locations:\n"
        f"{checked}\n\n"
        "If your dataset is somewhere else, run for example:\n"
        '  python quality_dataloader.py --dataset_dir "C:\\path\\to\\dataset"\n'
    )


class QualityDataset(Dataset):
    def __init__(self, dataset_dir, split, transform=None):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.metadata_path = self.dataset_dir / "metadata.jsonl"
        self.transform = transform or default_transform()
        self.records = []

        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")

        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"metadata.jsonl not found in: {self.dataset_dir}"
            )

        with self.metadata_path.open("r", encoding="utf-8") as metadata_file:
            for line_number, line in enumerate(metadata_file, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of "
                        f"{self.metadata_path}"
                    ) from exc

                if record.get("split") != split:
                    continue

                quality_label = record.get("quality_label")
                if quality_label not in CLASS_TO_LABEL:
                    continue

                relative_image_path = record.get("image_path")
                if not relative_image_path:
                    continue

                image_path = self.dataset_dir / relative_image_path
                if not image_path.is_file():
                    continue

                record["_full_image_path"] = image_path
                self.records.append(record)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        with Image.open(record["_full_image_path"]) as image_file:
            image = image_file.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = CLASS_TO_LABEL[record["quality_label"]]
        return image, label


def main():
    parser = argparse.ArgumentParser(
        description="Test the Lab 05 PyTorch DataLoader."
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=None,
        help=(
            "Folder containing metadata.jsonl, good/, and bad/. "
            "If omitted, the script tries to find it automatically."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    dataset_dir = find_dataset_dir(args.dataset_dir)

    datasets = {
        split: QualityDataset(dataset_dir, split=split)
        for split in SPLITS
    }

    for split, dataset in datasets.items():
        if len(dataset) == 0:
            raise ValueError(
                f"No valid records found for split={split}. "
                f"Dataset: {dataset_dir}"
            )

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
        )
        for split, dataset in datasets.items()
    }

    print("=" * 60)
    print("Lab 05 - PyTorch DataLoader Test")
    print("=" * 60)
    print(f"Dataset directory: {dataset_dir}")
    print(f"Train samples: {len(datasets['train'])}")
    print(f"Validation samples: {len(datasets['validation'])}")
    print(f"Test samples: {len(datasets['test'])}")
    print()

    images, labels = next(iter(loaders["train"]))
    label_ids = labels.tolist()

    print("Training batch loaded successfully.")
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Example labels: {label_ids}")
    print(f"Label names: {[LABEL_TO_CLASS[label] for label in label_ids]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
