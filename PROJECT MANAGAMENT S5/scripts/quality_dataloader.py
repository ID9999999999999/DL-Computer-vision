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


class QualityDataset(Dataset):
    def __init__(self, dataset_dir, split, transform=None):
        self.dataset_dir = Path(dataset_dir)
        self.metadata_path = self.dataset_dir / "metadata.jsonl"
        self.transform = transform or default_transform()
        self.records = []

        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"metadata.jsonl not found in {self.dataset_dir}")

        with self.metadata_path.open("r", encoding="utf-8") as metadata_file:
            for line in metadata_file:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                if record.get("split") != split:
                    continue

                quality_label = record.get("quality_label")
                if quality_label not in CLASS_TO_LABEL:
                    continue

                image_path = self.dataset_dir / record.get("image_path", "")
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
    parser = argparse.ArgumentParser(description="Test the Lab 05 PyTorch DataLoader.")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("data/demo_quality_dataset"),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    datasets = {
        split: QualityDataset(args.dataset_dir, split=split)
        for split in SPLITS
    }

    for split, dataset in datasets.items():
        if len(dataset) == 0:
            raise ValueError(f"No valid records found for split={split}")

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
        )
        for split, dataset in datasets.items()
    }

    print(f"Dataset directory: {args.dataset_dir}")
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


if __name__ == "__main__":
    main()
