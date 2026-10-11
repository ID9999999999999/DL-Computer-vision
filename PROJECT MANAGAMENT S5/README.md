# Lab 05 - Data Loading, Validation, and Dataset Auditing

## Run the audit

```bash
python scripts/audit_quality_dataset.py --dataset_dir data/demo_quality_dataset
```

The JSON report is saved to:

```text
reports/dataset_audit_report.json
```

## Test the PyTorch Dataset/DataLoader

```bash
python scripts/quality_dataloader.py --dataset_dir data/demo_quality_dataset --batch_size 16
```

The script loads the train, validation, and test splits, prints the number of samples in each split, and prints one training batch shape with example labels.
