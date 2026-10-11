# Lab 05 - Data Loading, Validation, and Dataset Auditing

This folder contains the completed Lab 05 work for loading, validating, auditing, and testing a document-quality dataset with PyTorch.

## Project structure

```text
LAB05_Data_Loading_Validation_Dataset_Auditing/
├── README.md
├── Lab05_Report.md
├── requirements.txt
├── scripts/
│   ├── audit_quality_dataset.py
│   └── quality_dataloader.py
├── reports/
│   ├── dataset_audit_report.json
│   └── dataloader_test_output.txt
└── data/
    └── demo_quality_dataset/
        └── summary.json
```

The local dataset used for the audit contains 1,800 images and follows this structure:

```text
data/demo_quality_dataset/
├── good/
├── bad/
├── metadata.jsonl
└── summary.json
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the audit

```bash
python scripts/audit_quality_dataset.py --dataset_dir data/demo_quality_dataset
```

The JSON report is written to:

```text
reports/dataset_audit_report.json
```

## Test the PyTorch Dataset/DataLoader

```bash
python scripts/quality_dataloader.py --dataset_dir data/demo_quality_dataset --batch_size 16
```

## Verified results

- Total records: **1800**
- Valid records: **1800**
- good: **900**
- bad: **900**
- inn: **600**
- passport: **600**
- snils: **600**
- train: **1230**
- validation: **228**
- test: **342**
- audit errors: **0**
- audit warnings: **0**
- duplicate IDs: **0**
- duplicate paths: **0**
- source leakage: **0**
- example batch shape: `torch.Size([16, 3, 224, 224])`

See `Lab05_Report.md` for the short report and `reports/` for the generated outputs.
