# Dataset location

The Lab 05 scripts expect the quality dataset in this folder.

Expected structure:

```text
data/demo_quality_dataset/
├── good/                 # 900 images
├── bad/                  # 900 images
├── metadata.jsonl        # 1800 metadata records
└── summary.json
```

The dataset used for the verified Lab 05 run contains 1,800 RGB images at 640×640 pixels. The image set is the dataset prepared in the preceding dataset-preparation work; it is kept out of the normal Git history because the local image payload is about 197 MB.

Verified distribution:

- good: 900
- bad: 900
- inn: 600
- passport: 600
- snils: 600
- train: 1230
- validation: 228
- test: 342

The generated audit output for this exact dataset is stored in `../../reports/dataset_audit_report.json`.
