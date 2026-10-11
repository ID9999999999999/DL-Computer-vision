# Lab 05: Data Loading, Validation, and Dataset Auditing

## Dataset

Dataset location: `data/demo_quality_dataset`

Total metadata records: **1800**  
Total valid records: **1800**

## Quality balance

- good: 900
- bad: 900

The two quality classes are balanced.

## Document class balance

- inn: 600
- passport: 600
- snils: 600

The document classes are also balanced.

## Split distribution

- train: 1230
- validation: 228
- test: 342

All three required splits contain data.

## Audit results

- Missing metadata fields: none found
- Missing image files: none found
- Corrupted images: none found
- Wrong image sizes: none found
- Invalid quality labels: none found
- Invalid split values: none found
- Duplicate image IDs: none found
- Duplicate image paths: none found
- Source ID leakage across splits: none found
- Total errors: 0
- Total warnings: 0

## DataLoader test

The PyTorch Dataset/DataLoader loaded the train, validation, and test splits successfully.

Example training batch shape:

`torch.Size([16, 3, 224, 224])`

Example label IDs from the test run:

`[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1]`

where `0 = good` and `1 = bad`.

## Conclusion

The dataset passed the audit without errors or leakage. It is balanced, all required splits are available, and the PyTorch DataLoader works correctly. The dataset is ready for model training.
