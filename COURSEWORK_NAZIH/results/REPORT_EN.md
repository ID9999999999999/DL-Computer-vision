# Training-set size — report (exploratory)

YASSER IDBOUZKRI | иикб-24 | 8 November 2026

## Design
The original 70-image, 12-epoch results are preserved exactly and are NOT retrained.
New conditions use 140/280/420 balanced official KSDD2 training images, always 35 normal and 35 defective image exposures per epoch, 12 reference-equivalent 70-image rounds (6/3/2 full-dataset passes for sizes 140/280/420 respectively), 216 AdamW optimizer steps, original preprocessing, three paired seeds and both losses.
BatchNorm recalibration uses the same original 70 training images for every size. The input samples, not the computational training budget or BN calibration dataset, are the intended intervention. Effective full-data passes change with N; numerical environment effects cannot be ruled out without rerunning the old baseline.

**Evaluation caveat:** the 1,004 official test images have already been studied in earlier diagnosis. These are exploratory results, NOT an independent confirmatory test.
**Specimen caveat:** official train/test isolation and exact-hash checks do not certify physical-part independence.

## Actual 12-epoch evaluations (mean across 3 seeds)

| Unique training images | Loss | Macro-F1 | Defective recall | Normal FPR | AUROC | Evidence |
|---:|---|---:|---:|---:|---:|---|
| 70 | ce | 76.64% | 61.21% | 5.85% | 85.68% | Archived 12-epoch baseline; not rerun |
| 70 | poly1 | 75.72% | 64.55% | 7.42% | 85.62% | Archived 12-epoch baseline; not rerun |
| 140 | ce | 77.86% | 66.67% | 6.49% | 88.25% | Newly evaluated; repeated test |
| 140 | poly1 | 79.24% | 71.21% | 6.71% | 88.95% | Newly evaluated; repeated test |
| 280 | ce | 84.76% | 72.42% | 3.24% | 89.67% | Newly evaluated; repeated test |
| 280 | poly1 | 86.10% | 71.21% | 2.27% | 90.80% | Newly evaluated; repeated test |
| 420 | ce | 84.80% | 77.27% | 4.25% | 92.27% | Newly evaluated; repeated test |
| 420 | poly1 | 86.07% | 75.15% | 3.21% | 93.07% | Newly evaluated; repeated test |

## Required before a strong causal claim
- All predeclared training sizes complete for each paired seed and both losses.
- Independently verify the absence of physical-part overlap; no proof presently available.
- A new untouched confirmation set is necessary for confirmatory performance statements.
- Do not choose the best size or hyperparameters using this previously inspected test set.

**Valid evaluated runs: 24/24.** Six of them are archived, and 18 are new runs requiring actual training and inference.
No results are written for pending runs.
