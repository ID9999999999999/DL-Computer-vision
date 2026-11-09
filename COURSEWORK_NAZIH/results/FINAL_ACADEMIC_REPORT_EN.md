# Industrial Defect Classification: Cross-Entropy vs Poly-1 Loss

**Course:** AI Implementation Technologies
**Student:** YASSER IDBOUZKRI
**Group:** иикб-24
**Date:** 10 November 2026

## 1. Research Objective

This exploratory experiment investigates whether increasing training-image diversity improves industrial defect classification, and whether Poly-1 Loss outperforms standard Cross-Entropy.

## 2. Dataset

Dataset: KolektorSDD2.
Binary classification: Normal (0) vs Defective (1).
Training-set sizes: 70, 140, 280, 420 images.
Each training set was balanced between the two classes.
Evaluation used 1,004 official test images.

## 3. Experimental Method

- Architecture: ResNet18 pretrained on ImageNet.
- Input resolution: 448 x 448 RGB.
- Loss functions: Cross-Entropy and Poly-1.
- Seeds: 20261028, 20261029, 20261030.
- Batch size: 4.
- Optimizer: AdamW.
- Learning rate: 0.0001.
- Weight decay: 0.0001.
- 12 reference-equivalent training rounds.
- 216 optimizer updates per run.
- 70 image exposures per round.
- BatchNorm recalibration on the original 70-image set.

The 70-image reference was archived from earlier experiments and was not retrained. Comparisons are exploratory because software-environment differences cannot be fully excluded.

## 4. Results

| Images | Loss | Macro-F1 | Defective Recall | Normal FPR | AUROC |
|---:|---|---:|---:|---:|---:|
| 70 | ce | 76.64% | 61.21% | 5.85% | 85.68% |
| 70 | poly1 | 75.72% | 64.55% | 7.42% | 85.62% |
| 140 | ce | 77.86% | 66.67% | 6.49% | 88.25% |
| 140 | poly1 | 79.24% | 71.21% | 6.71% | 88.95% |
| 280 | ce | 84.76% | 72.42% | 3.24% | 89.67% |
| 280 | poly1 | 86.10% | 71.21% | 2.27% | 90.80% |
| 420 | ce | 84.80% | 77.27% | 4.25% | 92.27% |
| 420 | poly1 | 86.07% | 75.15% | 3.21% | 93.07% |

## 5. Discussion

Macro-F1 improved substantially from 70 to 280 training images. The gain from 280 to 420 was negligible under the fixed optimizer-update budget.

Poly-1 achieved higher mean Macro-F1 at 140, 280, and 420 images, but not at 70 images. Across the nine paired new-run comparisons, Poly-1 exceeded Cross-Entropy in seven cases.

These results suggest that image diversity may contribute more to performance gains than changing the loss function alone. However, the experiment does not establish a causal effect independent of numerical and historical-run differences.

## 6. Limitations

- The official test set had already been inspected in earlier work.
- Results are exploratory, not blind confirmatory estimates.
- Physical-part independence across splits was not independently verified.
- Only three random seeds were used per condition.
- The number of effective full-data passes differs by training-set size.
- The historical 70-image runs were not repeated in the new environment.

## 7. Conclusion

Under a fixed budget of 216 optimizer updates, increasing training-set diversity from 70 to 280 images was associated with higher Macro-F1. Poly-1 provided a modest average advantage at larger sample sizes, although the benefit was not consistent across every seed.

A separate untouched evaluation set and additional replication would be required for stronger confirmation.

## 8. References

- KolektorSDD2: official ViCoS dataset.
- PolyLoss: ICLR 2022.
- ResNet: He et al., 2016.
