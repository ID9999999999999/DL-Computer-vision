# Training duration: 3 vs 12 epochs
YASSER IDBOUZKRI — иикб-24 — 8 November 2026

One fixed KSDD2 training subset (35 normal, 35 defective); three paired seeds; identical initialization and batch order for CE and Poly-1. ResNet18 full fine-tuning, same square 448 image preprocessing, AdamW lr=wd=0.0001, batch size 4, no augmentation. The optimizer continues uninterrupted from update 54 to 216. Endpoint BN recalibration uses only training images. Epoch 12 was fixed before evaluation.

All conditions use the same 1,004 previously examined test images. This is an exploratory diagnosis, not independent confirmation, proof of convergence, or a tuning recommendation. The number of training images was not changed.

| Loss | Epochs | Macro-F1 | Defect recall | Normal false-positive rate | Balanced accuracy | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| ce | 3 | 71.73% | 60.91% | 9.73% | 75.59% | 83.87% |
| ce | 12 | 76.64% | 61.21% | 5.85% | 77.68% | 85.68% |
| poly1 | 3 | 72.79% | 61.82% | 9.06% | 76.38% | 83.97% |
| poly1 | 12 | 75.72% | 64.55% | 7.42% | 78.56% | 85.62% |

Paired Macro-F1 changes (percentage points):
- ce: +6.96, +3.58, +4.20; mean +4.91.
- poly1: +6.19, +1.18, +1.41; mean +2.93.

The per-run metrics, confusion matrices, predictions, training losses and checkpoint hashes are included. See metrics.csv for whether an epoch-3 prediction set was reused after exact state verification or evaluated afresh. Results are means over the same test population across seeds, not three independent test sets.

Open Results.html to view without installing software. To reproduce, extract this folder beside failure_diagnosis and external_test from the previous packages. Restore target-70 checkpoints and test images. Run train.py --seed SEED for each seed, then evaluate_target.py --seed SEED --epoch E for each seed and endpoint in a fresh output directory layout; archive original outputs first. Weights for epoch 12 are supplied separately.

Data: KolektorSDD2, Bozic, Tabernik and Skocaj, 2021, https://www.vicos.si/resources/kolektorsdd2/ (CC BY-NC-SA 4.0).

## Changes in individual decisions
Counts below compare the same images, at the fixed argmax decision rule. Each row is one paired run; do not sum across seeds as independent specimens.
| Seed | Loss | Defects recovered | Newly missed defects | Normal false alarms avoided | New normal false alarms |
|---|---|---:|---:|---:|---:|
| 20261028 | ce | 7 | 9 | 82 | 24 |
| 20261028 | poly1 | 8 | 6 | 74 | 33 |
| 20261029 | ce | 15 | 4 | 24 | 24 |
| 20261029 | poly1 | 8 | 1 | 24 | 31 |
| 20261030 | ce | 5 | 13 | 58 | 12 |
| 20261030 | poly1 | 9 | 9 | 42 | 32 |

## Reproduction of the 3-epoch control
The old and new runs used the same initialization and first three data orders. Exact state matches: 0/6. Where weights differed, the new control was evaluated, not substituted by the prior 3-epoch score. See baseline_reproduction.json for per-run class-decision differences; the cause of any numeric differences between executions was not isolated.

## Main finding
Extending training improved Macro-F1 in all six paired runs: mean gains were 4.91 percentage points for CE and 2.93 for Poly-1. For CE the main practical change was fewer false alarms (9.73% to 5.85% of normal images), with little change in defect recall (60.91% to 61.21%). Poly-1 gained 2.73 points of defect recall and also reduced false alarms. At 12 epochs CE led on Macro-F1, while Poly-1 detected more defects with more false alarms. Neither result proves general superiority or an optimal stopping epoch.