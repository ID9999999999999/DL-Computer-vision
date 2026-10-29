# 02 - PROJECT PLAN
Course Work Nazih | Last updated: 29 October 2026

## Goal
Supervised binary image classification of M2AD Motor images using one RGB image per decision.

- Class 0 = Normal
- Class 1 = Defective
- Main baseline = fully fine-tuned ResNet18 + Cross-Entropy
- Paper extension = identical ResNet18 + Poly-1 Cross-Entropy (epsilon = 1.0)
- Research question = Does Poly-1 improve validation macro-F1 or defective recall compared with Cross-Entropy when only the loss changes?

## Dataset and split
M2AD Motor; single illumination I01 (A000).
Existing development pilot: 99 images, including 50 Normal and 49 Defective.
Training: 70 images (35 Normal, 35 Defective).
Validation: 29 images (15 Normal, 14 Defective).
Resize RGB to 448 x 448, ImageNet normalization, no augmentation. Identical split and model initialization for both losses.

IMPORTANT: This is a DEVELOPMENT comparison, not a blind final evaluation. Motor images were previously inspected; physical part identity independence has not been proven. Source-image SHA256 checks and group isolation are mandatory.

## Known verified findings
- Anomaly detection experiments: frozen ResNet18 and adapted AnomalyDINO; dual-illumination residual fusion; improvement over Max not statistically established. These are historical supplementary studies, not the main two-class classifier.
- Supervised ResNet18 Cross-Entropy pilot: validation accuracy 28/29 = 96.55%; macro-F1 96.54% (NOT final independent test).
- Poly-1 mathematics implemented; code tests passed. Matched CE versus Poly-1 training has NOT been run yet.
- Previous environment lacked 95 of 99 source images; the notebook downloads only required images with exact SHA256 verification.

## Two paper candidates
1. Preferred: Leng et al., "PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions", ICLR 2022.
   Idea: L = CrossEntropy + epsilon * (1 - p_true).
   https://openreview.net/forum?id=gSdSJoenupI
2. Backup (NOT implemented): Akhmedova & Körber, "Next Generation Loss Function for Image Classification", arXiv 2024.
   https://arxiv.org/abs/2404.12948

## One code file
**03_ALL_CODE.ipynb**: run all cells from top to bottom in GPU-enabled Google Colab.
The single notebook holds ALL operative code in order, including pinned manifest/configuration, range-based source-image retrieval, SHA256 verification, ResNet18 baseline, Poly-1, two matched training runs, metrics, and export.

No external Python modules from this project's deleted old layout are needed. Original images are retrieved from the licensed dataset source; model pretrained weights are retrieved by torchvision. Output models, downloaded data, and result ZIP stay in the Colab runtime, NOT in GitHub.

## Remaining tasks in order
1. Run the single notebook with an internet connection and GPU; verify exact 99 original images.
2. Train both variants under the same settings; save authentic metrics, confusion matrices and errors.
3. Check leakage/duplicates and verify genuine part identity independence.
4. Secure a suitable independent holdout (do not relabel the already exposed test as blind).
5. Only later prepare the final paper/report and presentation.

## Professor milestones
30 Oct: plan, two candidate papers, three-minute project overview.
6 Nov: dataset and working supervised baseline.
13 Nov: paper loss idea and first controlled comparison.
20 Nov: results, error examples, report progress.
27 Nov: final experiments and report draft.
3 Dec: internal submission target.
4 Dec: formal defense (submission before meeting).

## Official sources
M2AD: https://github.com/hustCYQ/M2AD
Data: https://huggingface.co/datasets/ChengYuQi99/M2AD
Course project note: professor's slides (23 October 2026).

Do not claim a result for Poly-1 until the notebook has actually finished training.
