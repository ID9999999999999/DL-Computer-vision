# Executed Project Plan — 13 November 2026

**Question.** With ResNet18 and a fixed train/validation split, does Poly-1 improve Macro-F1, defective recall, or false-positive rate over standard Cross-Entropy for binary surface-defect classification?

**Data.** Official ViCoS KolektorSDD2 images. Labels: Normal (0) and Defective (1). Nested, balanced training pools of 70, 140, 280 and 420 images. Original official test had been previously inspected; the new 30-epoch experiment therefore uses a fixed 180-image internal validation subset from unused official TRAIN images.

**Candidate research papers.** (1) Leng et al., *PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions*, ICLR 2022 — central tested idea: Poly-1 = CE + ε(1−p_true), ε=1. (2) Akhmedova and Körber, *Next Generation Loss Function for Image Classification*, arXiv 2024, https://arxiv.org/abs/2404.12948 — considered as an alternative, **not implemented**.

**Baseline.** ImageNet-pretrained ResNet18 + Cross-Entropy; RGB 448×448, AdamW lr 1e−4 and weight decay 1e−4, batch size 4.

**Control.** Paired CE/Poly-1 experiments use the same architecture, images, sample order, seed and hyperparameters within each image-count group. Study A is a fixed-216-update exploratory experiment with three seeds; Study B is a distinct eight-run, 30-full-epoch single-seed follow-up. Their scores are not directly comparable because the evaluation datasets differ.

**Metrics.** Macro-F1, defective recall, false-positive rate, accuracy and confusion matrix; AUROC for the original exploratory test records. Failure-case examples are regenerated using the saved individual predictions.

**Scope.** Original 24 evaluation records and eight new full-epoch model histories saved. No claim of fresh blind test accuracy or physical-part independence. Future multi-seed follow-up is optional, not completed work.

**Chronology note.** This document records the **executed** KSDD2 plan retrospectively. It does not claim that KSDD2 was the original data choice for the 30 October review, that the instructor approved the change, or that a live checkpoint was attended.
