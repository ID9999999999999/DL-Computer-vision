# Image Classification Research Project — 30 October 2026

## Project title
**Supervised Binary Classification of Industrial Motor Defects: A Controlled Comparison of Cross-Entropy and Poly-1 Loss**

## Problem and classes
Given a **single RGB image** of an industrial motor, predict one of **two classes**:

- **Class 0:** Normal
- **Class 1:** Defective

This is a two-class **supervised image classifier**. It is separate from our earlier multi-illumination anomaly detection experiments.

## Dataset and protocol
- **Dataset:** M2AD, Motor category (Cao et al., Pattern Recognition, 2026); public and annotated for anomaly detection, repurposed into a two-class supervised problem.
- **Image configuration:** A000 / I01, one image per decision.
- **Existing development sample:** 99 images (50 Normal / 49 Defective); train 70 (35 / 35); validation 29 (15 / 14).
- **Preprocessing:** 448×448 RGB bicubic resize; ImageNet normalization.
- **Known risk:** Previous Motor images and old final-test subsets have already been inspected. Current grouping by numeric suffix does NOT prove physical-specimen independence. This split is only an **exploratory development split**, not a valid untouched final test.

## Baseline (completed pilot)
- **Architecture:** torchvision ResNet-18 with ImageNet initialization and 2 output logits.
- **Training:** all layers fine-tuned; Cross-Entropy loss, AdamW, learning rate 0.0001, weight decay 0.0001, batch 4, 3 epochs, seed 20261028.
- **Decision:** argmax of logits; predictions are Normal or Defective.
- **Reported preliminary validation:** accuracy 28/29 = 96.55%; macro-F1 = 0.96535; confusion matrix (rows true, cols predicted): [[15,0],[1,13]]. **These are not independent test metrics.**

## Candidate research papers (two)
1. **Leng et al. — “PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions”** (ICLR 2022, peer-reviewed). **Idea:** add the Poly-1 correction to ordinary Cross-Entropy, `L = CE + ε(1 - p_true)`. We propose ε=1 without tuning on the previously inspected test. This is the **preferred primary idea** because it changes only the loss and can be compared fairly against the baseline. Link: https://openreview.net/forum?id=gSdSJoenupI
2. **Akhmedova & Körber — “Next Generation Loss Function for Image Classification”** (arXiv 2024 preprint). **Idea:** replace CE with a loss function identified by genetic programming. **Limitation:** reproduction complexity is higher; the original paper's formula and implementation must be checked before proceeding. This is **a candidate only**, not a method we have implemented or tested. Link: https://arxiv.org/abs/2404.12948

**Classical alternative, not one of the two main candidate papers:** Lin et al., *Focal Loss for Dense Object Detection* (ICCV 2017). It is older and addresses detection, so confirm acceptability before proposing it as a main course paper.

## Controlled experiment
- **Experimental question:** Does Poly-1 change macro-F1 or Defective recall compared with CE when architecture, initialization, data split, optimizer, preprocessing, and training budget are matched?
- **Change exactly one factor:** CE → Poly-1 loss. Train the two models from the same deterministic initialization.
- **Validation reporting:** confusion matrix, class-wise precision and recall, macro-F1, accuracy, per-image predictions, example failures, and potential false positives.
- **Safety checks:** image SHA256 integrity; group overlap; near-duplicates; class balance; split contamination; selection of epochs based only on validation; no after-the-fact tuning against old final-test performance.
- **Interpretation:** do not claim causal improvement without a completed matched run; do not claim an independent test result without a separate verified specimen-level holdout.

## Status as of 29 October 2026
**Completed:** dataset inspection; supervised ResNet18 CE baseline; code/weights/manifest/predictions; Poly-1 implementation; six unit tests passed; 4-image checkpoint reload and inference sanity check.

**Not completed:** 95 original images are still missing from current runtime; matched CE-vs-Poly1 training not executed; independent confirmatory evaluation not available. The images are available from a pinned Motor.zip online source; downloading requires a network-enabled environment.

## Immediate presentation plan — 30 October
1. Show the included **single slide**.
2. Explain normal vs defective, the single-image input, and the baseline already run.
3. Introduce the two papers and explain why PolyLoss is preferred.
4. Explicitly ask the instructor whether the M2AD binary reformulation and chosen PolyLoss paper are approved.
5. Show the old supervised checkpoint and the code for the controlled experiment, stating honestly that the Poly-1 comparison is pending.

## Milestones
- **30 Oct:** plan, 2 paper candidates, one progress slide and 3-minute update.
- **6 Nov:** prepared data and baseline run evidence.
- **13 Nov:** paper idea implemented and first matched controlled comparison.
- **20 Nov:** metrics, errors, limitations, initial report.
- **27 Nov:** final experiments and full report draft.
- **3 Dec:** internal report/code deadline.
- **4 Dec:** instructor's formal defense; submit final code and B-BRICS report before the meeting.

## Sources
- M2AD official repository: https://github.com/hustCYQ/M2AD
- M2AD dataset (Motor archive): https://huggingface.co/datasets/ChengYuQi99/M2AD
- PolyLoss (ICLR 2022): https://openreview.net/forum?id=gSdSJoenupI
- Next Generation Loss (2024 preprint): https://arxiv.org/abs/2404.12948
