# Industrial Surface Defect Classification

**Student:** YASSER IDBOUZKRI  
**Group:** иикб-24  
**Course:** AI Implementation Technologies  
**Dataset:** KolektorSDD2  
**Final date:** 12 November 2026

## Submission

- `REPORT.pdf` — methodology, results, limitations and references.
- `PRESENTATION.pptx` — 12-slide summary of the completed work.

## Verified experimental scope

- **Study A:** 24 exploratory evaluations, with three seeds and 216 optimizer updates per run. The official test set had been inspected previously.
- **Study B:** eight completed 30-full-epoch runs, comparing Cross-Entropy and Poly-1 at 70, 140, 280 and 420 training images, using seed 20261028.
- **Validation:** a fixed 180-image holdout from the official training partition. At 420 training images, both losses achieved **98.00% best internal validation Macro-F1**, not independent test accuracy.

**Reproducibility:** [Source code, protocols and raw evaluation records](https://github.com/ID9999999999999/DL-Computer-vision/tree/nazih-training-set-size-20261109/COURSEWORK_NAZIH) · [Original data and saved model checkpoints](https://drive.google.com/drive/folders/1OI3zrAtmA9ZfwH3l_DB3IiwFBK2P1_ny).
