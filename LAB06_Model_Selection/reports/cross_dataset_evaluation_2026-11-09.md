# Lab 06 — Cross-Dataset Evaluation (2026-11-09)

**Status:** Executed on the original 270-image Dataset A test split. No retraining.

## Experiment

Test the classmate's trained YOLOv8n classification model on our independent Dataset A test images, using fixed P(bad) >= 0.5.

- 270 test images: 135 good, 135 bad. All 270 image SHA-256 hashes agree with the original manifest.
- 45 source groups in Test; no group overlap with the original Train or Validation splits.
- Our and peer checkpoints are YOLOv8n classification models, labels {0: bad, 1: good}; each checkpoint's stored transforms applied.
- CPU inference used a restricted PyTorch compatibility adapter because Ultralytics was unavailable offline. **Our own model's results reproduced the archived 267/270 result and confusion matrix exactly**; no separate official-Ultralytics rerun of the peer model was made.

## Metrics

| Metric | Our YOLOv8n | Peer YOLOv8n |
| --- | ---: | ---: |
| Test accuracy | **98.89% (267/270)** | **78.15% (211/270)** |
| Macro F1 | **0.9889** | **0.7705** |
| Precision (bad) | 1.0000 | 0.6959 |
| Recall (bad) | 0.9778 | 1.0000 |
| False alarms (good → bad) | 0 | 59 |
| Missed defective images (bad → good) | 3 | 0 |
| Total errors | 3 | 59 |

Confusion matrices are ordered **rows=true [good,bad], columns=predicted [good,bad]**:

- Our model: `[[135, 0], [3, 132]]`.
- Peer model: `[[76, 59], [0, 135]]`.

Of the peer model's 59 false alarms, **45 were INN clean images** and **14 were SNILS clean images**. It had no false alarms for the 45 clean passport images.

## Interpretation and cautions

This is evidence of reduced performance of the peer's checkpoint on **our** synthetic document test distribution, not evidence that the peer's model performs poorly on its own dataset. The peer's original training images have **not** been supplied, so independence from those images cannot be proven through an audit. The test set was not used for threshold tuning.

## Artifacts

- [Complete results and scripts ZIP in Google Drive](https://drive.google.com/file/d/1CsRe-t3hDA8IH1ar4pQs8-zKg16JtKTF/view)
- [Evaluation Drive folder](https://drive.google.com/drive/folders/1UDY10ahCXKvZsvkuA4vt3UjjRdYB5O0y)
- [Our model package to share with peer](https://drive.google.com/file/d/1z6TeWfS5vy6tc0PDvZFl_3F2CIyvzINk/view)

Contents of ZIP: image-level predictions and probabilities, each model's errors-only CSV, per-class metrics and confusion matrices, test-split integrity checks, figures, and evaluation scripts. **The opposite direction remains pending:** our model evaluated by the classmate on their independent Test.
