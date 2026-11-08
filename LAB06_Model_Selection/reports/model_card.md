# Model Card — YOLOv8n-cls (Lab 06)
**Prepared:** 2026-11-09 | **Status:** experimental / provisional, not production-approved.

## Intended use and excluded uses
Binary **good/bad** quality screening for synthetic document images before downstream processing. This is an image classifier, not an object detector, OCR system, identity verifier, or an authenticity classifier. It is **not validated for automated legal, financial, or identity decisions**.

## Model and training data
- Architecture: Ultralytics YOLOv8n classification variant, ImageNet-pretrained, full fine-tuning.
- Train: 1,260; validation: 270; held-out test: 270 images (total 1,800).
- Source grouped across 300 underlying documents, with 210/45/45 sources in train/validation/test.
- Document families: INN, passport, SNILS; synthetic good and bad captures.
- Positive class for reporting: **bad**. Class mapping in the saved checkpoint: `{0: bad, 1: good}`.
- Student random seed: 2473986. Original input: RGB 224 × 224.
- Historical completed training: max epochs 15, batch 16, patience 4, stopped at 11 epochs, best checkpoint selected by validation fitness. This **differs** from later instructor protocol (max 50 / batch 32 / patience 5 / YOLO 640).

## Held-out test evidence
- Accuracy: **267/270 = 98.89%**.
- Macro-F1: **0.9889**.
- Positive (bad) recall: **132/135 = 97.78%**.
- Confusion matrix, rows actual [good,bad], columns predicted [good,bad]: `[[135,0],[3,132]]`.
- Errors: three defective **glare** images accepted as good; no good image falsely rejected in that held-out test.
- CPU benchmark: **26.97 ms/image** mean, measured in the original benchmark environment (20 warm-ups, 100 timed predictions, batch 1, four threads; already-decoded input).
- Model artifact: **2.83 MiB**, **1,440,850** recorded parameters. Original best.pt on Google Drive: https://drive.google.com/file/d/1htkgf8oiox9QT-s-JWnvxJxRKJTpPNiL/view

## New external evaluation evidence
On 9 Nov 2026, a **different student's YOLOv8n checkpoint** was evaluated on our 270-image Test: 211/270 accuracy (78.15%), macro-F1 0.7705. This is **not a cross-dataset test of our own model**. Testing our checkpoint on the other student's held-out data is still pending. See [peer evaluation report](cross_dataset_evaluation_2026-11-09.md).

## Limitations and responsible handling
The images are synthetic and only 45 underlying sources are held out, so apparent per-image precision overstates independent evidence. Dataset composition and some defect types may not represent real-world smartphone captures. Error costs are asymmetric: bad-as-good can pass unreadable images downstream. No deployment threshold was tuned on Test. The model may incorrectly reject or accept documents in unfamiliar conditions. Do not publish actual personal documents or images without permission; keep any real documents and checkpoint exchange permissions controlled. The peer model's original training image corpus has not been independently audited.

## Reproducibility and provenance
Original evaluation, training logs, predicted probabilities, confusion matrix, latency samples and seed are under `results/yolov8n/`; source code is `lab06.py`, original checkpoint remains external in Drive. Training ran on 25 Oct 2026. Later JSON compatibility views were **derived from original measurements**, not evidence of another training run. A repeat end-to-end Colab execution has not been verified.
