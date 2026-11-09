# Model Card — Document Quality Classification (YOLOv8n-cls)

**Model:** YOLOv8n-cls  
**Project:** Lab 06 — Document Quality Classification  
**Baseline experiment:** 25 October 2026  
**Card updated:** 9 November 2026  
**Seed:** `2473986`

## 1. Intended use

The classifier labels a document photograph as:

- `good` — suitable capture for a document-processing workflow;
- `bad` — degraded by blur, glare, skew, occlusion or exposure problems.

Its intended role is an initial quality check before a downstream OCR workflow. It is a **coursework/research model**, not a validated production system. It does **not** verify document authenticity, recognize identity or guarantee OCR success.

## 2. Model selection

Four pretrained classifiers were compared under the recorded experimental protocol. **MobileNetV2 had the highest test accuracy (269/270, 99.63%)**. **YOLOv8n-cls was selected** using the predefined **validation quality and CPU cost rule**, not by searching for the highest test score. The selected model is the subject of this card.

The YOLOv8n-cls checkpoint has **1,440,850 parameters**. The model was initialized from a pretrained classification model and fine-tuned for this task.

## 3. Data and training

| Item | Description |
| --- | --- |
| Source data | **1,800 synthetic document images** derived from **300 source documents** |
| Document types | Passport, INN and SNILS |
| Train / validation / test | **1,260 / 270 / 270** images |
| Split rule | Source-grouped: variants of one source document remain in the same partition |
| Input | RGB images resized to **224 × 224** |
| Training | Batch size **16**; maximum **15** epochs; patience **4**; **11 epochs executed** for this model |
| Seed | **2473986** |
| Decision threshold | `P(bad) >= 0.50` |

The recorded settings differ from a later written configuration (image size 640, batch 32, max 50 epochs, patience 5). The historical settings and outcomes are reported unchanged; the instructor verbally indicated that these differences were acceptable.

## 4. Measured results

**Original Dataset A held-out test: 270 images (135 `good`, 135 `bad`)**

| Metric | Value |
| --- | ---: |
| Accuracy | **98.89% (267/270)** |
| Macro F1 | **0.9889** |
| Precision for `bad` | **1.0000** |
| Recall for `bad` | **0.9778** |
| Confusion matrix (true/predicted: good, bad) | **[[135, 0], [3, 132]]** |
| Mean CPU inference-pipeline latency | **26.97 ms/image** |
| Saved checkpoint size | **2.826 MiB** (2.963 MB) |

The original CPU measurement used four threads, 20 warm-up calls and 100 timed calls. The timed pipeline included preprocessing and forward inference, but excluded model loading, disk reading and image decoding. Latency depends on hardware and software environment.

There were **three defective images classified as good**, a relevant safety limitation for an automated quality gate.

**ONNX export (separate Task 5D check):** on the same 270-image test set, PyTorch and ONNX both achieved **267/270**, with **zero differences in predicted classes**. A same-host, forward-only CPU benchmark measured **6.104 ms** for PyTorch and **2.451 ms** for ONNX. These times have a different scope from the original **26.97 ms/image** pipeline latency and should not be compared directly.

## 5. Limitations and generalization

- The evaluation used **synthetic images** and only **45 distinct source documents** in the held-out test partition (270 generated variants).
- No independent real-world dataset evaluation was completed for this checkpoint. Its real-world generalization gap is therefore **unknown**.
- The course capstone references a **2.8–5.8 percentage-point synthetic-to-real gap**. That range is background context from the course, **not a measured result for this model**.
- Glare appeared among the original test failures. Unseen cameras, backgrounds, document layouts and lighting could change performance.
- The model should not be used as the sole basis for rejecting an important document.

## 6. Ethical and privacy considerations

Document photos may contain sensitive personal information. Use authorized datasets, restrict access to images, avoid publishing identifiable documents and retain data only as necessary. Incorrect predictions should allow manual review or a new photograph rather than an irreversible decision.

## 7. Evidence and related work

- [Original experiment report](../results/Final_Report.md) and [evaluation records](../results/yolov8n/evaluation.json)
- [Selection and comparison](week8_decision_memo.md)
- [Task 5C: glare augmentation](bonus_augmentation.md) — no additional glare recall over its matched control
- [Task 5D: ONNX verification](bonus_deployment_memo.md)
- [Task 5E: classmate reproducibility audit](bonus_audit_note.md)

The classmate's model achieved **211/270** on our test set, but that **is not** a transfer-performance result for our own checkpoint. No new evaluation of our checkpoint on the classmate's original dataset is claimed.
