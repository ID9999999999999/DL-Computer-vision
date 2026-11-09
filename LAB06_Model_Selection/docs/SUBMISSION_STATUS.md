# Lab 06 — Submission Status

**Reviewed:** 9 November 2026  
**Project:** Document Quality Classification (GOOD / BAD)  
**Seed:** `2473986`  
**Current state:** Experimental work prepared; formal course hand-in and oral presentation not confirmed.

This page is the **single current status record**. The [requirement-by-requirement audit](INSTRUCTOR_REQUIREMENTS_AUDIT.md) provides supporting evidence. Older dated checklists and progress notes are historical snapshots, not current action lists.

## Main links

- [Project README](../README.md): methodology, reproduction and full results.
- [Original instructor documents](https://drive.google.com/drive/folders/1pRFQQHEpripLdUE4sx6Qd0sqHW7UjD7o): evaluation guide and Tasks 1–5.
- [Organized Drive master](https://drive.google.com/drive/folders/105nt1H9wJsG-32jFrBHRqTDyB01USqta): source data, original weights and task evidence.
- [Submission folder](https://drive.google.com/drive/folders/16ioAY5oTkkspCM5iKLnHtJ3XEbKKb_ml): Word memo, final four-slide deck and supporting archive.
- [Full report](../results/Final_Report.md) · [Four-model comparison](../reports/comparison.md) · [Task 4 decision](../reports/decision_memo.md).
- [Task 5E audit](../reports/bonus_audit_note.md) · [selected model card](../reports/model_card.md).

## Required work: Tasks 1–4

| Task | Evidence | Current status |
| --- | --- | --- |
| **1. Architecture selection memo** | [Dated memo](Model_Selection_Memo.md), [Word file](https://drive.google.com/file/d/1XmGj5Tse1OaRENXPBHbm1G0jrAvIjULp/view), 4 models, 6 references, predeclared numeric targets | **Written and staged**; original on-time receipt and oral answers unconfirmed |
| **2. MobileNetV2 frozen baseline** | [Report](../reports/baseline.md), [result JSON](../results/mobilenetv2_frozen/results.json), curves, confusion matrix, one authentic [error image](../results/mobilenetv2_frozen/errors/README.md) | **Experiment completed**; exact executed original Colab notebook not preserved here, but training/evaluation logs exist |
| **3. YOLOv8 classification** | [Comparison](../reports/week7_comparison.md), [nano](../results/week7_yolov8n_cls/results.json), [small](../results/week7_yolov8s_cls/results.json), [training launcher](../scripts/train_yolo_quality.py), [verified Week 7 tag](TAGGING.md) | **Experiments completed**; oral instructor check unconfirmed |
| **4. Third model and deployment decision** | [EfficientNetB0](../results/week8_efficientnetb0/results.json), [comparison](../reports/comparison.md), [decision memo](../reports/decision_memo.md), [4-slide deck](../presentation/Lab06_Presentation_4slides.pptx), [verified Week 8 tag](TAGGING.md) | **Technical material ready**; in-class 5-minute presentation and two questions not confirmed |

**Actual test results (270 held-out synthetic images):** MobileNetV2 **269/270** (99.63%); EfficientNetB0 **267/270** (98.89%); YOLOv8n-cls **267/270** (98.89%); YOLOv8s-cls **268/270** (99.26%). **YOLOv8n-cls** was selected by the predefined validation-and-cost rule, not by highest test accuracy.

## Optional work: Task 5

| Task | Evidence | Outcome |
| --- | --- | --- |
| **5A** Partial fine-tuning | [Delta report](../reports/bonus_finetune_delta.md) | Executed; MobileNetV2 stayed **269/270** |
| **5B** Threshold/cost analysis | [Validation-only memo](../reports/bonus_threshold_memo.md) | Executed; threshold 0.10 is a **validation candidate**, not a new test result |
| **5C** Glare augmentation | [Controlled experiment](../reports/bonus_augmentation.md) | Executed from original trained checkpoint; no extra glare recall vs matched control |
| **5D** ONNX export | [Deployment memo](../reports/bonus_deployment_memo.md) | Executed; PyTorch and ONNX **267/270 each**; forward-only CPU **6.104 vs 2.451 ms** |
| **5E** Peer audit + model card | [Audit](../reports/bonus_audit_note.md), [model card](../reports/model_card.md) | **Two documents prepared**; peer's original inference script ran unchanged on Dataset A (**211/270**), but original peer training/own-test was not reproduced |

The teacher describes Tasks 5A–5E as **bonus work**. Task 5E evidence is explicitly scoped to the available peer inference package.

## Resolved questions and genuine remaining actions

**No further training is scheduled.** The instructor verbally accepted the original Tasks 2–4 training-setting differences (**actual 15 epochs / batch 16 / patience 4 / YOLO input 224**, versus later written **50 / 32 / 5 / 640**). This acceptance does not alter the historical measurements.

The instructor also stated that a reciprocal independent **Dataset B** exchange is **not required**. It is **not an open submission blocker**. The old 152-item checklist counted 13 uncompleted dataset-exchange subtasks; those entries must not be interpreted as 13 mandatory missing tasks.

The only follow-up items requiring attention are:

1. **Final submission:** send the four-slide deck, Task 1 Word memo, GitHub project and supporting evidence via the instructor's required channel; preserve the receipt.
2. **Oral defense:** present the four slides (five minutes), answer at least two Task 4 questions and, if asked, explain the three Task 1 concepts.
3. **Original training execution evidence:** the saved training/evaluation log is present, but the originally executed Colab notebook itself has not been recovered. Provide it if specifically requested.
4. **Failure gallery wording:** the teacher asks for five misclassified test images per model; the actual counts are **1, 3, 3, 2**. All authentic errors are linked; no additional errors can be supplied without falsifying results. Explain this discrepancy if asked.

Task 5E's full original peer-training reproduction and public redistribution permission are **unverified**, but do not block Tasks 1–4. No external real-photo evaluation has been claimed; it is a scientific limitation, not an assigned experiment.

## Drive ↔ GitHub: which copy is authoritative?

- **GitHub (this branch):** current readable reports, code, results summaries, decisions and task audit.
- **Drive original folders:** original private dataset, four original model checkpoints, error-image evidence, optional experiment ZIPs and the submission slides/Word memo.
- **Drive evidence ZIP:** [78.5 MB verified review snapshot](https://drive.google.com/file/d/1ZAp0JdWW_DruPzZrG9ll96Z-Idjr1-4D/view) (SHA-256 `47962e1536ff6ce5fad1bf1074cb9dcfea6612b5ce674b7fca3155fef0069dd1`). Its embedded source snapshot is **older than the latest GitHub documentation edits**; use the live GitHub branch for current text. This ZIP is supporting evidence, **not a live synchronized source copy or official hand-in receipt**.

The original raw measurements, source images, and saved weights are preserved. No test-set tuning or fabricated error images have been introduced.
