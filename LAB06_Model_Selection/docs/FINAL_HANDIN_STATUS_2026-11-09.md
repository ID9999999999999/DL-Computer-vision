# Lab 06 — final hand-in readiness (9 November 2026)

**Status: Prepared for review, NOT yet formally submitted or instructor-approved.** This is an evidence map, not a certificate of completion.

## Where to find the required material

- [Single master Google Drive](https://drive.google.com/drive/folders/105nt1H9wJsG-32jFrBHRqTDyB01USqta) — 00–09 organized folders.
- [Task 1 Word memo](https://drive.google.com/file/d/1bbrB9GnlR-xrOtEAjCMAp_h5EUQbSOJC/view) — converted from 19 Oct dated Markdown; transcription is not proof of on-time submission.
- Task 2: `results/mobilenetv2_frozen/results.json`, `results/mobilenetv2/curves.png`, `reports/baseline.md`. Checkpoint remains in original Drive.
- Task 3: `results/week7_yolov8n_cls/results.json`, `results/week7_yolov8s_cls/results.json`, `reports/week7_comparison.md`; **new** `scripts/train_yolo_quality.py` meets expected filename, and is a NEW rerun launcher rather than the actually executed historical script.
- Task 4: `results/comparison_from_json.csv` (derived only from the four verified results.json files), `reports/comparison.md`, `reports/week8_decision_memo.md`, `presentation/Lab06_Presentation_4slides.pptx`.
- Task 5A: `reports/bonus_finetune_delta.md`, results and checkpoint in original Task5A Drive folder; 99.63% before/after.
- Optional Task 5B: `reports/bonus_threshold_memo.md`, validation-only sweep CSV/PR curve. **Threshold 0.10 is an unapproved validation candidate**, not deployment policy.
- Optional Task 5E: `reports/model_card.md` and partial `reports/bonus_audit_note.md`.
- Classmate exchange: [our YOLOv8n share ZIP](https://drive.google.com/file/d/1z6TeWfS5vy6tc0PDvZFl_3F2CIyvzINk/view) and [peer-on-our-test results](https://drive.google.com/file/d/1CsRe-t3hDA8IH1ar4pQs8-zKg16JtKTF/view).
- Student explanations/defense: `docs/Defense_Notes.md`, `presentation/Speaking_Script.md`, and [four-slide deck](../presentation/Lab06_Presentation_4slides.pptx).

## Confirmed unresolved blockers, not concealed

1. **Peer's own held-out Dataset B results have NOT arrived.** Our own model has not been tested on that domain. Do not report a two-way exchange result.
2. Tasks 2–4 **historical training deviations**: max epochs 15 rather than 50, batch 16 vs 32, patience 4 vs 5 and YOLO resolution 224 vs 640. New launcher does not remove this discrepancy.
3. Exact original executed training notebook and teacher's oral in-class defense/on-time acceptance are **not verified**. The review notebook reproduces analysis, not training.
4. Genuine historical Git tags remain absent: connection has no tags-creation action; [tag instructions](TAGGING.md). No fake historical tag was created.
5. The Task 5B candidate and Task 5E peer audit are bonus work, not proof of successful external validation or teacher acceptance.
6. At most 1–3 real test errors/model occurred; task guidance asking 5 distinct error images cannot be satisfied without inventing errors.

## Safe submission procedure

Show the four-slide presentation and GitHub branch to the instructor, link the Drive master, upload Task 1 Word file where required, and explicitly call out deviations/remaining pending peer test. Record the actual teacher acknowledgment once it occurs; do not mark it complete in advance.


## Current 152-point audit — 9 November 2026

- [Updated Markdown checklist](FULL_CHECKLIST_CURRENT_2026-11-09.md) — verified **100 green / 25 yellow / 27 red**, counting optional bonus tasks.
- [Updated ten-page PDF in Drive](https://drive.google.com/file/d/1WcQMktdHFhZoAA6aFcGUDLVGKWnFhD60/view); [Markdown copy in Drive](https://drive.google.com/file/d/1QVYsMvIponyR7Tel9B8dKKsrSjkY14ey/view). The 8 Nov snapshot remains intact.
- [Full zipped submission REVIEW package including FOUR ORIGINAL CHECKPOINTS (SHA-256 verified), the Task 5A checkpoint package, Word memo, four slides, experiment evidence and scripts](https://drive.google.com/file/d/1l6LKs3tCjIoetjxnl5uklINYCaLteAO0/view). **Prepared for review, NOT submitted or accepted.**


## 9 November final verification addendum

The **152-item checklist is now 100 green / 25 yellow / 27 red**, recounted from individual markers; the earlier 85/22/45 statistics are superseded. Both Week 7/Week 8 Git tags are **created and verified** on the same authentic October 25 experiment commit; see [TAGGING](TAGGING.md). Tasks 5D ONNX native-vs-exported accuracy parity and CPU benchmark have been **executed on 270 original Test images**; the unchanged peer inference README script was run for Task 5E, but the peer's original training/holdout and reciprocal Dataset B test remain unavailable. No teacher acceptance is implied.


## Final consolidated academic review archive — 9 November 2026

**Canonical, checksum-verified 78,526,618-byte package:** [Lab06_COMPLETE_SUBMISSION_REVIEW_FINAL_2026-11-09.zip](https://drive.google.com/file/d/1ZAp0JdWW_DruPzZrG9ll96Z-Idjr1-4D/view). ZIP SHA-256: `47962e1536ff6ce5fad1bf1074cb9dcfea6612b5ce674b7fca3155fef0069dd1`. It contains all four original model checkpoints, the executed Task 5A, 5C, 5D and 5E evidence packages, Task 1 Word memo, four-slide presentation, latest source/results snapshot and a 24-file SHA-256 manifest. The nested source snapshot is from commit `b5e2f5c2937dcd3fee249b5eaa13a78931443b0f`; later link updates to this file are navigational only.

**Final 152-point checklist:** [PDF](https://drive.google.com/file/d/1WcQMktdHFhZoAA6aFcGUDLVGKWnFhD60/view), [Markdown](https://drive.google.com/file/d/1QVYsMvIponyR7Tel9B8dKKsrSjkY14ey/view): **100 completed / 25 partial / 27 open**. Full backup in the ZIP. Original Dataset A images (about 399 MB) are referenced but not bundled.

**Open dependencies:** classmate's independent Dataset B reciprocal evaluation, original training settings (15/16/4/224 vs later 50/32/5/640), original executed Colab training notebook verification, and the instructor's oral presentation / acknowledgment. Two verified Week7/8 tags point to one authentic Oct 25 commit; no backdating. **Archive creation is not formal submission or acceptance.**
