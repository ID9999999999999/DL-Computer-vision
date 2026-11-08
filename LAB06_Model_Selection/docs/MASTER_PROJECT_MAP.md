> **Current Drive organization (9 Nov 2026):** [START HERE](https://docs.google.com/document/d/1Pv5PZu81vR2gaNotgyRmq1Q3MaGgj3PtHheSu2wBfYU/edit) / [RESUME LATER](https://docs.google.com/document/d/1QT9NeGwCxnpNiH2KiIMf6Peiipf8th8AfVnRJwM5eKQ/edit) / [current folder map](DRIVE_FOLDERS_2026-11-09.md). Organized folders 10–13 now correspond to Bonus 5B–5E. Earlier summary paragraphs are historical and do not override this handoff.

# Lab06 — Canonical Project Map

**Unified Drive project folder:** https://drive.google.com/drive/folders/105nt1H9wJsG-32jFrBHRqTDyB01USqta

## Organization

| Drive folder | Contents and original references |
|---|---|
| 00_INSTRUCTIONS | Four original teacher documents copied from the instructor source folder |
| 01_DATASET_A | Metadata and links to the 1,800-image source dataset ZIP and prepared split |
| 02_TASK_1_MODEL_SELECTION | Link to `docs/Model_Selection_Memo.md` |
| 03_TASK_2_MOBILENETV2 | Baseline curves and report links; original frozen `best.keras` remains in Drive |
| 04_TASK_3_YOLOV8 | YOLOv8n/s evidence and model results; original `best.pt` weights remain in Drive |
| 05_TASK_4_COMPARISON | Four-slide PPTX and comparison figures |
| 06_TASK_5A_FINE_TUNING | Full ZIP, report, curves and links to the trained fine-tuned checkpoint |
| 07_DATASET_EXCHANGE_PENDING | Our YOLOv8n sharing package and the peer-model-on-our-Test evaluation (2026-11-09); peer's actual Dataset B is still not received |
| 08_FINAL_SUBMISSION | Curated presentation and submission checklist; not formally handed in |
| 09_ARCHIVE_AND_ORIGINALS | Source pointers and originals preservation records |
| 11_BONUS_TASK_5C_AUGMENTATION | Real two-arm CPU fine-tuning, per-defect analysis, exact checkpoints and reproducibility ZIP |

## Source-of-truth links

- Original dataset ZIP: https://drive.google.com/file/d/1QNHsbtpWcmeo0asG2Mrrruwy5w2lmobB/view
- Original experiment Drive: https://drive.google.com/drive/folders/1ja5uljMIcMSNLd41scx2HHV5B0SLallu
- Original Task 5A evidence: https://drive.google.com/drive/folders/1krJ2b7SXPcHBnkDQHajKBq5ZEC1fAYAP
- Task 5A report: https://github.com/ID9999999999999/DL-Computer-vision/blob/lab06-model-selection-2473986-20261025/LAB06_Model_Selection/reports/bonus_finetune_delta.md
- Four-slide deck: https://github.com/ID9999999999999/DL-Computer-vision/blob/lab06-model-selection-2473986-20261025/LAB06_Model_Selection/presentation/Lab06_Presentation_4slides.pptx

All originals preserved. Historical Task 2–4 protocol differences remain: actual max epochs 15/batch 16/patience 4/YOLO 224px versus later detailed instructions 50/32/5/640px. Task 5A used original Dataset A and achieved no test improvement. **Peer model exchange and peer-on-our-test evaluation completed 2026-11-09. Actual classmate Dataset B images have not been received**, and our model has not yet been evaluated on the classmate's held-out images. Do not imply acceptance by instructor.

## Complete requirement checklist (8 November 2026)

- [FULL CHECKLIST (editable Markdown)](FULL_CHECKLIST_2026-11-08.md) — 152 items: 74 green, 19 yellow, 59 red. Counts include optional 5B–5E tasks.
- [Seven-page PDF in Drive](https://drive.google.com/file/d/1JBEnOTl9Nz2VYs8QY6756ynQhKfIPjBy/view)
- The same PDF is staged in `08_FINAL_SUBMISSION`, but **official instructor acceptance is not verified**.


## Peer model evaluation (9 November 2026)

- Shared our YOLOv8n model package: https://drive.google.com/file/d/1z6TeWfS5vy6tc0PDvZFl_3F2CIyvzINk/view
- Evaluated peer's YOLOv8n on our exact 270-image held-out Test (135 good / 135 bad, test hashes verified).
- Our YOLOv8n: 267/270 correct, accuracy 98.89%, macro-F1 0.9889; confusion matrix [[135,0],[3,132]].
- Peer YOLOv8n: 211/270 correct, accuracy 78.15%, macro-F1 0.7705; confusion matrix [[76,59],[0,135]].
- Complete evidence (predictions per image, scores, figures, reproducibility scripts): https://drive.google.com/file/d/1CsRe-t3hDA8IH1ar4pQs8-zKg16JtKTF/view
- **Next pending dependency:** classmate runs our model against an independent labeled Test of their data (or supplies that Test).
- Checklist counts cited above are an **8 Nov snapshot**, not a revised 9 Nov audit.


## 9 November submission-readiness update

- Task 1 Word-formatted memo now stored under Drive `02_TASK_1_MODEL_SELECTION` (not proof of historical hand-in).
- New `scripts/train_yolo_quality.py` and `scripts/collect_results.py` with synthetic utility tests; not a rerun of the original models.
- Derived comparison CSV in `results/comparison_from_json.csv` uses only four saved standardized results.json sources.
- Optional Task 5B validation-only threshold sweep, model card, and partial peer audit added.
- [Final hand-in readiness and blockers](FINAL_HANDIN_STATUS_2026-11-09.md) is the current working status.

## Additional submission preparation (9 November 2026)

- [Staged final ZIP (9 files)](https://drive.google.com/file/d/1dvrnRlEyXgiUDFyVZraOaWU-iyNSlX3o/view): four-slide PPTX, Task 1 Word memo, figures, peer evaluation, 8 Nov checklist snapshot, START_HERE, checksums.
- [Task 1 Word memo in Drive](https://drive.google.com/file/d/1XmGj5Tse1OaRENXPBHbm1G0jrAvIjULp/view) is **prepared but not proven submitted**.
- [Task 5B threshold analysis](../reports/bonus_threshold_memo.md) computed entirely from the existing 270 validation probabilities: 0.50 -> 268/270 correct; validation F1 optimum 0.10 -> 270/270 correct. **No test tuning or changed historic metrics**.
- [Model card](../reports/model_card.md), [current 4-slide script](../presentation/Final_4_Slide_Script.md), and [submission manifest](FINAL_SUBMISSION_MANIFEST_2026-11-09.md) are now available.
- Historical Git tags, original executed Colab notebook, teacher sign-off and dataset B reverse evaluation remain unverified. Bonus 5B implementation is not evidence of instructor acceptance.


## Current 152-point audit — 9 November 2026

- [Updated Markdown checklist](FULL_CHECKLIST_CURRENT_2026-11-09.md) — verified **100 green / 25 yellow / 27 red**, counting optional bonus tasks.
- [Updated ten-page PDF in Drive](https://drive.google.com/file/d/1WcQMktdHFhZoAA6aFcGUDLVGKWnFhD60/view); [Markdown copy in Drive](https://drive.google.com/file/d/1QVYsMvIponyR7Tel9B8dKKsrSjkY14ey/view). The 8 Nov snapshot remains intact.
- [Full zipped submission REVIEW package including FOUR ORIGINAL CHECKPOINTS (SHA-256 verified), the Task 5A checkpoint package, Word memo, four slides, experiment evidence and scripts](https://drive.google.com/file/d/1l6LKs3tCjIoetjxnl5uklINYCaLteAO0/view). **Prepared for review, NOT submitted or accepted.**


## Task 5C completed experiment addendum (9 November 2026)

- [Full experiment in Google Drive](https://drive.google.com/file/d/1HyxDmT2yJ7xX_wN4R76b0F8jREVrqfjM/view).
- [Report](../reports/bonus_augmentation.md) and [validation-vs-test honest audit](TASK5C_2026-11-09_AUDIT.md).
- 2 arms start from identical previously trained YOLOv8n; real continued fine-tuning on original Dataset A; original dataset/manifest and historical checkpoints untouched.
- Control: **268/270**, macro-F1 0.9926; train-only stronger glare: **267/270**, macro-F1 0.9889. Glare recall both 26/28, augmentation no benefit versus matched continuation, and occlusion declined.
- Task 5C formal protocol exact-retrain criterion remains **partial** because this is CPU/PyTorch continued training from a trained checkpoint, not an Ultralytics retrain from ImageNet and not 50/32/5/640.
- 8 Nov 152-item master checklist is an historical snapshot; later work documented as an addendum, not retroactively inserted into original results.

## Task 5D — completed paired Test after earlier cloud export

[Final ONNX + 270-image paired evaluation ZIP](https://drive.google.com/file/d/1zIEXnesFzi4JPrzxuNRbYaemqEUEbQ0w/view). Original PT and ONNX both **267/270 correct**, macro-F1 **0.9889**, zero individual prediction differences and zero accuracy drop. CPU same-host forward-only means **6.104 ms PT / 2.451 ms ONNX**, measured after 20 warm-ups over 100 timed calls each. The earlier 8.651/2.965ms values were on a *different GitHub cloud CPU* and are retained only as separate provenance. Teacher acceptance not evidenced.

## Task 5E — literal peer code executed

[Verified unchanged classmate `predict.py` execution logs/CSV](https://drive.google.com/file/d/1BmdQlk-aY5e97KGm7phdXk8P2pN6Aw2i/view) on our original 270-image Test: **211/270**, 59 false alarms, no false accepts; 0 prediction differences against older adapter. This is a README inference reproduction on our data, **not the peer's own original test/training reproduction**. The reverse evaluation of our model on the peer's Dataset B is still pending; [final audit](../reports/bonus_audit_note.md).


## Final verified 9-November checklist and Git tags

- [Updated 152-item checklist](FULL_CHECKLIST_CURRENT_2026-11-09.md): **100 green / 25 yellow / 27 red**, recounted from the individual status markers after the 5C, 5D and 5E experiments.
- [Verified historical evidence tag record](TAGGING.md) — both `lab06-week7-verified` and `lab06-week8-verified` reference the **same original 25 October commit**, as the two phases were first uploaded together.
- [Task 5D addon results row](../results/task5d_deployment_addon.json); actual ONNX/Test evidence in [private Drive ZIP](https://drive.google.com/file/d/1zIEXnesFzi4JPrzxuNRbYaemqEUEbQ0w/view).
- [Task 5E original peer README execution audit](../reports/bonus_audit_note.md); peer Dataset B reverse evaluation is still pending.


## Final consolidated academic review archive — 9 November 2026

**Canonical, checksum-verified 78,526,618-byte package:** [Lab06_COMPLETE_SUBMISSION_REVIEW_FINAL_2026-11-09.zip](https://drive.google.com/file/d/1ZAp0JdWW_DruPzZrG9ll96Z-Idjr1-4D/view). ZIP SHA-256: `47962e1536ff6ce5fad1bf1074cb9dcfea6612b5ce674b7fca3155fef0069dd1`. It contains all four original model checkpoints, the executed Task 5A, 5C, 5D and 5E evidence packages, Task 1 Word memo, four-slide presentation, latest source/results snapshot and a 24-file SHA-256 manifest. The nested source snapshot is from commit `b5e2f5c2937dcd3fee249b5eaa13a78931443b0f`; later link updates to this file are navigational only.

**Final 152-point checklist:** [PDF](https://drive.google.com/file/d/1WcQMktdHFhZoAA6aFcGUDLVGKWnFhD60/view), [Markdown](https://drive.google.com/file/d/1QVYsMvIponyR7Tel9B8dKKsrSjkY14ey/view): **100 completed / 25 partial / 27 open**. Full backup in the ZIP. Original Dataset A images (about 399 MB) are referenced but not bundled.

**Open dependencies:** classmate's independent Dataset B reciprocal evaluation, original training settings (15/16/4/224 vs later 50/32/5/640), original executed Colab training notebook verification, and the instructor's oral presentation / acknowledgment. Two verified Week7/8 tags point to one authentic Oct 25 commit; no backdating. **Archive creation is not formal submission or acceptance.**
