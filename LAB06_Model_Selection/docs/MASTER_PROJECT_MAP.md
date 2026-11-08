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
