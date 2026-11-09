# Lab06 submission audit and corrections — 8 November 2026

**No models were retrained. Original measurement files are preserved.** The corrections organize saved evidence for instructor review.

| Problem identified | Fix | Status |
|---|---|---|
| No standardized results.json | Generated per-model JSON from real original evaluation, training, histories | Complete |
| Folder names differ | Added matching compatibility results/ and reports/ paths without deleting old | Complete (except missing original source script exact path) |
| Reports under different names | Added baseline, comparison, week7 comparison and five-part decision memo | Complete |
| Comparison omitted epochs run | Generated results/comparison_with_epochs.csv with actual epochs 15/15/11/7 | Complete |
| Eight original main slides, teacher asks four | New four-slide presentation script and editable PPTX | **Complete in `presentation/Lab06_Presentation_4slides.pptx`, GitHub and Drive; original retained** |
| Errors only in composite galleries | Copied nine genuine error JPEGs to private Google Drive model folders; created per-model errors manifests | Complete (Drive JPEGs; links in GitHub) |
| No Week 7/8 tags verified | Added docs/TAGGING.md with safe historical instructions | **Pending actual tag creation** |
| Decision memo structure differs | Added reports/week8_decision_memo.md with five required sections | Complete |

## Four differences in the existing training that cannot be corrected retroactively

| Parameter | Teacher's detailed requirement | Actual completed run |
|---|---|---|
| Maximum epochs | 50 | 15 |
| Batch size | 32 | 16 |
| Early-stopping patience | 5 | 4 |
| YOLO input | 640×640 | 224×224 |

Do not edit archived metrics or claim that the trained models satisfied a different protocol. Teacher approval or actual retraining would be necessary to establish literal compliance. Dataset exchange and the new partial fine-tuning assignment remain separate future work.

## Original evidence

`results/run.log`, `results/protocol.json`, every `evaluation.json`, `training.json`, raw probabilities, curves, plots, original report, GitHub commits and previously created slide deck remain untouched. All standardized JSON records carry a provenance statement; error images remain private in [Individual_Errors_Lab06](https://drive.google.com/drive/folders/1D157EQQNOM6LMtnOGUvuvVyjS0v8PuOv). The four-slide PPTX contains the three actual synthetic error images. The original repository does **not** contain raw input images or original executed Colab training notebook. 

## Post-repair update: Task 5A completed after this audit (8 November 2026)

A separate authentic MobileNetV2 partial fine-tuning experiment was completed; see [Task 5A report](../reports/bonus_finetune_delta.md). It used the instructor's Task 5A numerical LR, 50 max epochs, patience 5, seed 2473986 and original split; effective batch 32 was implemented with gradient accumulation of microbatches of 8 and the 30 chosen layers' BatchNorm remained frozen. Keras with PyTorch backend was used instead of the original TensorFlow/T4. Epochs run: 20, best: 15. Accuracy 99.63% and macro F1 0.9963 were unchanged versus frozen. The earlier Task 2–4 training-protocol deviations above remain historical facts; do not change them. Classmate dataset exchange is still pending.
