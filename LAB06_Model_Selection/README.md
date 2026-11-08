> **STOP POINT (9 November 2026): organization complete; experiments paused.**
> Start at [one current Drive index](https://docs.google.com/document/d/1Pv5PZu81vR2gaNotgyRmq1Q3MaGgj3PtHheSu2wBfYU/edit) and [resume-later handoff](https://docs.google.com/document/d/1QT9NeGwCxnpNiH2KiIMf6Peiipf8th8AfVnRJwM5eKQ/edit).
> Current source-of-truth checklist: [100 completed / 25 partial / 27 open](docs/FULL_CHECKLIST_CURRENT_2026-11-09.md) of 152; latest [review ZIP](https://drive.google.com/file/d/1ZAp0JdWW_DruPzZrG9ll96Z-Idjr1-4D/view) is **not** an official submission. [Folder map](docs/DRIVE_FOLDERS_2026-11-09.md).
> Older ZIPs/checklists have been archived, no original data/models/results modified.

# Lab 06: Model selection for document quality classification

Seed **2473986**. Experiment completed in Colab on 25 October 2026.

## Start here

1. Read [the final report](results/Final_Report.md).
2. Open [Review_Results.ipynb](Review_Results.ipynb) for executed verification and plots. It uses saved outputs and requires no GPU, images or model downloads.
3. Present [Lab06_Presentation_4slides.pptx](presentation/Lab06_Presentation_4slides.pptx) (four-slide instructor format) with the [current 5-minute script](presentation/Final_4_Slide_Script.md). The longer deck remains an archive. Speaker notes contain the English script.
4. Read [Model_Selection_Memo.md](docs/Model_Selection_Memo.md) for the pre-run choices and architecture references.

## Reproduce the experiment

The image dataset and model weights are external to this repository. Preserve the original `demo_quality_dataset` with `metadata.jsonl`, `good/` and `bad/`, and the four best model files in the original Drive output. The review archive does not contain them.

From this folder, install `requirements.txt` in a dedicated environment. Exact versions from the completed run are in `results/environment.json` and `results/environment.txt`. GPU packages may need platform-specific installation.

```bash
python -m pip install -r requirements.txt
python lab06.py run --dataset /path/to/demo_quality_dataset --out /path/to/NEW_lab06_results
```

Always use a **new output folder** for a new experiment. Do not point a training run at the packaged `results/` snapshot. `Lab06.ipynb` is the unexecuted training workflow. The original training happened in Colab through the same CLI and its evidence is in `results/run.log` and the model histories. The results-review notebook executes only analysis and displays the real saved plots.

## Main findings

MobileNetV2 achieves 99.63% test accuracy with one error out of 270 images. The predefined validation-and-cost rule selects YOLOv8n-cls, the fastest passing pipeline at 26.97 ms/image on CPU. All observed test mistakes involve glare. These are results on a small synthetic dataset, not proof of deployment robustness.

## Files

| Location | Purpose |
|---|---|
| `lab06.py`, `config.json` | Training, evaluation and fixed protocol |
| `Lab06.ipynb` | Unexecuted training workflow |
| `Review_Results.ipynb` | Executed analysis of completed results |
| `docs/` | Rationale, defense preparation and verification |
| `results/` | Measured metrics, raw probabilities, histories, logs and plots |
| `presentation/` | Editable slides and five-minute script |
| `tests/` | Protocol and selection tests |

Model selection uses validation metrics and cost only. Test characterization uses fixed threshold 0.5 and never triggers reselection. The original automatically generated report remains as `results/Final_Report_Original.md`. The revised report adds interpretations and links without changing measured results.


## Submission corrections — 8 November 2026

This additive update preserves the original experiment evidence. See [repair audit](docs/Submission_Repairs_2026-11-08.md), [corrected reports](reports/README.md), [comparison with executed epochs](results/comparison_with_epochs.csv), [four-slide script](presentation/Presentation_4slides.md), [individual error archive](results/yolov8n/errors/README.md) and [Git tagging instructions](docs/TAGGING.md).

The newly added `results.json` records were **derived from saved measurements, not retrained**. Actual training differed from the instructor's later specified epochs, batch size, patience and YOLO input size; see audit. The four-slide PPTX is provided separately from the original 8-slide deck.

[**Corrected four-slide PPTX**](presentation/Lab06_Presentation_4slides.pptx) — teacher's specified four-slide format, with original result plots and real synthetic failure examples. Original longer PPTX remains unchanged.

## Task 5A — completed partial fine-tuning (8 November 2026)

This is a genuine separate training run using the original MobileNetV2 frozen checkpoint and the **same original 70/15/15 source-grouped manifest**. We trained with Adam LR 1e-5, selected the top 30 backbone layers with BatchNorm frozen, effective batch 32 (four microbatches of 8), max 50 epochs and patience 5. Early stopping ran **20 epochs**, preserving the best checkpoint from epoch **15**. Test accuracy **99.63%** and macro F1 **0.9963**: **no improvement over frozen**. The working environment used Keras 3's PyTorch CPU backend, not the original TensorFlow GPU backend. The pretrained backbone layers outside the top 30 were verified unchanged.

- [Actual experiment report and delta table](reports/bonus_finetune_delta.md)
- [Actual results.json](results/mobilenetv2_finetune_top30/results.json)
- [Training curves](results/mobilenetv2_finetune_top30/curves.png) and [confusion matrix](results/mobilenetv2_finetune_top30/confusion_matrix.png)
- [Executable training script](scripts/train_task5a.py) and [Colab notebook](Task5A_FineTuning_Reproducible.ipynb)
- [Training log](results/mobilenetv2_finetune_top30/train.log) and [verification record](docs/Task5A_Verification.json)
- [New private Google Drive experiment folder](https://drive.google.com/drive/folders/1krJ2b7SXPcHBnkDQHajKBq5ZEC1fAYAP) — contains best.keras, original image error, JSON, scripts, plots and complete ZIP. The original frozen checkpoint remains in Drive unchanged.

**Not the dataset-exchange exercise:** this Task 5A run uses our own original dataset. Receiving and fine-tuning on a classmate's dataset remains a separate task.


## ONE PROJECT ENTRY POINT — Organized Drive master (8 November 2026)

Open [LAB06_MASTER_2026-11-08](https://drive.google.com/drive/folders/105nt1H9wJsG-32jFrBHRqTDyB01USqta) first. It contains numbered folders 00–09, a `START_HERE_MASTER_INDEX.md`, original teacher instructions (copies), the curated four-slide deck, completed Task 5A package and an official-submission staging folder. Original datasets, models and past records were not moved or deleted.

See [docs/MASTER_PROJECT_MAP.md](docs/MASTER_PROJECT_MAP.md) for the complete map. **Peer model exchange and testing on our held-out Dataset A have been completed** (9 Nov 2026); the classmate's independent Dataset B images/our-model-on-B results are still pending. Git tags, oral defense, and final instructor acceptance remain unverified. See [cross-dataset report](reports/cross_dataset_evaluation_2026-11-09.md).


## Submission-ready navigation (9 November 2026)

- [Submission manifest and honest open items](docs/FINAL_SUBMISSION_MANIFEST_2026-11-09.md)
- [Four-slide speaking script](presentation/Final_4_Slide_Script.md)
- [Model card of selected YOLOv8n classifier](reports/model_card.md)
- [Our model package for peer](https://drive.google.com/file/d/1z6TeWfS5vy6tc0PDvZFl_3F2CIyvzINk/view)
- [Peer model tested on our Test, complete evidence](https://drive.google.com/file/d/1CsRe-t3hDA8IH1ar4pQs8-zKg16JtKTF/view)
- [Final submission Drive staging folder](https://drive.google.com/drive/folders/16ioAY5oTkkspCM5iKLnHtJ3XEbKKb_ml)

**Compliance note:** the Tasks 2–4 protocol differences are preserved, not corrected by editing history. Tags and teacher submission cannot be claimed completed. The Task 5A notebook is a reproducible workflow, not an uploaded executed Colab notebook.

## Optional Task 5B — completed validation-only sensitivity analysis (9 November 2026)

[Threshold sweep and cost memo](reports/bonus_threshold_memo.md) | [precision–recall curve](results/bonus_threshold_pr_curve.svg) | [sweep CSV](results/bonus_threshold_sweep_val.csv). On **validation only**, P(bad) threshold 0.10 achieved 270/270 compared with 268/270 at 0.50. This **does not change** historical 0.50-threshold test evidence and is not a verified deployment setting.


## 9 November 2026 — audited submission additions

- [Submission readiness with unresolved blockers](docs/FINAL_HANDIN_STATUS_2026-11-09.md)
- [Model card](reports/model_card.md); [partial peer audit](reports/bonus_audit_note.md)
- [Optional validation-only threshold sweep](reports/bonus_threshold_memo.md) (no test-set threshold tuning; historical threshold remains 0.5)
- [Four-model CSV regenerated from source results.json](results/comparison_from_json.csv) and [generator](scripts/collect_results.py)
- [New instructor-named YOLO training launcher](scripts/train_yolo_quality.py) with 50/32/5/640 configuration; no new training claimed
- [Task 1 Word artifact](https://drive.google.com/file/d/1bbrB9GnlR-xrOtEAjCMAp_h5EUQbSOJC/view) converted from 19 October memo
- [Peer-on-our-Test evaluation](reports/cross_dataset_evaluation_2026-11-09.md) (opposite direction still pending)


## Current 152-point audit — 9 November 2026

- [Updated Markdown checklist](FULL_CHECKLIST_CURRENT_2026-11-09.md) — verified **85 green / 22 yellow / 45 red**, counting optional bonus tasks.
- [Updated ten-page PDF in Drive](https://drive.google.com/file/d/1WcQMktdHFhZoAA6aFcGUDLVGKWnFhD60/view); [Markdown copy in Drive](https://drive.google.com/file/d/1QVYsMvIponyR7Tel9B8dKKsrSjkY14ey/view). The 8 Nov snapshot remains intact.
- [Full zipped submission REVIEW package including FOUR ORIGINAL CHECKPOINTS (SHA-256 verified), the Task 5A checkpoint package, Word memo, four slides, experiment evidence and scripts](https://drive.google.com/file/d/1l6LKs3tCjIoetjxnl5uklINYCaLteAO0/view). **Prepared for review, NOT submitted or accepted.**


## Optional Task 5C — train-only glare augmentation (9 November 2026)

**Executed:** two matched continued-training arms initialized from the **same original YOLOv8n** checkpoint, on the original 1,260/270/270 source-grouped images. The control (no new augmentation) scored **268/270** on held-out Test (macro-F1 **0.9926**); targeted stronger glare on BAD training images scored **267/270** (macro-F1 **0.9889**). Both achieved glare recall **26/28**; the glare arm gained no additional glare recall and missed one more occluded image. These are real CPU training/evaluation results, not a claimed improvement. Same seed 2473986; max 15 epochs, actual 5, best epoch 1, batch 16, patience 4, RGB 224; **not full original-ImageNet restart nor exact later 50/32/5/640 teacher protocol**.

- [Task 5C technical report](reports/bonus_augmentation.md) | [new comparison rows](results/bonus_augmentation_comparison.csv) | [per-defect deltas](results/bonus_augmentation_per_defect_delta.csv) | [training and verification scripts](scripts/task5c_glare_train.py).
- [Complete original checkpoint + two new trained state_dicts, plots, SHA-verified manifest, histories and reproducibility package](https://drive.google.com/file/d/1HyxDmT2yJ7xX_wN4R76b0F8jREVrqfjM/view)
- [Requirement-status addendum](docs/TASK5C_2026-11-09_AUDIT.md).

## Task 5D — complete real ONNX/270-image parity test (9 Nov)

Both source PyTorch and ONNX scored **267/270 (98.89% accuracy; macro-F1 0.9889)** on identical source-hash-verified, original private Test images, **0/270 differing predictions and zero accuracy drop**. Same host forward-only CPU mean: **6.104ms PT / 2.451ms ONNX**, with 20 warm-ups and 100 timed calls per backend. ONNX is larger (**5.501 MiB vs PT 2.826 MiB**). End-to-end and edge-device performance remain unmeasured. [Real paired-Test ONNX+CSV+JSON+source ZIP](https://drive.google.com/file/d/1zIEXnesFzi4JPrzxuNRbYaemqEUEbQ0w/view) | [final measured memo](reports/bonus_deployment_memo.md).

## Task 5E — literal peer README inference reproduced (9 Nov)

Original classmate `predict.py` was run unchanged using Ultralytics on **our** original 270-image Test. Result **211/270 (78.15%)** and 0/270 per-image class discrepancies versus the earlier adapter-based evaluation; exit code **0**. Their original own-domain test images/training remain unavailable, so their reported original score was **not** independently reproduced. [Original-script log and metrics ZIP](https://drive.google.com/file/d/1BmdQlk-aY5e97KGm7phdXk8P2pN6Aw2i/view) | [peer audit report](reports/bonus_audit_note.md) | [own model card](reports/model_card.md).

**Unresolved elsewhere:** reverse-direction exchange (our model on classmate Dataset B), historical training-protocol deviations, Week7/8 Git tags, original executed Colab training notebook, official teacher acceptance and oral defense. The dated 152-check audit snapshot has not been silently rewritten.


## Final 9 November evidence audit and Git tags

- [Recounted 152-item checklist](docs/FULL_CHECKLIST_CURRENT_2026-11-09.md): **100 green / 25 yellow / 27 red**, after Tasks 5C, 5D, 5E and tag verification. Older snapshot totals are historical.
- [Verified annotated Week 7 and Week 8 tags](docs/TAGGING.md) created on the authentic October 25 commit. Both phases were originally committed together, so both tags point to one SHA (not backdated).
- [Task 5D measured results add-on](results/task5d_deployment_addon.json), preserving original result files.
- Still missing: classmate's independent Dataset B reciprocal results, exact instructor protocol matching for original Tasks 2-4, original executed Colab session evidence, in-class defense, and formal teacher acceptance.


## Final consolidated academic review archive — 9 November 2026

**Canonical, checksum-verified 78,526,618-byte package:** [Lab06_COMPLETE_SUBMISSION_REVIEW_FINAL_2026-11-09.zip](https://drive.google.com/file/d/1ZAp0JdWW_DruPzZrG9ll96Z-Idjr1-4D/view). ZIP SHA-256: `47962e1536ff6ce5fad1bf1074cb9dcfea6612b5ce674b7fca3155fef0069dd1`. It contains all four original model checkpoints, the executed Task 5A, 5C, 5D and 5E evidence packages, Task 1 Word memo, four-slide presentation, latest source/results snapshot and a 24-file SHA-256 manifest. The nested source snapshot is from commit `b5e2f5c2937dcd3fee249b5eaa13a78931443b0f`; later link updates to this file are navigational only.

**Final 152-point checklist:** [PDF](https://drive.google.com/file/d/1WcQMktdHFhZoAA6aFcGUDLVGKWnFhD60/view), [Markdown](https://drive.google.com/file/d/1QVYsMvIponyR7Tel9B8dKKsrSjkY14ey/view): **100 completed / 25 partial / 27 open**. Full backup in the ZIP. Original Dataset A images (about 399 MB) are referenced but not bundled.

**Open dependencies:** classmate's independent Dataset B reciprocal evaluation, original training settings (15/16/4/224 vs later 50/32/5/640), original executed Colab training notebook verification, and the instructor's oral presentation / acknowledgment. Two verified Week7/8 tags point to one authentic Oct 25 commit; no backdating. **Archive creation is not formal submission or acceptance.**
