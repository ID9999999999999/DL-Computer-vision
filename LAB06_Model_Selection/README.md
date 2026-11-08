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
